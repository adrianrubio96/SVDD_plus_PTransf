import h5py
import numpy as np
from tqdm import tqdm 
import torch   
from torch.utils.data import DataLoader
from base.base_dataset import BaseADDataset

import sys

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

def convert_data(data):
    MeV_to_GeV = 1e-3

    n_data = data.shape[0]

    # One-hot encoding for ids
    data_ids = torch.nn.functional.one_hot(data[:,:18].long())

    # MET data
    data_aux = data[:,18:20]

    # Particle momenta in E, pt, eta, phi
    data_momenta = data[:, 20:].reshape(data.shape[0], 18, 4)

    # Particle data for interaction term
    data_four_vec = torch.zeros_like(data_momenta)
    data_four_vec[:,:,0] = MeV_to_GeV*torch.exp(data_momenta[:,:,1])*torch.cos( data_momenta[:,:,3])
    data_four_vec[:,:,1] = MeV_to_GeV*torch.exp(data_momenta[:,:,1])*torch.sin( data_momenta[:,:,3])
    data_four_vec[:,:,2] = MeV_to_GeV*torch.exp(data_momenta[:,:,1])*torch.sinh(data_momenta[:,:,2])
    data_four_vec[:,:,3] = MeV_to_GeV*torch.exp(data_momenta[:,:,0])

    # Particle ids for interaction term
    data_id_int = data[:,:18].long()

    # Set back to zero
    data_four_vec[data_momenta == 0.] = 0.

    # Transpose 
    data_ids = torch.transpose(data_ids, 1, 2)
    data_momenta = torch.transpose(data_momenta, 1, 2)
    data_four_vec = torch.transpose(data_four_vec, 1, 2)

    # Tokens are concat of momenta and ids
    data_tokens = torch.cat((data_momenta, data_ids), dim=1)

    # Generate padding mask
    data_mask = (data_momenta[:,0,:] != 0.).unsqueeze(1)

    return data_aux, data_tokens, data_four_vec, data_id_int, data_mask


class FTOPS():
 
    def __init__(self, root, mode='train'):

        self.mode = mode

        #root = '/lustre/ific.uv.es/ml/ific005/projects/4tops/data/h5/4top_data_unsupervised/4tops.h5'
        root = '/lhome/ific/a/adruji/DarkMachines/unsupervised/Deep_SVDD_PTransf/data/4tops.h5'

        hf = h5py.File(root, 'r')

        if self.mode == 'train':
         train_data = torch.tensor(np.array(hf.get('X_train')), dtype=torch.float32)
         self.train_labels = torch.tensor(hf.get('Y_train'), dtype=torch.long)
         data_aux_train, data_tokens_train, data_momenta_train, data_id_int_train, data_mask_train = convert_data(train_data)
         self.train_data = torch.utils.data.TensorDataset(data_aux_train, data_tokens_train, data_momenta_train, data_id_int_train, data_mask_train)
        if self.mode == 'validation':
         val_data = torch.tensor(np.array(hf.get('X_val')), dtype=torch.float32)
         self.val_labels = torch.tensor(hf.get('y_val'), dtype=torch.long)
         data_aux_val, data_tokens_val, data_momenta_val, data_id_int_val, data_mask_val = convert_data(val_data)
         self.val_data = torch.utils.data.TensorDataset(data_aux_val,   data_tokens_val,   data_momenta_val,   data_id_int_val,   data_mask_val)
        if self.mode == 'test':
         test_data = torch.tensor(np.array(hf.get('X_test')), dtype=torch.float32)
         self.test_labels = torch.tensor(hf.get('y_test'), dtype=torch.long)
         data_aux_test, data_tokens_test, data_momenta_test, data_id_int_test, data_mask_test = convert_data(test_data)
         self.test_data = torch.utils.data.TensorDataset(data_aux_test, data_tokens_test, data_momenta_test, data_id_int_test, data_mask_test)

  
    def __len__(self):
        if self.mode == 'train':
         return len(self.train_labels)
        if self.mode == 'validation':        
         return len(self.val_labels)
        if self.mode == 'test':
         return len(self.test_labels)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.mode == 'train':       
            data, target = self.train_data[index], self.train_labels[index]
        if self.mode == 'validation':  
            data, target = self.val_data[index], self.val_labels[index]
        if self.mode == 'test':
            data, target = self.test_data[index], self.test_labels[index]

        return data, target, index  # only line changed


class FTOPS_Dataset(BaseADDataset):
  
    def __init__(self, root: str, normal_class=0): 
     super().__init__(root)
 
     self.train_set = FTOPS(root, mode='train')
     self.val_set = FTOPS(root, mode='validation')
     self.test_set = FTOPS(root, mode='test')

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, val_loader, test_loader


