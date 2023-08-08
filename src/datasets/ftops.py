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
    #print("data[0,:18]: ", data[0,:18])
    # Get maximum value of data[:,:18]
    data_maxs = data[:,:18].max(axis=0)
    #print("data_maxs: ", data_maxs)
    #print("data[6,:18]: ", data[6,:])

    data_ids = torch.nn.functional.one_hot(data[:,:18].long()).float()
    #print("data_ids.shape: ", data_ids.shape)
    #print("data_ids: ", data_ids[0,:,:])

    # MET data
    data_aux = data[:,18:20]

    # Particle momenta in E, pt, eta, phi
    data_momenta = data[:, 20:].reshape(data.shape[0], 18, 4)
    #print("data_momenta.shape: ", data_momenta.shape)

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
    #print("data_ids_transpose.shape: ", data_ids.shape)
    data_momenta = torch.transpose(data_momenta, 1, 2)
    #print("data_momenta_transpose.shape: ", data_momenta.shape)
    data_four_vec = torch.transpose(data_four_vec, 1, 2)

    # Tokens are concat of momenta and ids
    data_tokens = torch.cat((data_momenta, data_ids), dim=1)
    #print("data_momenta_and_ids.shape: ", data_tokens.shape)

    # Generate padding mask
    data_mask = (data_momenta[:,0,:] != 0.).unsqueeze(1)

    return data_aux, data_tokens, data_four_vec, data_id_int, data_mask


class FTOPS():
 
    def __init__(self, root, mode='train', net_name='ftops_Mlp'):

        if net_name == 'ftops_Mlp':
            self.mode = mode
    
            #root = '/lustre/ific.uv.es/ml/ific005/projects/4tops/data/h5/4top_data_unsupervised/4tops_fcn.h5'
            print("Input file: ", root)
            hf = h5py.File(root, 'r')
    
            train_data = np.array(hf.get('X_train'), dtype=np.float32)
    
            #split data in nobj, reg
            num_objects = train_data[:,:6]
            reg = train_data[:,6:] 
    
            scaler = StandardScaler()
    
            scaler.fit(reg)
            reg_normalized = scaler.transform(reg)
    
            if self.mode == 'train':
                self.train_data = np.concatenate((num_objects, reg_normalized), axis=1)
                self.train_labels = np.array(hf.get('Y_train'))
    
            if self.mode == 'validation':
                val_data = np.array(hf.get('X_val'), dtype=np.float32)
                num_objects = val_data[:,:6]
                reg = val_data[:,6:]
                reg_normalized = scaler.transform(reg)
                self.val_data = np.concatenate((num_objects, reg_normalized), axis=1)
                self.val_labels = np.array(hf.get('y_val'))
            
            if self.mode == 'test':
                test_data = np.array(hf.get('X_test'), dtype=np.float32)
                num_objects = test_data[:,:6]
                reg = test_data[:,6:]
                reg_normalized = scaler.transform(reg)
                self.test_data = np.concatenate((num_objects, reg_normalized), axis=1)
                self.test_labels = np.array(hf.get('y_test'))

        if net_name == 'ftops_Transformer':
            self.mode = mode
    
            #root = '/lustre/ific.uv.es/ml/ific005/projects/4tops/data/h5/4top_data_unsupervised/4tops.h5'
            #root = '/lhome/ific/a/adruji/DarkMachines/unsupervised/Deep_SVDD_PTransf/data/4tops.h5'
            print("Input file: ", root)
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

        if net_name == 'ftops_ParticleNET':
            self.mode = mode
    
            #root = '/lustre/ific.uv.es/ml/ific005/projects/4tops/data/h5/4top_data_unsupervised/4tops.h5'
            #root = '/lhome/ific/a/adruji/DarkMachines/unsupervised/Deep_SVDD_PTransf/data/4tops.h5'
            print("Input file: ", root)
            hf = h5py.File(root, 'r')
    
            if self.mode == 'train':
                train_data = torch.tensor(np.array(hf.get('X_train')), dtype=torch.float32)
                self.train_labels = torch.tensor(hf.get('Y_train'), dtype=torch.long)
                data_aux_train, data_tokens_train, data_momenta_train, _, data_mask_train = convert_data(train_data)
                self.train_data = torch.utils.data.TensorDataset(data_aux_train, data_tokens_train, data_momenta_train, data_mask_train, self.train_labels)
            if self.mode == 'validation':
                val_data = torch.tensor(np.array(hf.get('X_val')), dtype=torch.float32)
                self.val_labels = torch.tensor(hf.get('y_val'), dtype=torch.long)
                data_aux_val, data_tokens_val, data_momenta_val, _, data_mask_val = convert_data(val_data)
                self.val_data = torch.utils.data.TensorDataset(data_aux_val,   data_tokens_val,   data_momenta_val, data_mask_val, self.val_labels)
            if self.mode == 'test':
                test_data = torch.tensor(np.array(hf.get('X_test')), dtype=torch.float32)
                self.test_labels = torch.tensor(hf.get('y_test'), dtype=torch.long)
                data_aux_test, data_tokens_test, data_momenta_test, _, data_mask_test = convert_data(test_data)
                self.test_data = torch.utils.data.TensorDataset(data_aux_test, data_tokens_test, data_momenta_test, data_mask_test, self.test_labels)

  
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
  
    def __init__(self, root: str, normal_class=0, net_name='ftops_Mlp'): 
        super().__init__(root)
    
        self.train_set = FTOPS(root, mode='train', net_name=net_name)
        self.val_set = FTOPS(root, mode='validation', net_name=net_name)
        self.test_set = FTOPS(root, mode='test', net_name=net_name)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, val_loader, test_loader


