import h5py
import numpy as np
from tqdm import tqdm    
from torch.utils.data import DataLoader
from base.base_dataset import BaseADDataset

import sys

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler

class FTOPS():
 
    def __init__(self, root, mode='train'):

        self.mode = mode

        #root = '/lustre/ific.uv.es/ml/ific005/projects/4tops/data/h5/4top_data_unsupervised/4tops_fcn.h5'

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


