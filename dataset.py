import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self,path,mode):
        self.mode = mode
        self.datakey = 'data'
        self.labelkey = 'label'
        if self.mode == 'test':
            self.datakey = 'test_data'
            self.labelkey = 'test_label'

        self.path = path
        if not os.path.exists(path):
            raise ValueError('path is not exists')
        self.data = []
        self.target = []
        self.labelnames = []
        for file in os.listdir(path):
            if not 'batch' in file:
                continue
            if self.mode == 'test' and not self.mode in file:
                # test인경우 test셋만 로딩
                continue
            with open(os.path.join(path,file), 'rb') as f:
                # we need only data and label
                p = pickle.load(f, encoding='latin1')
                if file == 'batches.meta':
                    self.labelnames.extend(p['label_names'])
                    continue
                self.data.append(p['data'])
                self.target.extend(p['labels'])
        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        self.data = self.data.transpose((0,2,3,1)) # HWC



    def __len__(self):
        return len(self.target)
    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]), torch.tensor(self.target[index])




