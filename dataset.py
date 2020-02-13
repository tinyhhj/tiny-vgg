import os
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataSet(Dataset):
    def __init__(self,path,mode):
        self.mode = mode
        self.path = path
        if not os.path.exists(path):
            raise ValueError('path is not exists')
        self.dict = {}
        for file in os.listdir(path):
            if '_batch' in file:
                data = 'data'
                label = 'label'
                if 'test' in file:
                    data = 'test_data'
                    label = 'test_label'
                with open(os.path.join(path,file), 'rb') as f:
                    # we need only data and label
                    p = pickle.load(f, encoding='bytes')
                    self.dict[data] = p[b'data'] if self.dict.get(data,None) is None else np.concatenate((self.dict[data],p[b'data']))
                    self.dict[label] = np.array(p[b'labels']) if self.dict.get(label,None) is None else np.concatenate(
                        (self.dict[label], p[b'labels']))

    def __len__(self):
        return len(self.dict)
    def __getitem__(self, index):
        if self.mode == 'test':
            return self.dict['test_data'][index] , self.dict['test_label'][index]
        return self.dict['data'][index], self.dict['label'][index]




