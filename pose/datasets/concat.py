import numpy as np
import torch
import torch.utils.data as data

class Concat(data.Dataset):

    def __init__(self, datasets, ratio = None):
        self.datasets = datasets
        self.ratio = ratio
        if self.ratio is not None:
            self.ratio = self.ratio / np.sum(self.ratio)
        self.reset()

    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i > 0:
                    index -= self.offsets[i-1]
                return self.datasets[i][self.choice_list[i][index]]
        raise IndexError(f'{index} exceeds {self.length}')

    def __len__(self):
        return self.length

    def reset(self):
        self.lengths = [len(d) for d in self.datasets]

        if sum(self.lengths) == 0:
            self.length = 0
            return

        if self.ratio is None:
            
            self.offsets = np.cumsum(self.lengths)
            self.length = np.sum(self.lengths)
            self.choice_list = []
            for i in range(len(self.datasets)):
                self.choice_list.append(list(range(self.lengths[i])))
                print('No. images of dataset {} : {}'.format(i+1, self.lengths[i]))

        else:
            assert(len(self.datasets) == len(self.ratio))
            benchmark = min(self.lengths / self.ratio) #the subset that depletes first
            self.valid_lengths = (self.ratio * benchmark).astype(np.int32)
            self.choice_list = []
            for i in range(len(self.datasets)):
                self.choice_list.append(np.random.choice(self.lengths[i], self.valid_lengths[i], replace=False))
                print('No. images of dataset {} : {}'.format(i+1, self.valid_lengths[i]))

            self.offsets = np.cumsum(self.valid_lengths)
            self.length = np.sum(self.valid_lengths)

        print('merging {} datasets, total No. images: {}'.format(len(self.datasets), self.length))

class Concat_w_class(data.Dataset): #concatenated dataset with class label, only support concating 2 datasets(sourse and target)

    def __init__(self, datasets):
        self.datasets = datasets
        self.reset()

    def __getitem__(self, index):
        return self.datasets[0][self.choice_list[0][index]], self.datasets[1][self.choice_list[1][index]]

    def __len__(self):
        return self.length

    def reset(self):
        self.lengths = [len(d) for d in self.datasets]

        if sum(self.lengths) == 0:
            self.length = 0
            return

        self.valid_length = min(self.lengths)#the subset that depletes first
        self.choice_list = []
        for i in range(len(self.datasets)):
            self.choice_list.append(np.random.choice(self.lengths[i], self.valid_length, replace=False))
            print('No. images of dataset {} : {}'.format(i+1, self.valid_length))

        self.length = self.valid_length

        print('merging {} datasets, total No. images: {}'.format(len(self.datasets), self.length))

            
            
