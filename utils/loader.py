import random
import torch as pt
from torch.utils.data import Dataset

class StatelessLoader(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset['x_in'].shape[0]

    def __getitem__(self, idx):
        item = {}
        item['input'] = self.dataset['x_in'][idx,:,:,:] #[batch, channel, height, width]
        item['output'] = self.dataset['x_out'][idx,:,:,:]
        return item

