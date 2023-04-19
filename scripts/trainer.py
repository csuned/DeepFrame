import os, sys
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

src_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/src")
data_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/data")
utils_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/utils")
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)

from cnns import UNet
from loader import StatelessLoader

class MyDeepFrame(nn.Module):
    def __init__(self, n_input, nf, n_output, data_train, dropout_rate=0.0, bilinear=True, batch_size=32, learning_rate=0.0001, device='cpu'):
        super(MyDeepFrame, self).__init__()
        self.unet = UNet(n_input, nf, n_output, dropout_rate, bilinear)
        self.optimizer = pt.optim.Adam(self.unet.parameters(), lr=learning_rate, betas=(0.5, 0.999),
                                          weight_decay=0)
        self.loss = nn.L1Loss()
        #self.loss = nn.MSELoss()
        loader = StatelessLoader(data_train)
        self.dataloader = DataLoader(loader, batch_size=batch_size, shuffle=True, drop_last=True)
        self.device = device
        print(self.unet)

    def train(self, epochs=100):
        losstrain = []
        for epoch in range(epochs):
            for i, x in enumerate(self.dataloader):
                d_in = x['input'].to(self.device)
                d_out = x['output'].to(self.device)
                self.optimizer.zero_grad()
                recon_x = self.unet(d_in)
                loss = self.loss(recon_x, d_out)
                loss.backward()
                self.optimizer.step()
                losstrain.append(loss.item())
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, epochs, i + 1, len(self.dataloader), loss.item()))
        pt.save(self.unet.state_dict(), 'unet.pth')
        return losstrain

    def test(self, data_test):
        self.unet.eval()
        out_video = []
        for i in range(data_test['x_in'].shape[0]):
            d_in = data_test['x_in'][i, :, :, :].unsqueeze(0).to(self.device)
            d_out = self.unet(d_in)
            out_video.append(d_out.cpu().detach())
        return out_video #return tensor for video reconstruction