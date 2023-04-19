import os
import sys
import pdb
import numpy as np
import torchvision as tv
import torch
import torch.functional as F

src_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/src")
data_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/data")
utils_dir = os.path.expanduser("~/PycharmProjects/DeepFrame/utils")
sys.path.insert(0, src_dir)
sys.path.insert(0, utils_dir)

from preprocess import read_video, save_video, split_data, stateless_dataset, dataset2video
from metrics import compare_images_series
from myplot import plot_series
from trainer import MyDeepFrame

'''
Hyperparameters
'''
learning_rate = 1e-3
batch_size=8
num_epochs=20
gpu_id = 0
device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and gpu_id in [0,1]) else "cpu")
frame_in = 10
frame_out = 2
nf = 4
bilinear = True
n_input = frame_in*3 #3 channels for RGB in CNN
n_output = frame_out*3
'''
Load&Process data
'''
testvideo_raw = read_video(f"{data_dir}/susan.mp4", start_pts=0, end_pts=2, pts_unit='sec')
testvideo_raw = testvideo_raw[:, :170,:,:] #Crop the video to 170 pixels
print(testvideo_raw.shape)
testvideo = testvideo_raw.numpy()
print(testvideo.max(), testvideo.min())

for color in range(3):
    s_mse_temp, s_ssim_temp=compare_images_series(testvideo[:-1, :, :, color], testvideo[1:, :, :, color])
    if color == 0:
        s_mse = np.array(s_mse_temp)
        s_ssim = np.array(s_ssim_temp)
    else:
        s_mse += np.array(s_mse_temp)
        s_ssim += np.array(s_ssim_temp)


s_mse = s_mse/s_mse.max() #Use nomralized MSE
s_ssim = s_ssim/3 #Averge of 3 channels

plot_series([s_mse, s_ssim], ['MSE', 'SSIM'])

data_train, data_val = split_data(testvideo, ratio=0.8)
data_train = stateless_dataset(data_train, frame_in, frame_out, step=1, device=device)
data_val = stateless_dataset(data_val, frame_in, frame_out, step=frame_out, device=device)

'''
Model Training
'''
prednet = MyDeepFrame(n_input, nf, n_output, data_train, dropout_rate=0.0, bilinear=bilinear, batch_size=batch_size, learning_rate=learning_rate, device=device)
prednet.to(device)
losstrain = prednet.train(epochs=num_epochs)

plot_series([losstrain,], ['Loss',])

'''
Model Testing
'''
video_out = prednet.test(data_val)
video_out = torch.cat(video_out, dim=0)
print(video_out.shape)
video_out = dataset2video(video_out)
save_video(video_out, f"{data_dir}/susan_out.mp4")