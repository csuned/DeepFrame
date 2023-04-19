import os
import sys
import torchvision as tv
import torch
import torch.functional as F


SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, SRC_DIR)

def read_video(v_path, start_pts=0, end_pts=None, pts_unit='sec'):
    """
    Read a video file and return a tensor of shape (T, C, H, W)
    """
    if pts_unit == 'sec':
        start_pts = int(start_pts * 30)
        if end_pts:
            end_pts = int(end_pts * 30)
    elif pts_unit == 'frame':
        pass
    else:
        raise ValueError(f"Invalid pts_unit: {pts_unit}")

    try:
        v_tensor = tv.io.read_video(v_path, start_pts=start_pts, end_pts=end_pts, pts_unit=pts_unit)[0]
    except:
        raise ValueError(f"Invalid video path: {v_path}")
    return v_tensor

def save_video(v_tensor, v_path, fps=30):
    """
    Save a tensor of shape (T, C, H, W) as a video file
    """
    try:
        tv.io.write_video(v_path, v_tensor, fps=fps)
    except:
        raise ValueError(f"Invalid video path: {v_path}")

def split_data(x, ratio=0.8):
    """
    Split the data into training and validation set
    """
    x_train = x[:int(x.shape[0]*ratio)]
    x_val = x[int(x.shape[0]*ratio):]
    return x_train, x_val

def stateless_dataset(x, frame_in, frame_out, step=1, device='cpu'):
    """
    Convert the data into training data
    Stateless version
    """
    x = torch.from_numpy(x)
    x_train = {}
    x_in = torch.zeros((x.shape[0]-frame_out-frame_in)//(frame_in*step), frame_in*x.shape[3], x.shape[1], x.shape[2])
    x_out = torch.zeros((x.shape[0]-frame_out-frame_in)//(frame_in*step), frame_out*x.shape[3], x.shape[1], x.shape[2])
    channel_num = x.shape[3]
    batch_num = (x.shape[0]-frame_out-frame_in)//(frame_in*step)
    for i in range(batch_num):
        for j in range(channel_num):
            x_in[i, j*frame_in:(j+1)*frame_in, :, :] = x[i*step:i*step+frame_in, :, :, j]
            x_out[i, j*frame_out:(j+1)*frame_out, :, :] = x[i*step+frame_in:i*step+frame_in+frame_out, :, :, j]
    x_train['x_in'] = x_in.to(device)
    x_train['x_out'] = x_out.to(device)
    return x_train

def dataset2video(x):
    """
    Convert the data into video
    """
    x_video = torch.zeros((x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_video[i, :, :, j] = x[i, j, :, :]
    frame_num = x.shape[0]*x.shape[1]//3 # 3 channels per frame (RGB), framenum=batch_size*channel_num/3
    final_video = torch.zeros((frame_num, x.shape[2], x.shape[3], 3))
    bundle_num = x.shape[1]//3
    for i in range(x_video.shape[0]):
        for j in range(bundle_num):
            final_video[i*bundle_num+j, :, :, 0] = x_video[i, :, :, j*3]
            final_video[i*bundle_num+j, :, :, 1] = x_video[i, :, :, j*3+1]
            final_video[i*bundle_num+j, :, :, 2] = x_video[i, :, :, j*3+2]
    return final_video




