import torch
import h5py
import scipy.io
import numpy as np
from scipy.signal import butter
from scipy.sparse import spdiags
from post_process import detrend
from background_aug import generate_scaled_face

class rPPG_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, frame_depth=10, fs=30, signal='pulse', input_channel='both', img_size=36,
                 num_gpus=4, background_aug=False, background_path=[]):
        self.dataset_path = dataset_path
        self.frame_depth = frame_depth
        self.fs = fs
        self.signal = signal
        self.input_channel = input_channel
        self.img_size = img_size
        self.num_gpus = num_gpus
        self.background_aug = background_aug
        self.background_path = background_path

    def __len__(self):
        return len(self.dataset_path)

    def data_load_func(self, path):
        # print('path: ', path)
        try:
            f1 = h5py.File(path, 'r')
        except OSError:
            f1 = scipy.io.loadmat(path)

        if f1["dXsub"].shape[0] == 6:
            dXsub = np.transpose(np.array(f1["dXsub"]), [3, 0, 2, 1])
        else:
            dXsub = np.array(f1["dXsub"])
            if dXsub.shape[1] != 6:
                dXsub = np.transpose(np.array(f1["dXsub"]), [0, 3, 1, 2])
        # Check Task 1 - 6 in AFRL:
        if 'AFRL' in path:
            chunk_name = path.split('/')[-1].split('.')[0]
            loc_T = chunk_name.find('T')
            loc_V = chunk_name.find('V')
            taskID = int(chunk_name[loc_T + 1:loc_V])
            if taskID <= 6 and self.background_aug:
                dXsub = generate_scaled_face(dXsub, self.img_size, self.img_size, target_dim_list=[32, 40, 48, 56, 64,72],
                                        background_path=self.background_path)
        return f1, dXsub

    def __getitem__(self, index):
        temp_path = self.dataset_path[index]
        f1, output = self.data_load_func(temp_path)
        label_pulse = np.array(f1["dysub"])
        # label_pulse = (label_pulse - np.min(label_pulse)) / (np.max(label_pulse) - np.min(label_pulse))
        # label_pulse = np.float32(np.expand_dims(label_pulse, axis=1))
        if self.signal == 'both':
            label_resp = np.array(f1["drsub"])
            label_resp = np.float32(np.expand_dims(label_resp, axis=1))
            label = (label_pulse, label_resp)
        else:
            label = label_pulse
        if output.shape[0] != 180:
            idea_multiplier = self.frame_depth * self.num_gpus
            output_len = (output.shape[0] // idea_multiplier) * idea_multiplier
            label_len = (label.shape[0] // idea_multiplier) * idea_multiplier
            output = output[:output_len]
            label = label[:label_len]
        # Average the frame
        if self.input_channel == 'both':
            motion_data = output[:, :3, :, :]
            apperance_data = output[:, 3:, :, :]
            apperance_data = np.reshape(apperance_data, (
            int(apperance_data.shape[0] / self.frame_depth), self.frame_depth, 3, self.img_size, self.img_size))
            apperance_data = np.average(apperance_data, axis=1)
            apperance_data = np.repeat(apperance_data[:, np.newaxis, :, :, :], self.frame_depth, axis=1)
            apperance_data = np.reshape(apperance_data, (apperance_data.shape[0] * apperance_data.shape[1],
                                                         apperance_data.shape[2], apperance_data.shape[3],
                                                         apperance_data.shape[4]))
            output = np.concatenate((motion_data, apperance_data), axis=1)
        elif self.input_channel == 'diff':
            output = output[:, :3, :, :]
        elif self.input_channel == 'raw':
            output = output[:, 3:, :, :]
        else:
            raise Exception('Please use keyboard diff, raw or both')
        output = np.float32(output)
        label = np.float32(label)
        return (output, label)


# This is for Swin-3D

class rPPG_Dataset_Swin(torch.utils.data.Dataset):
    def __init__(self, dataset_path, frame_depth=10, fs=30, signal='pulse', swin_input_channel='diff',
                 img_size=36, num_gpus=4):
        self.dataset_path = dataset_path
        self.frame_depth = frame_depth
        self.fs = fs
        self.signal = signal
        self.swin_input_channel = swin_input_channel
        self.img_size = img_size
        self.num_gpus = num_gpus

    def __len__(self):
        return len(self.dataset_path)

    def data_load_func(self, path):
        try:
            f1 = h5py.File(path, 'r')
        except OSError:
            f1 = scipy.io.loadmat(path)

        if f1["dXsub"].shape[0] == 6:
            dXsub = np.transpose(np.array(f1["dXsub"]), [3, 0, 2, 1])
        else:
            dXsub = np.array(f1["dXsub"])
        return f1, dXsub

    def __getitem__(self, index):
        temp_path = self.dataset_path[index]
        f1, output = self.data_load_func(temp_path)
        label_pulse = np.array(f1["dysub"])
        label_pulse = np.float32(np.expand_dims(label_pulse, axis=1))
        if self.signal == 'both':
            label_resp = np.array(f1["drsub"])
            label_resp = np.float32(np.expand_dims(label_resp, axis=1))
            label = (label_pulse, label_resp)
        else:
            label = label_pulse

        # Average the frame
        if self.swin_input_channel == 'diff':
            data = output[:, :3, :, :]
            output = np.reshape(data, (-1, self.frame_depth, 3, self.img_size, self.img_size))
        elif self.swin_input_channel == 'raw':
            data = output[:, 3:, :, :]
            output = np.reshape(data, (-1, self.frame_depth, 3, self.img_size, self.img_size))
        elif self.swin_input_channel == 'both':
            data = output
            output = np.reshape(data, (-1, self.frame_depth, 6, self.img_size, self.img_size))
        else:
            raise Exception('Please use keyboard diff, raw or both')
        output = np.swapaxes(output, 1, 2)
        output = np.float32(output)
        label = np.float32(label)
        return (output, label)
