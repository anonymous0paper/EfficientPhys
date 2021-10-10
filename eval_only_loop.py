import argparse
import json
import os
import glob

import numpy as np
import scipy.io
from scipy.signal import butter

from utils import get_nframe_video, read_from_txt, print_cuda_memory
from data_generator import rPPG_Dataset, rPPG_Dataset_Swin
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from model import TSCAN, MTTS_CAN, TSCAN_Single
from swin_transfomer import SwinTransformer3D
from post_process import calculate_metric, calculate_metric_per_video, calculate_metric_peak_per_video
from swin_transfomer_2d import SwinTransformer, SwinTransformer_Both
from swin_transfomer_2d_noTSM import SwinTransformerNoTSM
import time
from ppg_loss import ppg_loss
from collections import OrderedDict
import h5py
import csv
from utils import plot_signal
# time.sleep(7000)

torch.manual_seed(100)
np.random.seed(100)
torch.cuda.empty_cache()


# %%
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str,
                    default='/gscratch/ubicomp/xliu0/data3/mnt/',
                    help='Location for the dataset')  # default='/gscratch/scrubbed/xliu0/', help='Location for the dataset')
parser.add_argument('-mo', '--model', type=str,
                    default='tscan', help='tscan or mtts-can')
parser.add_argument('-tr_txt', '--train_txt', type=str, default='./filelists/Train.txt',
                    help='train file')  # ./filelists/AFRL/36/meta/train.txt', help='train file')
parser.add_argument('-ts_txt', '--test_txt', type=str, default='./filelists/Test.txt',
                    help='test file')  # ./filelists/AFRL/36/meta/test.txt', help='test file')
parser.add_argument('-a', '--nb_filters1', type=int, default=32,
                    help='number of convolutional filters to use')
parser.add_argument('-b', '--nb_filters2', type=int, default=64,
                    help='number of convolutional filters to use')
parser.add_argument('-c', '--dropout_rate1', type=float, default=0.25,
                    help='dropout rates')
parser.add_argument('-d', '--dropout_rate2', type=float, default=0.5,
                    help='dropout rates')
parser.add_argument('-e', '--nb_dense', type=int, default=128,
                    help='number of dense units')
parser.add_argument('-f', '--cv_split', type=int, default=0,
                    help='cv_split')
parser.add_argument('-g', '--nb_epoch', type=int, default=48,
                    help='nb_epoch')
parser.add_argument('-t', '--nb_task', type=int, default=12,
                    help='nb_task')
parser.add_argument('-x', '--batch_size', type=int, default=24,
                    help='batch')
parser.add_argument('-fd', '--frame_depth', type=int, default=20,
                    help='frame_depth for 3DCNN')
parser.add_argument('-save', '--save_all', type=int, default=0,
                    help='save all or not')
parser.add_argument('-shuf', '--shuffle', type=str, default=True,
                    help='shuffle samples')
parser.add_argument('-freq', '--fs', type=int, default=30,
                    help='shuffle samples')
parser.add_argument('--window_size', type=int, default=256,
                    help='window size for filtering and FFT')
parser.add_argument('--signal', type=str, default='pulse')
parser.add_argument('--swin_window_size', type=int, default=4)
parser.add_argument('--img_size', type=int, default=72)
parser.add_argument('--swin_2d_patch', type=int, default=3)
parser.add_argument('--swin_input_channel', type=str, default='diff')
parser.add_argument('--light', action='store_true', help='light swin')
parser.add_argument('--superlight', action='store_true', help='light swin')
parser.add_argument('--regular', action='store_true', help='Non-PPV2')
parser.add_argument('--debug_method', type=str, default='peak')
parser.add_argument('--error_cut', type=int, default=5)
parser.add_argument('--total_epoch', type=int, default=10)

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Evaluation


def eval(args, model_epoch, test_file, checkpoint_folder):
    # Reading Data
    args.test_file = test_file
    if args.test_file == 'UBFCPPV2Crop':
        args.test_txt = './filelists/UBFC/UBFC_72_all_video_PPV2.txt'
    elif args.test_file == 'UBFCPPV2NoCrop':
        args.test_txt = './filelists/UBFC/UBFC_72_all_video_PPV2_nocrop.txt'
    elif args.test_file == 'PUREPPV2Crop':
        args.test_txt = './filelists/PURE/PURE72x72PPV2.txt'
    elif args.test_file == 'PUREPPV2NoCrop':
        args.test_txt = './filelists/PURE/PURE72x72PPV2NoCrop.txt'
    elif args.test_file == 'MMSEPPV2':
        args.test_txt = './filelists/MMSE/MMSE72x72PPV2.txt'
        args.fs = 25
    elif args.test_file == 'UBFCCrop':
        args.test_txt = './filelists/UBFC/UBFC_72_all_video.txt'
    elif args.test_file == 'UBFCNoCrop':
        args.test_txt = './filelists/UBFC/UBFC_72_all_video_nocrop.txt'
    elif args.test_file == 'PURECrop':
        args.test_txt = './filelists/PURE/PURE72x72.txt'
    else:
        raise Exception('The dataset is not supported yet.')
    _, path_of_video_test = read_from_txt(args.train_txt, args.test_txt, args.data_dir)
    # num_gpus = 1
    # num_gpus = torch.cuda.device_count()
    # path_ts_len = (len(path_of_video_test) // num_gpus) * num_gpus
    # path_of_video_test = path_of_video_test[:path_ts_len]
    num_gpus = 1
    testing_dataset = rPPG_Dataset(path_of_video_test, frame_depth=args.frame_depth,
                                   signal=args.signal, img_size=args.img_size,
                                   input_channel=args.swin_input_channel, num_gpus=num_gpus,
                                   background_aug=False)
    ts_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)

    if args.swin_input_channel == 'diff' or args.swin_input_channel == 'raw':
        in_chans = 3
    else:
        in_chans = 6
    if args.model == 'MTTS-CAN':
        model = MTTS_CAN(frame_depth=args.frame_depth)
    elif args.model == 'swin':
        if args.img_size == 36:
            if args.swin_window_size == 9:
                dense_layer = 768
            elif args.swin_window_size == 4:
                dense_layer = 3072
            elif args.swin_window_size == 2:
                dense_layer = 6912
            else:
                raise Exception('Unsupported window size in Img size of 36')
        elif args.img_size == 72:
            if args.swin_window_size == 9:
                dense_layer = 768
            elif args.swin_window_size == 4:
                dense_layer = 6912
            elif args.swin_window_size == 2:
                dense_layer = 19200
            else:
                raise Exception('Unsupported window size in Img size of 72')
        else:
            raise Exception('Unsupported Img Size. Only supported 36 or 72 now')

        model = SwinTransformer3D(patch_size=(1, args.swin_window_size, args.swin_window_size), pretrained2d=False,
                                  dense_layer_number=dense_layer, in_chans=in_chans)
    elif args.model == 'TS-CAN':
        if args.swin_input_channel == 'both':
            model = TSCAN(frame_depth=args.frame_depth, img_size=args.img_size)
        else:
            model = TSCAN_Single(frame_depth=args.frame_depth, img_size=args.img_size, channel=args.swin_input_channel)
    elif args.model == 'swin-2d':
        if args.swin_input_channel == 'both':
            model = SwinTransformer_Both(img_size=args.img_size, patch_size=3,
                                         in_chans=6, window_size=3, depths=[2, 6, 2], frame_depth=args.frame_depth)
        else:
            if args.light:
                model = SwinTransformer(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3, window_size=3,
                                    depths=[1, 2, 1], channel=args.swin_input_channel, frame_depth=args.frame_depth)
            elif args.superlight:
                model = SwinTransformer(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3, window_size=3,
                                    depths=[1], channel=args.swin_input_channel, frame_depth=args.frame_depth)
            else:
                model = SwinTransformer(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3, window_size=3,
                                    depths=[2, 6, 2], channel=args.swin_input_channel, frame_depth=args.frame_depth)

    elif args.model == 'swin-2d-notsm':
        if args.swin_input_channel == 'raw':
            model = SwinTransformerNoTSM(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3,
                                         window_size=3, depths=[2, 6, 2],
                                         channel=args.swin_input_channel, frame_depth=args.frame_depth)
        else:
            raise Exception('Only raw frames are supported in swin-2d-notsm')
    else:
        raise Exception('Unsupported Model!')
    # model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise ValueError('Your training is not using GPU!')
    ##############################################################################################################
    pre_trained_path = str(os.path.join('./checkpoints', args.exp_name, args.exp_name + '_' + str(model_epoch)+'.pth'))
    checkpoint = torch.load(pre_trained_path)
    state_dict = checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    num_gpus = 1
    ##############################################################################################################
    # Evaluation
    with torch.no_grad():
        model.eval()
        final_preds = []
        final_labels = []
        final_HR = []
        final_HR0 = []
        final_HR_peak = []
        final_HR0_peak = []
        filename_csv = str(os.path.join(checkpoint_folder, 'Epoch_' + str(model_epoch) + '_summary.csv'))
        csvfile = open(filename_csv, 'w', newline='')
        fieldnames = ['filename', 'HR0-FFT', 'HR-FFT', 'MAE-FFT', 'HR0-Peak', 'HR-Peak', 'MAE-Peak']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, ts_data in enumerate(ts_dataloader):
            video_filename = path_of_video_test[i].split('/')[-1]
            ts_inputs, ts_labels = ts_data[0].to(device), ts_data[1].to(device)
            if args.model == 'swin':
                ts_inputs = ts_inputs.view(-1, in_chans, args.frame_depth, args.img_size, args.img_size)
                if args.swin_input_channel == 'raw':
                    padd_frames = torch.zeros(ts_inputs.shape[0], in_chans, 1, args.img_size, args.img_size)
                    ts_inputs = torch.cat((ts_inputs, padd_frames.to(device)), -3)
            else:
                ts_inputs = ts_inputs.view(-1, in_chans, args.img_size, args.img_size)
                if args.swin_input_channel == 'raw':
                    last_frame = torch.unsqueeze(ts_inputs[-1, :, :, :], 0).repeat(num_gpus, 1, 1, 1)
                    ts_inputs = torch.cat((ts_inputs, last_frame), 0)
            ts_labels = ts_labels.view(-1, 1)
            if args.model == 'TS-CAN' and args.swin_input_channel == 'both':
                if ts_inputs.shape[0] >= 1500:
                    ts_inputs_1 = ts_inputs[:1500]
                    ts_inputs_rest = ts_inputs[1500:]
                    ts_outputs = model(ts_inputs_1)
                    del ts_inputs_1
                    ts_outputs_rest = model(ts_inputs_rest)
                    ts_outputs = torch.cat((ts_outputs, ts_outputs_rest), 0)
                else:
                    ts_outputs = model(ts_inputs)
            else:
                ts_outputs = model(ts_inputs)
            ts_outputs_numpy = ts_outputs.cpu().numpy()
            ts_labels_numpy = ts_labels.cpu().numpy()
            HR0, HR = calculate_metric_per_video(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                                 fs=args.fs, bpFlag=True)
            HR0_peak, HR_peak, all_peaks, all_peaks0, filter_pred_signal, filter_label_signal = \
                calculate_metric_peak_per_video(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                                                window_size=args.window_size, fs=args.fs, bpFlag=True)
            # plot_signal(filter_pred_signal, filter_label_signal, video_filename, checkpoint_folder, 'pred', HR0, HR, HR0_peak,
            #             HR_peak, all_peaks, all_peaks0)
            final_HR.append(HR)
            final_HR0.append(HR0)
            final_HR_peak.append(HR_peak)
            final_HR0_peak.append(HR0_peak)
            final_preds.append(ts_outputs_numpy)
            final_labels.append(ts_labels_numpy)
            writer.writerow({'filename': video_filename, 'HR0-FFT': HR0, 'HR-FFT': HR, 'MAE-FFT': abs(HR0-HR),
                                 'HR0-Peak': HR0_peak, 'HR-Peak': HR_peak, 'MAE-Peak': abs(HR0_peak-HR_peak)})
        final_HR = np.array(final_HR)
        final_HR0 = np.array(final_HR0)
        final_HR_peak = np.array(final_HR_peak)
        final_HR0_peak = np.array(final_HR0_peak)
        print('----------------------------')
        print('FFT Metric')
        print('Avg MAE across subjects: ', np.mean(np.abs(final_HR - final_HR0)))
        print('Avg RMSE across subjects: ', np.sqrt(np.mean(np.square(final_HR - final_HR0))))
        print('Pearson FFT: ', abs(np.corrcoef(final_HR, final_HR0)[1, 0]))
        print('Std FFT', np.std(final_HR))
        print('----------------------------')
        print('Peak Detection Metric')
        print('Avg MAE across subjects: ', np.mean(np.abs(final_HR_peak - final_HR0_peak)))
        print('Avg RMSE across subjects: ', np.sqrt(np.mean(np.square(final_HR_peak - final_HR0_peak))))
        print('Pearson Peak: ', abs(np.corrcoef(final_HR_peak, final_HR0_peak)[1, 0]))
        print('Std Peak', np.std(final_HR_peak))
        print('----------------------------')
        csvfile.close()
        with open(str(os.path.join(checkpoint_folder, 'Epoch_' + str(model_epoch) + '_summary.txt')), "w") as text_file:
            text_file.write('----------------------------\n')
            text_file.write('FFT Metric' + '\n')
            text_file.write('Avg MAE across subjects: \n')
            text_file.write(str(np.mean(np.abs(final_HR - final_HR0)))+ '\n')
            text_file.write('Avg RMSE across subjects: \n')
            text_file.write(str(np.sqrt(np.mean(np.square(final_HR - final_HR0)))) + '\n')
            text_file.write('Pearson FFT: \n')
            text_file.write(str(abs(np.corrcoef(final_HR, final_HR0)[1, 0])) + '\n')
            text_file.write('Std FFT: \n')
            text_file.write(str(np.std(final_HR)) + '\n')
            text_file.write('----------------------------' + '\n')
            text_file.write('Peak Detection Metric' + '\n')
            text_file.write('Avg MAE across subjects: \n')
            text_file.write(str(np.mean(np.abs(final_HR_peak - final_HR0_peak))) + '\n')
            text_file.write('Avg RMSE across subjects: \n')
            text_file.write(str(np.sqrt(np.mean(np.square(final_HR_peak - final_HR0_peak)))) + '\n')
            text_file.write('Pearson Peak: \n')
            text_file.write(str(abs(np.corrcoef(final_HR_peak, final_HR0_peak)[1, 0])) + '\n')
            text_file.write('Std Peak: \n')
            text_file.write(str(np.std(final_HR_peak)) + '\n')
            text_file.write('----------------------------' + '\n')


if __name__ == '__main__':

    checkpoint_folder = str(os.path.join('./results', args.exp_name))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    test_file_list = ['UBFCPPV2NoCrop', 'PUREPPV2NoCrop', 'UBFCPPV2Crop', 'PUREPPV2Crop']
    if args.regular:
        test_file_list = ['UBFCNoCrop', 'PURECrop', 'UBFCCrop']
    for test_file in test_file_list:
        print('==============================')
        print('Test File: ', test_file)
        checkpoint_test_folder = str(os.path.join(checkpoint_folder, test_file))
        if not os.path.exists(checkpoint_test_folder):
            os.makedirs(checkpoint_test_folder)
        for model_epoch in range(args.total_epoch):
            print('Epoch : ', str(model_epoch))
            eval(args, model_epoch, test_file, checkpoint_test_folder)