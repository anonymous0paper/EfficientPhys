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
from model import TSCAN, MTTS_CAN, EfficientPhys_Conv
from post_process import calculate_metric, calculate_metric_per_video, calculate_metric_peak_per_video
from ts_transfomer_2d import EfficientPhys_Transformer, EfficientPhys_Transformer_Both
from ts_transfomer_2d_noTSM import EfficientPhys_TransformerNoTSM
import time
from ppg_loss import ppg_loss

torch.manual_seed(100)
np.random.seed(100)
# %%
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--exp_name', type=str,
                    help='experiment name')
parser.add_argument('-i', '--data_dir', type=str,
                    default='C:\\Data\\ippg\\',
                    help='Location for the dataset')  # default='/gscratch/scrubbed/xliu0/', help='Location for the dataset')
parser.add_argument('-mo', '--model', type=str,
                    default='tscan', help='tscan or mtts-can')
parser.add_argument('-o', '--save_dir', type=str, default='./checkpoints',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-tr_data', '--tr_dataset', type=str, default='AFRL', help='training dataset name')
parser.add_argument('-ts_data', '--ts_dataset', type=str, default='AFRL', help='test dataset name')
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
parser.add_argument('-fd', '--frame_depth', type=int, default=10,
                    help='frame_depth for 3DCNN')
parser.add_argument('-save', '--save_all', type=int, default=0,
                    help='save all or not')
parser.add_argument('-shuf', '--shuffle', type=str, default=True,
                    help='shuffle samples')
parser.add_argument('-freq', '--fs', type=int, default=30,
                    help='shuffle samples')
parser.add_argument('--window_size', type=int, default=256,
                    help='window size for filtering and FFT')
parser.add_argument('--eval_only', type=int, default=0,
                    help='if eval only')
parser.add_argument('--signal', type=str, default='pulse')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--swin_window_size', type=int, default=4)
parser.add_argument('--swin_2d_patch', type=int, default=3)
parser.add_argument('--img_size', type=int, default=36)
parser.add_argument('--swin_input_channel', type=str, default='diff')
parser.add_argument('--opt', type=str, default='Adam')
parser.add_argument('--pilot', action='store_true', help='conduct pilot debug')
parser.add_argument('--loss', type=str, default='negative_pearson')
parser.add_argument('--background_aug', action='store_true', help='background augmentation')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

# %% Spliting Data

print('Spliting Data...')
subNum = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26, 27])
taskList = list(range(1, args.nb_task + 1))
[b, a] = butter(1, [0.75 / args.fs * 2, 2.5 / args.fs * 2], btype='bandpass')


def train(args):
    checkpoint_folder = str(os.path.join(args.save_dir, args.exp_name))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    print('================================')
    print('Train...')

    # Reading Data
    path_of_video_tr, path_of_video_test = read_from_txt(args.train_txt, args.test_txt, args.data_dir)
    nframe_per_video_tr = 180  # get_nframe_video(path_of_video_tr[0], dataset=args.tr_dataset)
    nframe_per_video_ts = 180  # get_nframe_video(path_of_video_test[0], dataset=args.ts_dataset)
    if args.pilot:
        path_of_video_tr = path_of_video_tr[:300]
        path_of_video_test = path_of_video_test[:]
    num_gpus = torch.cuda.device_count()
    path_tr_len = (len(path_of_video_tr) // num_gpus) * num_gpus
    path_of_video_tr = path_of_video_tr[:path_tr_len]
    path_ts_len = (len(path_of_video_test) // num_gpus) * num_gpus
    path_of_video_test = path_of_video_test[:path_ts_len]
    print('sample path: ', path_of_video_tr[0])
    print('Trian Length: ', len(path_of_video_tr))
    print('Test Length: ', len(path_of_video_test))
    print('nframe_per_video_tr', nframe_per_video_tr)
    print('nframe_per_video_ts', nframe_per_video_ts)

    # %% Create data genener
    if args.model == 'swin':
        training_dataset = rPPG_Dataset_Swin(path_of_video_tr, frame_depth=args.frame_depth,
                                             signal=args.signal, swin_input_channel=args.swin_input_channel,
                                             img_size=args.img_size)
        testing_dataset = rPPG_Dataset_Swin(path_of_video_test, frame_depth=args.frame_depth,
                                            signal=args.signal, swin_input_channel=args.swin_input_channel,
                                            img_size=args.img_size)
    else:  # This is for TS-CAN and Swin-2D
        training_dataset = rPPG_Dataset(path_of_video_tr, frame_depth=args.frame_depth,
                                        signal=args.signal, img_size=args.img_size,
                                        input_channel=args.swin_input_channel, num_gpus=num_gpus,
                                        background_aug=args.background_aug,
                                        background_path=glob.glob('./Backgrounds/*.jpg'))
        testing_dataset = rPPG_Dataset(path_of_video_test, frame_depth=args.frame_depth,
                                       signal=args.signal, img_size=args.img_size,
                                       input_channel=args.swin_input_channel, num_gpus=num_gpus,
                                       background_aug=False)

    tr_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
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

        model = EfficientPhys_Transformer3D(patch_size=(1, args.swin_window_size, args.swin_window_size), pretrained2d=False,
                                  dense_layer_number=dense_layer, in_chans=in_chans)
    elif args.model == 'conv':
        if args.swin_input_channel == 'both':
            model = TSCAN(frame_depth=args.frame_depth, img_size=args.img_size)
        else:
            model = EfficientPhys_Conv(frame_depth=args.frame_depth, img_size=args.img_size, channel=args.swin_input_channel)
    elif args.model == 'trans':
        if args.swin_input_channel == 'both':
            model = EfficientPhys_Transformer_Both(img_size=args.img_size, patch_size=3,
                                         in_chans=6, window_size=3, depths=[2, 6, 2], frame_depth=args.frame_depth)
        else:
            model = EfficientPhys_Transformer(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3, window_size=3, depths=[2, 6, 2],
                                    channel=args.swin_input_channel, frame_depth=args.frame_depth)

    elif args.model == 'trans_notsm':
        if args.swin_input_channel == 'raw':
            model = EfficientPhys_TransformerNoTSM(img_size=args.img_size, patch_size=args.swin_2d_patch, in_chans=3, window_size=3, depths=[2, 6, 2],
                                    channel=args.swin_input_channel, frame_depth=args.frame_depth)
        else:
            raise Exception('Only raw frames are supported in swin-2d-notsm')
    else:
        raise Exception('Unsupported Model!')
    model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    criterion = ppg_loss(args.loss)
    if args.opt == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.lr*10)
    else:
        raise Exception('Unsupported Optimizer. Pleae use Adam or AdamW')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        raise ValueError('Your training is not using GPU!')

    model = model.to(device)
    train_loss_freq = 50
    ##############################################################################################################
    for epoch in range(24):
        if args.eval_only == 0:
            print('Epoch: ', epoch)
            running_loss = 0.0
            tr_loss = 0.0
            for i, data in enumerate(tr_dataloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                if args.model == 'swin':
                    inputs = inputs.view(-1, in_chans, args.frame_depth, args.img_size, args.img_size)
                    if args.swin_input_channel == 'raw':
                        padd_frames = torch.zeros(inputs.shape[0], in_chans, 1, args.img_size, args.img_size)
                        inputs = torch.cat((inputs, padd_frames.to(device)), -3)
                else:
                    inputs = inputs.view(-1, in_chans, args.img_size, args.img_size)
                    if args.swin_input_channel == 'raw':
                        last_frame = torch.unsqueeze(inputs[-1, :, :, :], 0).repeat(num_gpus, 1, 1, 1)
                        inputs = torch.cat((inputs, last_frame), 0)
                labels = labels.view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item()
                tr_loss += loss
                if i % train_loss_freq == (train_loss_freq - 1):
                    print('[%d, %5d] tr loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / train_loss_freq))
                    running_loss = 0.0
            # scheduler1.step()
            print('Average Training loss: ', tr_loss / len(tr_dataloader))
            del running_loss
            del loss
            #torch.cuda.empty_cache()
        ##############################################################################################################
        if epoch % 1 == 0:
            # Evaluation
            with torch.no_grad():
                print('Evaluate...')
                model.eval()
                ts_loss = 0.0
                final_preds = []
                final_labels = []
                final_HR = []
                final_HR0 = []
                final_HR_peak = []
                final_HR0_peak = []

                for i, ts_data in enumerate(ts_dataloader):
                    # print('idx: ', i)
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
                    ts_outputs = model(ts_inputs)
                    loss = criterion(ts_outputs, ts_labels)
                    ts_loss += loss
                    ts_outputs_numpy = ts_outputs.cpu().numpy()
                    ts_labels_numpy = ts_labels.cpu().numpy()
                    HR0, HR = calculate_metric_per_video(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                         fs=args.fs, bpFlag=True)
                    HR0_peak, HR_peak, _, _, _, _ = calculate_metric_peak_per_video(ts_outputs_numpy, ts_labels_numpy, signal=args.signal,
                                                                        window_size=args.window_size, fs=args.fs, bpFlag=True)
                    final_HR.append(HR)
                    final_HR0.append(HR0)
                    final_HR_peak.append(HR_peak)
                    final_HR0_peak.append(HR0_peak)
                    final_preds.append(ts_outputs_numpy)
                    final_labels.append(ts_labels_numpy)

                final_preds = np.array(final_preds)
                final_labels = np.array(final_labels)
                final_HR = np.array(final_HR)
                final_HR0 = np.array(final_HR0)
                final_HR_peak = np.array(final_HR_peak)
                final_HR0_peak = np.array(final_HR0_peak)

                print('Avg Validation Loss: ', ts_loss / len(ts_dataloader))
                print('FFT Metric')
                print('Avg MAE across subjects: ', np.mean(np.abs(final_HR - final_HR0)))
                print('Avg RMSE across subjects: ', np.sqrt(np.mean(np.square(final_HR - final_HR0))))
                print('Peak Detection Metric')
                print('Avg MAE across subjects: ', np.mean(np.abs(final_HR_peak - final_HR0_peak)))
                print('Avg RMSE across subjects: ', np.sqrt(np.mean(np.square(final_HR_peak - final_HR0_peak))))

                model_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '.pth'))
                pred_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_pred'))
                label_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_label'))
                final_HR_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR_all'))
                final_HR0_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR0_all'))
                final_HR_peak_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR_all_peak'))
                final_HR0_peak_path = str(os.path.join(checkpoint_folder, str(args.exp_name) + '_' + str(epoch) + '_HR0_all_peak'))

                torch.save(model.state_dict(), model_path)
                np.save(pred_path, final_preds)
                np.save(label_path, final_labels)
                np.save(final_HR_path, final_HR)
                np.save(final_HR0_path, final_HR0)
                np.save(final_HR_peak_path, final_HR_peak)
                np.save(final_HR0_peak_path, final_HR0_peak)
                print('Pearson Results')
                print('Pearson FFT: ', abs(np.corrcoef(final_HR, final_HR0)[1, 0]))
                print('Pearson Peak: ', abs(np.corrcoef(final_HR_peak, final_HR0_peak)[1, 0]))
                print('Std FFT', np.std(final_HR))
                print('Std Peak', np.std(final_HR_peak))
    print('Finished Training')


if __name__ == '__main__':
    train(args)