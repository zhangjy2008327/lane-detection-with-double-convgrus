from torch.utils.data import Dataset
# from PIL import Image
import torch
# import config
# import torchvision.transforms as transforms
import numpy as np
# import os.path as ops
import json
import cv2
import os
import copy
import imageio as io
shape = (256, 128)
root_path = '/media/lab9102/FA0DAF2C6CB5A0CE/tusimple/train_set/'
root_path_test = '/media/lab9102/FA0DAF2C6CB5A0CE/tusimple/test_set/'
list1 = [1, 5, 10, 15, 20] #index-1
# list1 = [2, 5, 9, 14, 20]#index-2
# list1 = [4, 8, 12, 16, 20]#index-3
# list1 = [6, 8, 11, 15, 20]#index-4
# list1 = [8, 11, 14, 17, 20]#index-5
# list1 = [10, 11, 13, 16, 20]#index-6
# list1 = [12, 14, 16, 18, 20]#index-7

def readTxt(label_file):
    json_gt = []
    json_gt += [json.loads(line) for line in open(label_file)]
    return json_gt

def readTxttest(label_file):
    json_gt = []
    json_gt += [json.loads(line) for line in open(label_file)]
    return json_gt


class RoadSequenceDataset(Dataset):

    def __init__(self, file_path, transforms):

        self.img_list = readTxt(file_path)

        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        gt = self.img_list[idx]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        label_name = raw_file.split('/')[0] + '_' + raw_file.split('/')[1] + '_' + raw_file.split('/')[2] + '_' + raw_file.split('/')[3][:-4]

        data = []
        for i in range(5):
            new_file = raw_file[:-6] + str(list1[i]) + '.jpg'
            img_name = os.path.join(root_path_test, new_file)
            img = io.imread(img_name)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)  # shape = (320, 192)

            img = img.astype('float32') / 255.0
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img1 = torch.from_numpy(img)
            data.append(torch.unsqueeze(img1, dim=0))
        data = torch.cat(data, 0)

        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        mask = np.zeros((128, 256), dtype='float32')  ##shape = (192, 320)
        for lane in gt_lanes_vis:
            if not lane: continue

            lane = np.array([lane], dtype='float32')
            lane *= [256, 128]  # shape = (224, 224)
            lane /= [1280, 720]
            lane = lane.astype('int32')
            lane_lable = lane_lable.astype('int32')
            cv2.polylines(mask, lane, isClosed=False, color=1, thickness=1)
        mask1 = mask1 * 255
        mask = mask[None, ...]  # mask.shape(1, 192, 320)
        mask = np.ascontiguousarray(mask)
        label = torch.from_numpy(mask)
        label = torch.squeeze(label)
        tmp_file_path = root_path_test + new_file
        sample = {'data': data, 'label': label, 'label_name': label_name, 'batch_file_names': label_name, 'raw_file': tmp_file_path, 'new_label':mask1}
        return sample

class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):
        self.img_list = readTxttest(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        gt = self.img_list[idx]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']
        label_name = raw_file.split('/')[0] + '_' + raw_file.split('/')[1] + '_' + raw_file.split('/')[2] + '_' + raw_file.split('/')[3][:-4]
        data = []
        for i in range(5):
            new_file = raw_file[:-6] + str(list1[i]) + '.jpg'
            img_name = os.path.join(root_path, new_file)
            img = io.imread(img_name)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)  # shape = (320, 192)
            img = img.astype('float32') / 255.0  # (192, 320, 3)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img1 = torch.from_numpy(img)
            data.append(torch.unsqueeze(img1, dim=0))
        data = torch.cat(data, 0)

        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        mask = np.zeros((128, 256), dtype='float32')  ##shape = (192, 320)

        for lane in gt_lanes_vis:
            if not lane: continue

            lane = np.array([lane], dtype='float32')
            lane_lable = copy.deepcopy(lane)
            lane *= [256, 128]  # shape = (224, 224)
            lane /= [1280, 720]
            # print('lane----', lane)
            lane = lane.astype('int32')
            lane_lable = lane_lable.astype('int32')
            cv2.polylines(mask, lane, isClosed=False, color=1, thickness=1)
        mask = mask[None, ...]  # mask.shape(1, 192, 320)
        mask = np.ascontiguousarray(mask)
        label = torch.from_numpy(mask)
        label = torch.squeeze(label)
        sample = {'data': data, 'label': label, 'label_name': label_name, 'batch_file_names': label_name, 'raw_file': raw_file}
        return sample
        # return torch.from_numpy(img), torch.from_numpy(mask)


