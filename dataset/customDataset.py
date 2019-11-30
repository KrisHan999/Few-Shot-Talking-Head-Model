import torch
from torch.utils.data import Dataset
import random

from dataset.processVoxCelDataset import *


class metaTrainVideoDataset(Dataset):
    def __init__(self, K, rootDir, outputDir, randomFrame, device='gpu', transform=None):
        '''

        :param K:
        :param rootDir: root directory of the video files
        :param outputDir: output directory for generated landmarks
        :param randomSeed: if none
        :param device:
        :param transform:
        '''
        self.K = K
        self.rootDir = rootDir
        self.outputDir = outputDir
        self.randomFrame = randomFrame
        self.device = device
        self.transform = transform

        self.videoList = generateVideoList(rootDir)
        self.fa = FaceAlignment(LandmarksType._2D, device=device)
        self.personNumber = len(os.listdir(rootDir))

    def __len__(self):
        return len(self.videoList)

    def __getitem__(self, idx):
        # get K+1 frames, K for embedder; 1 for generator
        data = generateKSelectedFramesAndLandmarksForSpecificVideo(self.K + 1, self.videoList[idx], self.outputDir,
                                                                   self.fa)

        if (self.randomFrame):
            random.shuffle(data)

        data_array = []
        for i, d in enumerate(data):
            if i == self.K+1:
                break;
            index = d["index"]
            frame = PIL.Image.fromarray(d['frame'], 'RGB')  # [H, W, 3]
            landmarks = PIL.Image.fromarray(d['landmarks'], 'RGB')  # [H, W, 3]
            if self.transform:
                frame = self.transform(frame)  # [3, H, W]
                landmarks = self.transform(landmarks)  # [3, H, W]
            assert torch.is_tensor(frame), "The source images must be converted to Tensors."
            assert torch.is_tensor(landmarks), "The source landmarks must be converted to Tensors."
            data_array.append(torch.stack((frame, landmarks)))  # [2, 3, H, W]
        data_array = torch.stack(data_array)  # [K+1, 2, 3, H, W]

        assert torch.eq(torch.tensor(data_array.shape), torch.tensor([self.K+1, 2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE])).all(), f'Wrong data size-> {data_array.shape}; target shape -> {[self.K+1, 2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE]}'

        return index, data_array
#         return idx, data_array


class FineTuneVideoDataset(Dataset):
    def __init__(self, lenDataset, K, videoPath, randomFrame=None, device='gpu', transform=None):

        super(FineTuneVideoDataset, self).__init__()
        self.lenDataset = lenDataset
        self.K = K
        self.videoPath = videoPath
        self.randomFrame = randomFrame
        self.device = device
        self.transform = transform

        self.fa = FaceAlignment(LandmarksType._2D, device=device)

    def __len__(self):
        return self.lenDataset

    def __getitem__(self, idx):

        data = generateDataForFineTuning(self.K+1, self.videoPath, self.fa)

        if self.randomFrame:
            random.shuffle(data)

        data_array = []
        for i, d in enumerate(data):
            if i == self.K + 1:
                break;
            frame = PIL.Image.fromarray(d['frame'], 'RGB')  # [H, W, 3]
            landmarks = PIL.Image.fromarray(d['landmarks'], 'RGB')  # [H, W, 3]
            if self.transform:
                frame = self.transform(frame)  # [3, H, W]
                landmarks = self.transform(landmarks)  # [3, H, W]
            assert torch.is_tensor(frame), "The source images must be converted to Tensors."
            assert torch.is_tensor(landmarks), "The source landmarks must be converted to Tensors."
            data_array.append(torch.stack((frame, landmarks)))  # [2, 3, H, W]
        data_array = torch.stack(data_array)  # [K+1, 2, 3, H, W]

        assert torch.eq(torch.tensor(data_array.shape), torch.tensor([self.K + 1, 2, 3, config.IMAGE_SIZE,
                                                                      config.IMAGE_SIZE])).all(), f'Wrong data size-> {data_array.shape}; target shape -> {[self.K + 1, 2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE]}'

        return data_array







