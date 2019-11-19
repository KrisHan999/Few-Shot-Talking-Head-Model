import torch
from torch.utils.data import Dataset
import random

from .processVoxCelDataset import *


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

    def __len__(self):
        return len(self.videoList)

    def __getitem__(self, idx):

        # get K+1 frames, K for embedder; 1 for generator
        data = generateKSelectedFramesAndLandmarksForSpecificVideo(self.K + 1, self.videoList[idx], self.outputDir,
                                                                   self.fa)

        if (self.randomFrame):
            random.shuffle(data)

        data_array = []
        for d in data:
            frame = PIL.Image.fromarray(d['frame'], 'RGB')  # [H, W, 3]
            landmarks = PIL.Image.fromarray(d['landmarks'], 'RGB')  # [H, W, 3]
            if self.transform:
                frame = self.transform(frame)  # [3, H, W]
                landmarks = self.transform(landmarks)  # [3, H, W]
            assert torch.is_tensor(frame), "The source images must be converted to Tensors."
            assert torch.is_tensor(landmarks), "The source landmarks must be converted to Tensors."
            data_array.append(torch.stack((frame, landmarks)))  # [2, 3, H, W]
        data_array = torch.stack(data_array)  # [K+1, 2, 3, H, W]

        return idx, data_array


