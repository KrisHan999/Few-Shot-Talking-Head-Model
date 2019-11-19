#!/usr/bin/env python
# coding: utf-8


import PIL
import cv2
import matplotlib.pyplot as plt
import numpy as np
from face_alignment import FaceAlignment, LandmarksType
import os


def generateVideoList(folderPath):
    def checkFilePath(dirnames, filenames):
         return len([x for x in filenames if os.path.splitext(x)[1] != ".mp4"]) == 0 and len(dirnames) == 0
        
    videoList = []
    for dirpath, dirnames, filenames in os.walk("./VoxCelebrityMp4"):
        if(checkFilePath(dirnames, filenames)):
            for filename in filenames:
                videoList.append(os.path.join(dirpath, filename))
    return videoList


def selectKRandomFramesForSpecificVideo(K, filePath, randomSeed=None):
    cap = cv2.VideoCapture(filePath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if(randomSeed is not None):
        np.random.seed(randomSeed)
    randomIndex = np.random.choice(total, K)
    
    # print(randomIndex)
    
    selectedFrames = []
    
    for index in randomIndex:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        selectedFrames.append(frame)
    
    cap.release()
    return selectedFrames


# In[16]:


def generateLandmarksForSpecificVideo(frames, fa):
    def generateLandmarkForSpecificFrame(frame, landmark):
        fig = plt.figure(figsize=(frame.shape[1]/100, frame.shape[0]/100), dpi=100)
        ax = fig.add_subplot(1,1,1)
        ax.imshow(np.ones(frame.shape))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Head
        ax.plot(landmark[0:17, 0], landmark[0:17, 1], linestyle='-', color='green', lw=2)
        # Eyebrows
        ax.plot(landmark[17:22, 0], landmark[17:22, 1], linestyle='-', color='orange', lw=2)
        ax.plot(landmark[22:27, 0], landmark[22:27, 1], linestyle='-', color='orange', lw=2)
        # Nose
        ax.plot(landmark[27:31, 0], landmark[27:31, 1], linestyle='-', color='blue', lw=2)
        ax.plot(landmark[31:36, 0], landmark[31:36, 1], linestyle='-', color='blue', lw=2)
        # Eyes
        ax.plot(landmark[36:42, 0], landmark[36:42, 1], linestyle='-', color='red', lw=2)
        ax.plot(landmark[42:48, 0], landmark[42:48, 1], linestyle='-', color='red', lw=2)
        # Mouth
        ax.plot(landmark[48:60, 0], landmark[48:60, 1], linestyle='-', color='purple', lw=2)

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return data
    
    landmarkFrames = []
    for frame in frames:
        # landmark -> [x, y]
        landmark = fa.get_landmarks(frame)[0]
        landmarkFrame = generateLandmarkForSpecificFrame(frame, landmark)
        landmarkFrames.append(landmarkFrame)
    
    return landmarkFrames


def generateKSelectedFramesAndLandmarksForSpecificVideo(K, videoPath, fa, randomSeed=None):
    selectedFrames = selectKRandomFramesForSpecificVideo(K, videoPath, randomSeed)
    landmarkFrames = generateLandmarksForSpecificVideo(selectedFrames, fa)
    return selectedFrames, landmarkFrames


'''
folderPath = './VoxCelebrityMp4'
K = 8
randomSeed = 0
videoList = generateVideoList(folderPath)
fa = FaceAlignment(LandmarksType._2D, device='cpu')


selectedFrames, landmarkFrames = generateKSelectedFramesAndLandmarksForSpecificVideo(K, videoList[0], fa, randomSeed)
for i in range(8):
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(selectedFrames[i])
    ax2.imshow(landmarkFrames[i])
'''





