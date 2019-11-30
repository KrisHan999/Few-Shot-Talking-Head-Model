import torch
import torch.nn as nn

import argparse
from dataset.customeDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import config
import logging
import os
import sys
from datetime import datetime
from PIL import Image


from network.model import *
from loss.loss import *

source = config.SOURCE_DATA_DIR
output = config.OUTPUT_DATA_DIR

# DATASET and DATALOADER -------------------------------------------------------------------------------------------
print(f'Original training dataset located in {source}')
print(f'Processed training dataset located in {output}')

dataset = metaTrainVideoDataset(
    K=config.K,
    rootDir=source,
    outputDir=output,
    randomFrame=True,
    device='cuda' if (torch.cuda.is_available()) else 'cpu',
    transform=transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor()
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])
    ])
)

dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# MODEL and GPU --------------------------------------------------------------------------------------------------------

device_0 = torch.device("cuda:0")

E = Embedder(gpu=config.GPU["E"])
G = Generator(gpu=config.GPU["G"])
D = Discriminator(num_person=len(dataset), gpu=config.GPU["D"])

# pytorch_total_params = sum(p.numel() for p in G.parameters()) + sum(p.numel() for p in E.parameters()) + sum(p.numel() for p in D.parameters())
# print("Total size of parameters for E, G, D is: ", pytorch_total_params)

cretirion_EG = LossEG(gpu=config.GPU["LossEG"])
cretirion_D = LossD(gpu=config.GPU["LossD"])

# if torch.cuda.device_count() > 1:
#     print("Let's use ", torch.cuda.device_count(), " GPUs.")
#     E = nn.DataParallel(E)
#     G = nn.DataParallel(G)
#     D = nn.DataParallel(D)
#     cretirion_EG = nn.DataParallel(cretirion_EG)
#     cretirion_D = nn.DataParallel(cretirion_D)

optimizer_EG = Adam(params=list(E.parameters()) + list(G.parameters()),
                    lr=config.LEARNING_RATE_EG)
optimizer_D = Adam(params=D.parameters(),
                   lr=config.LEARNING_RATE_D)


# Load Model if exist
cpu = torch.device("cpu")
if(os.path.isfile(config.MODELS_path)):
    print("===== Loading model =====")
    checkpoint = torch.load(config.MODELS_path, map_location=cpu)
    E.load_state_dict(checkpoint['E_state_dict'])
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_EG.load_state_dict(checkpoint['optimizer_EG_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    epochCurrent = checkpoint['epoch']
    loss_EG = checkpoint['loss_EG']
    loss_D = checkpoint['loss_D']
    num_vid = checkpoint['num_vid']
    batch_current = checkpoint['batch_num'] +1
    print("===== Done loading model =====")
else:
    exist()


index_data_array = next(iter(dataLoader))
target_img = data_array[:, -1, 0, ...]                                   # [B, 3, 256, 256]
target_landmark = data_array[:, -1, 1, ...]                              # [B, 3, 256, 256]

embedded_img = data_array[:, :-1, 0, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)          # [BxK, 3, 256, 256]
embedded_landmark = data_array[:, :-1, 1, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)     # [BxK, 3, 256, 256]

embedded_vector = E(embedded_img, embedded_landmark)
mean_vector = embedded_vector.view(-1, config.K, 512, 1).mean(dim=1)                               # [B, 512, 1]

generated_img = G(target_landmark, mean_vector)

for i in range(config.BATCH_SIZE):
    temp_output.append(np.concatenate([target_img[i].permute(1, 2, 0).detach().cpu().numpy(), 
                    target_landmark[i].permute(1, 2, 0).detach().cpu().numpy(), 
                    generated_img[i].permute(1, 2, 0).detach().cpu().numpy()],
                    axis=0))
temp_output = np.concatenate(temp_output, axis=1)

img = (temp_output * 255.0).clip(0, 255).astype("uint8")

plt.imshow(img)
plt.show()

