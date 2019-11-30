import torch
import config
from dataset.customeDataset import *
from torchvision import transforms

source = config.SOURCE_DATA_DIR
output = config.OUTPUT_DATA_DIR

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

for i, data in enumerate(dataset):
	# print(f'Process video {i}')
    pass