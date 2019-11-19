import argparse
from .dataset.customeDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms


def main():
    argmentParser = argparse.ArgumentParser(description='Few-Shot-Talking-Head-Model')
    argmentParser.add_argument("--source", type=str, required=True,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    argmentParser.add_argument("--output", type=str, required=True,
                                help="Path to the folder where the pre-processed dataset will be stored.")
    args = argmentParser.parse_args()

    dataset = metaTrainVideoDataset(
        K = 8,
        rootDir = args.source,
        outputDir = args.output,
        randomFrame = True,
        device = 'gpu',
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])
        )

    dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i in enumerate(dataset):
        pass