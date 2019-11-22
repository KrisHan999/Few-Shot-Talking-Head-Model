import argparse
from dataset.customeDataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import config


def main():
    argmentParser = argparse.ArgumentParser(description='Few-Shot-Talking-Head-Model')
    argmentParser.add_argument("--source", type=str, required=True,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    argmentParser.add_argument("--output", type=str, required=True,
                                help="Path to the folder where the pre-processed dataset will be stored.")
    argmentParser.add_argument("--gpu", action="store_true",
                                help="Run the model on GPU.")
    args = argmentParser.parse_args()

    dataset = metaTrainVideoDataset(
        K = config.K,
        rootDir = args.source,
        outputDir = args.output,
        randomFrame = True,
        device='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu',
        transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.CenterCrop(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        )

    dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print("*"*8, " start process data", "*"*8)
    for i, data in enumerate(dataset):
        print(f'Process video{i}')
        pass
    print("*"*8, " finish processing data", "*"*8)


    dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

    idx, data_array = dataset[3]

if __name__ == '__main__':
    main()
