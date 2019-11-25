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

from network.model import *
from loss.loss import *


def main():
    argmentParser = argparse.ArgumentParser(description='Few-Shot-Talking-Head-Model')
    argmentParser.add_argument("--source", type=str, required=True,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    argmentParser.add_argument("--output", type=str, required=True,
                                help="Path to the folder where the pre-processed dataset will be stored.")
    argmentParser.add_argument("--gpu", action="store_true",
                                help="Run the model on GPU.")
    args = argmentParser.parse_args()

    # LOGGING ----------------------------------------------------------------------------------------------------------

    if not os.path.isdir(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config.LOG_DIR, f'{datetime.now():%Y%m%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("==== Meta-training ====")
    logging.info(f'Running on {torch.cuda.device_count()} GPU')


    # DATASET and DATALOADER -------------------------------------------------------------------------------------------
    logging.info(f'Original training dataset located in {args.source}')
    logging.info(f'Processed training dataset located in {args.output}')

    dataset = metaTrainVideoDataset(
        K=config.K,
        rootDir=args.source,
        outputDir=args.output,
        randomFrame=True,
        device='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu',
        transform=transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    )

    dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # MODEL and GPU --------------------------------------------------------------------------------------------------------

    device = torch.device("cuda:0")

    E = Embedder()
    G = Generator()
    D = Discriminator(num_person=6)

    cretirion_EG = LossEG()
    cretirion_D = LossD()

    if torch.cuda.device_count() > 1:
        print("Let's use ", torch.cuda.device_count(), " GPUs.")
        E = nn.DataParallel(E)
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)
        cretirion_EG = nn.DataParallel(cretirion_EG)
        cretirion_D = nn.DataParallel(cretirion_D)

    E.to(device)
    G.to(device)
    D.to(device)
    cretirion_EG.to(device)
    cretirion_D.to(device)

    optimizer_EG = Adam(params=list(E.parameters()) + list(G.parameters()),
                        lr=config.LEARNING_RATE_EG)
    optimizer_D = Adam(params=D.parameters(),
                       lr=config.LEARNING_RATE_D)


    # TRAIN

    logging.info(f'Start training -> EPOCHS: {config.EPOCHS}; BATCHES: {len(dataset)}; BATCH_SIZE: {config.BATCH_SIZE}')


    # person_Id_batches = torch.randn(config.BATCH_SIZE, 1)
    # raw_data_batches = torch.randn(config.BATCH_SIZE, config.K + 1, 2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    #
    # target_imgs = raw_data_batches[:, -1, 0, ...]                                   # [B, 3, 256, 256]
    # target_landmarks = raw_data_batches[:, -1, 1, ...]                              # [B, 3, 256, 256]
    #
    # embedder_imgs = raw_data_batches[:, :-1, 0, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)       # [BxK, 3, 256, 256]
    # embedder_landmarks = raw_data_batches[:, :-1, 1, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)  # [BxK, 3, 256, 256]
    #
    # embedder_vector = torch.randn(config.BATCH_SIZE*config.K, 512, 1)                                           # [BxK, 3, 512, 1]
    # embedder_vector_mean = embedder_vector.view(-1, config.K, 512, 1).mean(dim=1)                               # [B, 512, 1]
    # wi = torch.randn(config.BATCH_SIZE, 512, 1)
    #
    # generated_imgs = torch.randn(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)                    # [B, 3, 256, 256]
    # scores_target = torch.randn(config.BATCH_SIZE, 1)                                                           # [B, 1]
    # scores_generated_imgs = torch.randn(config.BATCH_SIZE, 1)                                                           # [B, 1]
    #
    #
    # loss_D = cretirion_D(scores_target, scores_generated_imgs)
    # loss_EG = cretirion_EG(target_imgs, generated_imgs, scores_generated_imgs, embedder_vector_mean, wi)
    #
    # loss = loss_D + loss_EG
    #
    # optimizer_EG.zero_grad()
    # optimizer_D.zero_grad()
    #
    # loss.backwad()
    # optimizer_EG.step()
    # optimizer_D.step()
    #
    # # train discriminator again
    # # detach the generated image
    # loss_D = cretirion_D(scores_target, scores_generated_imgs)
    # loss = loss_D
    #
    # optimizer_D.zero_grad()
    # loss.backwad()
    # optimizer_D.step()










if __name__ == '__main__':
    main()
