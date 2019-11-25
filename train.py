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
    argmentParser.add_argument("--source", type=str, required=False,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    argmentParser.add_argument("--output", type=str, required=False,
                                help="Path to the folder where the pre-processed dataset will be stored.")
    argmentParser.add_argument("--gpu", action="store_true",
                                help="Run the model on GPU.")
    args = argmentParser.parse_args()

    args.source = config.SOURCE_DATA_DIR
    args.output = config.OUTPUT_DATA_DIR

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

    for epoch in range(config.EPOCHS):
        epoch_start = datetime.now()

        E.train()
        G.train()
        D.train()

        for batch_num, (index, data_array) in enumerate(dataset):
            batch_start = datetime.now()

            target_img = data_array[:, -1, 0, ...]                                   # [B, 3, 256, 256]
            target_landmark = data_array[:, -1, 1, ...]                              # [B, 3, 256, 256]

            embedded_img = data_array[:, :-1, 0, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)          # [BxK, 3, 256, 256]
            embedded_landmark = data_array[:, :-1, 1, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)     # [BxK, 3, 256, 256]

            embedded_vector = E(embedded_img, embedded_landmark)
            mean_vector = embedded_vector.view(-1, config.K, 512, 1).mean(dim=1)                               # [B, 512, 1]

            generated_img = G(target_landmark, mean_vector)

            score_generated_img, fm_teature_hat = D(generated_img, target_landmark, index.numpy())
            score_target_img, fm_teature = D(target_img, target_landmark, index.numpy())

            wi = D.W[:, index.numpy()].T.unsqueeze(-1)

            loss_D = cretirion_D(score_target_img, score_generated_img)
            loss_EG = cretirion_EG(target_img, generated_img, score_generated_img, mean_vector, wi, fm_teature, fm_teature_hat)

            loss = loss_D + loss_EG

            optimizer_EG.zero_grad()
            optimizer_D.zero_grad()

            loss.backward()
            optimizer_EG.step()
            optimizer_D.step()

            # train discriminator again
            # detach the generated image
            score_generated_img = D(generated_img.detach(), target_landmark, index.numpy())
            score_target_img = D(target_img, target_landmark, index.numpy())
            loss_D = cretirion_D(score_target_img, score_generated_img)
            loss = loss_D

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            batch_end = datetime.now()

            logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(dataset)}] | '
                         f'Time: {batch_end - batch_start} | '
                         f'Loss_E_G = {loss_EG.item():.4f} Loss_D = {loss_D.item():.4f}')
            logging.debug(f'D(x) = {score_target_img.mean().item():.4f} D(x_hat) = {score_generated_img.mean().item():.4f}')

if __name__ == '__main__':
    main()
