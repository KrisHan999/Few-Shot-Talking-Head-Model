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
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
    )

    dataLoader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

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
        logging.info("===== Loading model =====")
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
        logging.info("===== Done loading model =====")
    else:
        batch_current = 0
        epochCurrent = 0


    # TRAIN

    logging.info(f'Start training -> EPOCHS: {config.EPOCHS}; BATCHES: {len(dataLoader)}; BATCH_SIZE: {config.BATCH_SIZE} ---> CURRENT EPOCH: {epochCurrent}; CURRENT_BATCH: {batch_current}')

    for epoch in range(epochCurrent, config.EPOCHS):
        epoch_start = datetime.now()

        E.train()
        G.train()
        D.train()

        for batch_num, (index, data_array) in enumerate(dataLoader, start=batch_current):

            if batch_num > len(dataLoader):
                batch_current = 0
                break

            with torch.autograd.enable_grad():

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

                wi = D.W[:, index.numpy()].transpose(0, 1).unsqueeze(-1)

                loss_D = cretirion_D(score_target_img, score_generated_img)
                loss_EG = cretirion_EG(target_img, generated_img, score_generated_img, mean_vector, wi, fm_teature, fm_teature_hat)

                loss = loss_D.to(device_0) + loss_EG.to(device_0)

                optimizer_EG.zero_grad()
                optimizer_D.zero_grad()

                loss.backward(retain_graph=False)
                optimizer_EG.step()
                optimizer_D.step()

                # train discriminator again
                # detach the generated image
                score_generated_img, fm_teature_hat = D(generated_img.detach(), target_landmark, index.numpy())
                score_target_img, fm_teature = D(target_img, target_landmark, index.numpy())
                loss_D = cretirion_D(score_target_img, score_generated_img)
                loss = loss_D

                optimizer_D.zero_grad()
                loss.backward(retain_graph=False)
                optimizer_D.step()

                batch_end = datetime.now()

                logging.info(f'Epoch {epoch + 1}: [{batch_num + 1}/{len(dataLoader)}] | '
                             f'Time: {batch_end - batch_start} | '
                             f'Loss_E_G = {loss_EG.item():.4f} Loss_D = {loss_D.item():.4f}')
                logging.debug(f'D(x) = {score_target_img.mean().item():.4f} D(x_hat) = {score_generated_img.mean().item():.4f}')

            if batch_num % 250 == 249:
                logging.info('Saving latest: epoch: {epoch}; batch: {batch_num}')
                torch.save({
                    'epoch': epoch,
                    'loss_EG': loss_EG,
                    'loss_D': loss_D,
                    'E_state_dict': E.state_dict(),
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizer_EG_state_dict': optimizer_EG.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'num_vid': dataset.__len__(),
                    'batch_num': batch_num
                    }, config.MODELS_path)
                logging.info('Done saving latest: epoch: {epoch}; batch: {batch_num}')

                if(target_img.shape[0] == config.BATCH_SIZE):
                    temp_output = []
                    for i in range(config.BATCH_SIZE):
                        temp_output.append(np.concatenate([target_img[i].permute(1, 2, 0).detach().cpu().numpy(), 
                                        target_landmark[i].permute(1, 2, 0).detach().cpu().numpy(), 
                                        generated_img[i].permute(1, 2, 0).detach().cpu().numpy()],
                                        axis=0))
                    temp_output = np.concatenate(temp_output, axis=1)

                    img = (temp_output * 255.0).clip(0, 255).astype("uint8")
                    img = Image.fromarray(img)
                    img.save(os.path.join(config.LOG_IMAGE_DIR, f'epoch_{epoch}_batch_{batch_num}.jpg'))
                    # print(temp_output.shape)
                    # plt.imshow(temp_output)
                    # plt.show()

        if(epoch % 10 == 499):
            logging.info('Saving model epoch: {epoch}')
            torch.save({
                'epoch': epoch,
                'loss_EG': loss_EG,
                'loss_D': loss_D,
                'E_state_dict': E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_EG_state_dict': optimizer_EG.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'num_vid': dataset.__len__(),
                'batch_num': batch_num
                }, config.MODELS_path)
            logging.info('Done saving model epoch: {epoch}')


if __name__ == '__main__':
    main()
