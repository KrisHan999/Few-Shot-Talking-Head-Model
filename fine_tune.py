import argparse
from dataset.customDataset import *
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
    argmentParser.add_argument("--videoPath", type=str, required=False,
                                help="Path to the source folder where the raw VoxCeleb dataset is located.")
    argmentParser.add_argument("--gpu", action="store_true",
                                help="Run the model on GPU.")
    args = argmentParser.parse_args()


    # LOGGING ----------------------------------------------------------------------------------------------------------

    if not os.path.isdir(config.FINETUNE_LOG_DIR):
        os.makedirs(config.FINETUNE_LOG_DIR)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config.FINETUNE_LOG_DIR, f'{datetime.now():%Y%m%d}.log'),
        format='[%(asctime)s][%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("==== Meta-training ====")


    # DATASET and DATALOADER -------------------------------------------------------------------------------------------
    logging.info(f'Fine tuning videoPath is {args.videoPath}')

    fine_tune_dataset = FineTuneVideoDataset(T=config.K,
                                             videoPath=args.videoPath,
                                             device='cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu',
                                             transform=transforms.Compose([
                                                 transforms.Resize(256),
                                                 transforms.CenterCrop(256),
                                                 transforms.ToTensor(),
                                             ])
                                             )

    dataLoader = DataLoader(fine_tune_dataset, batch_size=config.FINE_TUNE_BATCH_SIZE, shuffle=True)

    # Load Model if exist
    cpu = torch.device("cpu")
    if (os.path.isfile(config.MODELS_path)):
        logging.info("===== Loading model =====")
        checkpoint = torch.load(config.MODELS_path, map_location=cpu)
    else:
        print("No meta-training model, failed")
        exit()

    # MODEL and GPU --------------------------------------------------------------------------------------------------------

    device_0 = torch.device("cuda:0")

    E = Embedder(gpu=config.GPU["E"])
    G = Generator(gpu=config.GPU["G"])
    D = Discriminator(num_person=checkpoint['num_vid'], gpu=config.GPU["D"])

    cretirion_EG = LossEG(gpu=config.GPU["LossEG"])
    cretirion_D = LossD(gpu=config.GPU["LossD"])


    optimizer_EG = Adam(params=list(E.parameters()) + list(G.parameters()),
                        lr=config.LEARNING_RATE_EG)
    optimizer_D = Adam(params=D.parameters(),
                       lr=config.LEARNING_RATE_D)


    # loading model for each individual part
    E.load_state_dict(checkpoint['E_state_dict'])
    G.load_state_dict(checkpoint['G_state_dict'])
    D.load_state_dict(checkpoint['D_state_dict'])
    optimizer_EG.load_state_dict(checkpoint['optimizer_EG_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    meta_epochCurrent = checkpoint['epoch']
    loss_EG = checkpoint['loss_EG']
    loss_D = checkpoint['loss_D']
    meta_num_vid = checkpoint['num_vid']
    meta_batch_current = checkpoint['batch_num']
    logging.info("===== Done loading model =====")


    # TRAIN

    logging.info(f'Start training -> EPOCHS: {config.EPOCHS}; BATCHES: {len(dataLoader)}; BATCH_SIZE: {config.BATCH_SIZE} ---> CURRENT EPOCH: {epochCurrent}; CURRENT_BATCH: {batch_current}')

    embedded_img = fine_tune_dataset.data_array[:, :, 0, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)  # [T, 3, 256, 256]
    embedded_landmark = fine_tune_dataset.data_array[:, :, 1, ...].reshape(-1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)  # [T, 3, 256, 256]

    embedded_vector = E(embedded_img, embedded_landmark)
    mean_vector = embedded_vector.mean(dim=1)  # [1, 512, 1]

    logging.info("===== Initialize fine tuning =====")
    G.initFinetuning(mean_vector)
    D.initFinetuning(mean_vector)
    logging.info("===== Done fine tuning =====")



    for epoch in range(config.FINE_TUNE_EPOCHS):

        E.train()
        G.train()
        D.train()

        for batch_num, data in enumerate(dataLoader):

            with torch.autograd.enable_grad():

                batch_start = datetime.now()

                target_img = data[:, 0, ...].unsqueeze(0)                                   # [1, 3, 256, 256]
                target_landmark = data[:, 1, ...].unsqueeze(0)                              # [1, 3, 256, 256]

                generated_img = G(target_landmark, mean_vector)

                score_generated_img, fm_teature_hat = D(generated_img, target_landmark)
                score_target_img, fm_teature = D(target_img, target_landmark)


                loss_D = cretirion_D(score_target_img, score_generated_img)
                loss_EG = cretirion_EG(target_img, generated_img, score_generated_img, None, None, fm_teature, fm_teature_hat)

                loss = loss_D.to(device_0) + loss_EG.to(device_0)

                optimizer_EG.zero_grad()
                optimizer_D.zero_grad()

                loss.backward(retain_graph=False)
                optimizer_EG.step()
                optimizer_D.step()

                # train discriminator again
                # detach the generated image
                score_generated_img, fm_teature_hat = D(generated_img.detach(), target_landmark)
                score_target_img, fm_teature = D(target_img, target_landmark)
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


        if(epoch % 2 == 1):
            logging.info('Saving model epoch: {epoch}')
            torch.save({
                'epoch': meta_epochCurrent,
                'loss_EG': loss_EG,
                'loss_D': loss_D,
                'E_state_dict': E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_EG_state_dict': optimizer_EG.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'num_vid': meta_num_vid,
                'batch_num': meta_batch_current
                }, config.MODELS_path)
            logging.info('Done saving model epoch: {epoch}')

            if (target_img.shape[0] == config.BATCH_SIZE):
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


if __name__ == '__main__':
    main()
