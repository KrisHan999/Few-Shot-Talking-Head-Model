SOURCE_DATA_DIR = r'/mnt/data/kunhan/test/mp4'
OUTPUT_DATA_DIR = r'/mnt/data/kunhan/test/outputHK'
VGG_FACE = r'../vgg_face/vgg_face_dag.pth'
LOG_DIR = r'logs'
FINETUNE_LOG_DIR = r'finetune_logs'
MODELS_path = r'models/model_weights.ckpt'
GENERATED_DIR = r'generated_img'
LOG_IMAGE_DIR = r'log_img'
FINETUNE_LOG_IMAGE_DIR = r'finetune_log_img'


# Dataset parameters
FEATURES_DPI = 100
K = 8
LEN_FINETUNE = 10

# Training hyperparameters
IMAGE_SIZE = 256  # 224
BATCH_SIZE = 2
EPOCHS = 1000

FINE_TUNE_BATCH_SIZE = 1
FINE_TUNE_EPOCHS = K

LEARNING_RATE_EG = 5e-5
LEARNING_RATE_D = 2e-4

LOSS_VGG_FACE_WEIGHT = 2.5e-2
LOSS_VGG19_WEIGHT = 1.5e-1	
LOSS_MCH_WEIGHT = 1e1
LOSS_FM_WEIGHT = 1e1

# GPU

GPU = {'E': 3,
       'G': 3,
       'D': 1,
       "LossEG": 2,
       "LossD": 1
       }