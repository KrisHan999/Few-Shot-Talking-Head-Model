# VGG_FACE = r'/home/<user>/Documents/NeuralNetworkModels/vgg_face_dag.pth'
VGG_FACE = r'../vgg_face/vgg_face_dag.pth'
LOG_DIR = r'logs'
MODELS_DIR = r'models'
GENERATED_DIR = r'generated_img'

# Dataset parameters
FEATURES_DPI = 100
K = 8

# Training hyperparameters
IMAGE_SIZE = 256  # 224
BATCH_SIZE = 3
EPOCHS = 1000

LEARNING_RATE_EG = 5e-5
LEARNING_RATE_D = 2e-4

LOSS_VGG_FACE_WEIGHT = 2.5e-2
LOSS_VGG19_WEIGHT = 1.5e-1
LOSS_MCH_WEIGHT = 1e1
LOSS_FM_WEIGHT = 1e1

# GPU

GPU = {'E': 'm',
       'G': 'm',
       'D': 'm',
       "LossEG": 'm',
       "LossG": 'm'
       }