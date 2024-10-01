import torch

# Configuration parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_FOLDER = 'where_to_save_the_model'


# Dataset parameters
IMG_DIR = 'location_of_the_images'
FRACTION = 1.0
IMG_DIM = 256
TRANSFORM = None
VALIDATION_SPLIT = 0.05
NORMALIZATION_MODE = 2


# Model parameters 
INPUT_CHANNELS = 1 # 3 channels for the image and 1 channel for the mask
OUTPUT_CHANNELS = 1
BASE_CHANNELS = 32
CHANNEL_MULTIPLIER = [1, 1, 2, 4, 8]
TEMB_DIM = 1024
NUM_RES_BLOCKS = [1, 2, 2, 4, 8]
HAS_ATTENTION = [False, False, False, True, True]
NUM_HEADS = 4
DROPOUT_FROM_RESOLUTION = 16
DROPOUT = 0.1
DOWNSAMPLING_KERNEL_DIM = 2


# Diffusion parameters
TIMESTEPS = 1000
CLIP_MAX = 1.0
CLIP_MIN = -1.0
BETA_START = 1e-4
BETA_END = 0.02
SCHEDULE = 'cosine_shifted'

# Training parameters
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
GRAD_CLIP = 1.0
GRAD_ACCUM_STEPS = 4

# Custom learning rate scheduler parameters
LR_START = LEARNING_RATE
WARMUP_STEP = 10
WARMUP_LR_END = 5e-4
ANNEALING_STEP = 40
ANNEALING_LR_END = 5e-6
NUM_CYCLES = 10
CYCLE_LENGTH = (NUM_EPOCHS - WARMUP_STEP - ANNEALING_STEP) // NUM_CYCLES
MAX_CYCLES_LR = 5e-5
SWA_UPDATE_EPOCHS = [i for i in range(WARMUP_STEP + ANNEALING_STEP - 1, NUM_EPOCHS) if (i-WARMUP_STEP-ANNEALING_STEP + 1) % CYCLE_LENGTH == 0]


# Generation parameters
GEN_BATCH_SIZE = 10
NUM_ITERATION = 100 # Num_gen_imgs = NUM_ITERATION * GEN_BATCH_SIZE
GEN_CLASS = 'CTR'
MODEL_DIR = 'location_of_the_model'
SAVE_SEMI_SYN_FOLDER = 'where_to_save_the_semi_synthetic_images'
SAVE_SYN_FOLDER = 'where_to_save_the_synthetic_images'
SYNTHETIC_MASK_DIR = 'where_the_synthetic_masks_are_stored'







