# --- Training & Model Hyperparameters ---
MODEL_NAME = "microsoft/trocr-small-stage1"
DATASET_NAME = "cyttic/diffusionpen-hebrew-handwriting-cer0"
OUTPUT_DIR = "./finetuned_models"

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
EPOCHS = 5
MAX_TARGET_LENGTH = 128
NUM_BEAMS = 4

# Dataset slicing limits
DATASET_SIZE = 100
IGNORE_INDEX = -100
