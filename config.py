DATA_FILE = '/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp/flickr8k_2.pkl'
IMAGE_DIR = '/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp/Images'
OUTPUT_DIR = '/data/circulars/DATA/layoutLM+Tactful/model_outputs/gcmi/temp/teacher_forcing_output'

BATCH_SIZE = 32
EPOCH = 50
G_LEARNING_RATE = 1e-5
D_LEARNING_RATE = 1e-5
TEACHER_FORCING_RATIO = 0.5

gpu_device = 1