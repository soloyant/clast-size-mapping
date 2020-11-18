import os
import sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config

############################################################
#  Configurations
############################################################


class clastsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "clasts"
    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + clasts

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 95% confidence
    DETECTION_MIN_CONFIDENCE = 0

    # Use full size mask
    USE_MINI_MASK = False

    #Number of validation steps after an epoch
    VALIDATION_STEPS = 50
    OPTIMIZER = "SGD" # can be ADAM or SGD
    DETECTION_MAX_INSTANCES = 1000 #number of discrete water polygons to detect

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024 #Etretat 2160 | Pourville 1216 | VillersCasino 1856 | Sardinero 1022 | 
    IMAGE_MAX_DIM = 1024 #Etretat 3840 | Pourville 1936 | VillersCasino 3264 | Sardinero 1280 | 
    IMAGE_MIN_SCALE = 1
    
    MEAN_PIXEL = np.array([111.3883, 110.9057, 106.6095])
	
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000


