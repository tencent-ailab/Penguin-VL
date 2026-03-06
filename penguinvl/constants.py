# Copyright (c) Penguin-VL team at Tencent AI Lab
# Model Constants
IGNORE_INDEX = -100

# Image arguments
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# Video arguments
VIDEO_TOKEN_INDEX = -201
DEFAULT_VIDEO_TOKEN = "<video>"
NUM_FRAMES = 128
MAX_FRAMES = 768
NUM_FRAMES_PER_SECOND = 1

MODAL_INDEX_MAP = {
    "<image>": -200,
    "<video>": -201,
}

# frame similarity
MIN_FRAME_SIMILARITY = 0.95
