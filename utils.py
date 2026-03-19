import os
import random
import numpy as np
import tensorflow as tf
from loguru import logger


def setup_logging(log_path):
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(log_path, rotation="10 MB", level="DEBUG")
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)