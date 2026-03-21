import os
import random
import numpy as np
import tensorflow as tf
from loguru import logger
from datetime import datetime


def setup_logging(log_dir, base_name="train"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{base_name}-{ts}.log")

    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(log_path, rotation="10 MB", level="DEBUG")
    return logger, log_path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_precision(precision: str):
    precision = precision.lower().strip()

    if precision == "fp32":
        tf.keras.mixed_precision.set_global_policy("float32")
    elif precision == "fp16":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        raise ValueError(f"Unsupported precision={precision}. Use fp32 or fp16.")

    gpus = tf.config.list_physical_devices("GPU")
    logger.info(f"Mixed precision policy set to {tf.keras.mixed_precision.global_policy()}")
    if precision == "fp16" and not gpus:
        logger.warning("fp16 selected but no GPU detected; on CPU this may be slower.")
