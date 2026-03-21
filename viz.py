import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def overlay_attention(image, attn_map):
    """
    image: [H, W, 1] or [H, W], grayscale (float in [0,1] or uint8)
    attn_map: [H_f, W_f], float attention from model
    Produces a grayscale image where attention darkens the ink.
    """
    img = image.squeeze()
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0)

    H, W = img.shape[:2]

    attn = attn_map.astype(np.float32)
    attn = cv2.resize(attn, (W, H), interpolation=cv2.INTER_NEAREST)
    attn -= attn.min()
    if attn.max() > 0:
        attn /= attn.max()

    gamma = 1.0
    attn = np.power(attn, gamma)

    alpha = 0.7
    factor = 1.0 - alpha * attn
    blended = img * factor
    blended = np.clip(blended, 0.0, 1.0)

    blended_u8 = (blended * 255).astype(np.uint8)
    blended_rgb = cv2.cvtColor(blended_u8, cv2.COLOR_GRAY2RGB)
    return blended_rgb


def draw_samples(rows, out_path):
    n = len(rows)
    fig = plt.figure(figsize=(12, 3 * n))
    gs = gridspec.GridSpec(n, 1)

    for i, row in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(row["attention_image"])
        ax.set_title(
            f"GT: {row['gt']}\n"
            f"PRED: {row['pred']}\n"
            f"COMPILES: {row['compiles']}\n"
            f"DIFF: {row['diff']}",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
