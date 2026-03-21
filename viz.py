import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def overlay_attention(image, attn_map):
    """
    image: [H, W, 1] or [H, W], grayscale, float32 in [0,1] or uint8
    attn_map: [H_f, W_f], float attention from model
    """
    img = image.squeeze()

    # normalize grayscale to [0,1]
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0)

    H, W = img.shape[:2]

    # resize attention to image size
    attn = attn_map.astype(np.float32)
    attn = cv2.resize(attn, (W, H), interpolation=cv2.INTER_NEAREST)

    # normalize attention to [0,1]
    attn -= attn.min()
    if attn.max() > 0:
        attn /= attn.max()

    # optional: sharpen contrast a bit (gamma)
    gamma = 1.0
    attn = np.power(attn, gamma)

    # we want: background ~1, strokes darker where attn is high
    # combine as: blended = img * (1 - alpha * attn)
    alpha = 0.7  # 0=no effect, 1=full darkening with attn
    factor = 1.0 - alpha * attn
    blended = img * factor
    blended = np.clip(blended, 0.0, 1.0)

    # convert to RGB uint8 for matplotlib
    blended_u8 = (blended * 255).astype(np.uint8)
    blended_rgb = cv2.cvtColor(blended_u8, cv2.COLOR_GRAY2RGB)
    return blended_rgb    

def draw_samples(rows, out_path):
    n = len(rows)
    fig = plt.figure(figsize=(18, 5 * n))
    gs = gridspec.GridSpec(n, 2, width_ratios=[1, 1.4])

    for i, row in enumerate(rows):
        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])

        ax0.imshow(row["input_image"], cmap="gray")
        ax0.set_title("Input")
        ax0.axis("off")

        ax1.imshow(row["attention_image"])
        ax1.set_title(
            f"GT: {row['gt']}\n"
            f"PRED: {row['pred']}\n"
            f"COMPILES: {row['compiles']}\n"
            f"DIFF: {row['diff']}",
            fontsize=10
        )
        ax1.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)