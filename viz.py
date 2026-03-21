import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def overlay_attention(image, attn_map):
    img = image.squeeze()
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    attn = cv2.resize(attn_map.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    attn = attn - attn.min()
    if attn.max() > 0:
        attn = attn / attn.max()

    heat = (plt.cm.jet(attn)[..., :3] * 255).astype(np.uint8)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    out = cv2.addWeighted(base, 0.65, heat, 0.35, 0)
    return out


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