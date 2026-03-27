import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


def overlay_attention(image, attn_map, colormap=cv2.COLORMAP_JET, alpha=0.7, gamma=1.0):
    """
    image: [H, W, 1] or [H, W], grayscale (float in [0,1] or uint8)
    attn_map: [H_f, W_f], float attention from model
    colormap: OpenCV colormap constant or None for grayscale mode
              Common options: cv2.COLORMAP_JET, cv2.COLORMAP_HOT, cv2.COLORMAP_VIRIDIS, etc.
              Use None for original grayscale darkening effect
    alpha: blending factor (0 = no attention effect, 1 = full attention effect)
    gamma: gamma correction for attention map (adjusts contrast)
    Produces RGB image with colored heatmap overlay or grayscale darkening.
    """
    img = image.squeeze()
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32) / 255.0
    else:
        img = np.clip(img, 0.0, 1.0)

    H, W = img.shape[:2]

    # Process attention map
    attn = attn_map.astype(np.float32)
    attn = cv2.resize(attn, (W, H), interpolation=cv2.INTER_NEAREST)
    attn -= attn.min()
    if attn.max() > 0:
        attn /= attn.max()
    
    # Apply gamma correction
    attn = np.power(attn, gamma)

    if colormap is None:
        # Original grayscale darkening mode
        factor = 1.0 - alpha * attn
        blended = img * factor
        blended = np.clip(blended, 0.0, 1.0)
        blended_u8 = (blended * 255).astype(np.uint8)
        blended_rgb = cv2.cvtColor(blended_u8, cv2.COLOR_GRAY2RGB)
    else:
        # Colored heatmap mode
        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Create colored attention map
        attn_u8 = (attn * 255).astype(np.uint8)
        attn_colored = cv2.applyColorMap(attn_u8, colormap)
        attn_colored = attn_colored.astype(np.float32) / 255.0
        
        # Blend original image with attention heatmap
        # Use alpha to control attention intensity
        blended = img_rgb.astype(np.float32) / 255.0 * (1 - alpha) + attn_colored * alpha
        blended = np.clip(blended, 0.0, 1.0)
        blended_u8 = (blended * 255).astype(np.uint8)
        blended_rgb = blended_u8
    
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
