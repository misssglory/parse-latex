import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_history(path):
    p = Path(path)
    if not p.exists():
        return {"loss": [], "val_loss": [], "token_acc": [], "val_token_acc": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def append_epoch(history, epoch_hist):
    for key in ("loss", "val_loss", "token_acc", "val_token_acc"):
        if key not in history:
            history[key] = []
    history["loss"].append(epoch_hist["loss"])
    history["val_loss"].append(epoch_hist["val_loss"])
    history["token_acc"].append(epoch_hist["token_acc"])
    history["val_token_acc"].append(epoch_hist["val_token_acc"])
    return history


def plot_history(history, out_png):
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    if not epochs:
        return

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["token_acc"], label="train_token_acc")
    plt.plot(epochs, history["val_token_acc"], label="val_token_acc")
    plt.xlabel("epoch")
    plt.ylabel("token acc")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=180)
    plt.close()
