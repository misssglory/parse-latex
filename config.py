from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainConfig:
    dataset_dir: str = "dataset"
    output_dir: str = "outputs"

    epochs: int = 20
    batch_size: int = 8
    lr: float = 1e-3
    seed: int = 42

    max_len: int = 160
    vocab_size: int = 800
    min_freq: int = 1

    target_height: int = 128
    max_width: int = 512
    scale_factor: float = 1.0

    d_model: int = 256
    emb_dim: int = 128
    dec_dim: int = 256
    attn_dim: int = 256

    visualize_every: int = 5
    num_visual_samples: int = 5

    precision: str = "fp32"   # fp32 | fp16
    run_eagerly: bool = False

    log_file: str = "train.log"

    def save_json(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)