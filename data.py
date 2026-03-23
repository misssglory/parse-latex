import os
import re
import json
import math
from pathlib import Path
from collections import Counter
from multiprocessing import Pool

import cv2
import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize_latex(s: str):
    s = s.strip()
    pattern = r"(\\[a-zA-Z]+|\\.|[{}_^&%$#~]|\\\\|[0-9]+|[A-Za-z]+|\\s+|.)"
    raw = re.findall(pattern, s)
    return [tok for tok in raw if not tok.isspace()]


class Vocab:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.pad_id = token_to_id["<pad>"]
        self.bos_id = token_to_id["<bos>"]
        self.eos_id = token_to_id["<eos>"]
        self.unk_id = token_to_id["<unk>"]

    @classmethod
    def build(cls, formulas, min_freq=1, max_size=None):
        counter = Counter()
        for f in formulas:
            counter.update(tokenize_latex(f))

        items = [tok for tok, c in counter.items() if c >= min_freq]
        items.sort(key=lambda t: (-counter[t], t))

        if max_size is not None:
            items = items[: max(0, max_size - len(SPECIAL_TOKENS))]

        vocab = SPECIAL_TOKENS + items
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(token_to_id)

    def encode(self, formula, max_len):
        toks = tokenize_latex(formula)
        ids = (
            [self.bos_id]
            + [self.token_to_id.get(t, self.unk_id) for t in toks]
            + [self.eos_id]
        )
        ids = ids[:max_len]
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))

        tgt_in = ids[:-1]
        tgt_out = ids[1:]
        return np.array(tgt_in, np.int32), np.array(tgt_out, np.int32)

    def decode(self, ids):
        toks = []
        for i in ids:
            tok = self.id_to_token.get(int(i), "<unk>")
            if tok in ("<pad>", "<bos>"):
                continue
            if tok == "<eos>":
                break
            toks.append(tok)
        return "".join(toks)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))


def read_text_auto(path):
    path = Path(path)
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None

    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            logger.info(f"Read {path} with encoding={enc}")
            return text, enc
        except UnicodeDecodeError as e:
            last_error = e
            logger.warning(f"Failed reading {path} with encoding={enc}: {e}")

    raise last_error


def load_formulas(path):
    text, enc = read_text_auto(path)
    lines = text.splitlines()
    logger.info(f"Loaded {len(lines)} formulas from {path} using encoding={enc}")
    return lines


def load_split(lst_path, formulas, image_dir):
    samples = []
    with open(lst_path, "r", encoding="utf-8") as f:
        for line in f:
            formula_idx, image_name, render_type = line.strip().split()
            samples.append(
                {
                    "image_path": os.path.join(image_dir, image_name + ".png"),
                    "formula": formulas[int(formula_idx)],
                    "render_type": render_type,
                }
            )
    return samples


def crop_formula(img, pad=8, threshold=250):
    if img is None:
        raise ValueError("Failed to load image")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coords = np.column_stack(np.where(img < threshold))
    if len(coords) == 0:
        return img

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    y0 = max(0, y0 - pad)
    x0 = max(0, x0 - pad)
    y1 = min(img.shape[0], y1 + pad)
    x1 = min(img.shape[1], x1 + pad)

    return img[y0:y1, x0:x1]


def preprocess_image(path, target_height=128, max_width=512, scale_factor=1.0):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = crop_formula(img, pad=8)

    if scale_factor <= 0:
        raise ValueError("scale_factor must be > 0")

    if scale_factor != 1.0:
        new_h = max(1, int(round(img.shape[0] * scale_factor)))
        new_w = max(1, int(round(img.shape[1] * scale_factor)))
        interp = cv2.INTER_LINEAR if scale_factor > 1.0 else cv2.INTER_AREA
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    h, w = img.shape
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_w = min(new_w, max_width)

    img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, -1)
    return img


def _preprocess_one(args):
    sample, token_to_id, max_len, target_height, max_width, scale_factor = args
    vocab = Vocab(token_to_id)
    img = preprocess_image(
        sample["image_path"],
        target_height=target_height,
        max_width=max_width,
        scale_factor=scale_factor,
    )
    tgt_in, tgt_out = vocab.encode(sample["formula"], max_len=max_len)
    return img, tgt_in, tgt_out


def get_preprocessed_dir(output_dir, dirname="preprocessed"):
    return os.path.join(output_dir, dirname)


def get_split_dir(output_dir, split_name, dirname="preprocessed"):
    return os.path.join(get_preprocessed_dir(output_dir, dirname), split_name)


def get_manifest_path(output_dir, split_name, dirname="preprocessed"):
    return os.path.join(get_split_dir(output_dir, split_name, dirname), "manifest.json")


def save_shard(shard_path, imgs, tins, touts):
    img_obj = np.empty((len(imgs),), dtype=object)
    for i, img in enumerate(imgs):
        img_obj[i] = img

    tin_arr = np.asarray(tins, dtype=np.int32)
    tout_arr = np.asarray(touts, dtype=np.int32)

    np.savez_compressed(
        shard_path,
        images=img_obj,
        tgt_in=tin_arr,
        tgt_out=tout_arr,
    )


def build_preprocessed_split(
    samples,
    vocab,
    split_name,
    output_dir,
    max_len,
    target_height,
    max_width,
    scale_factor,
    shard_size=512,
    num_workers=4,
    dirname="preprocessed",
):
    split_dir = get_split_dir(output_dir, split_name, dirname)
    os.makedirs(split_dir, exist_ok=True)
    manifest_path = get_manifest_path(output_dir, split_name, dirname)

    if os.path.exists(manifest_path):
        logger.info(
            f"Found preprocessed manifest for split={split_name}: {manifest_path}"
        )
        return manifest_path

    logger.info(f"Building preprocessed split={split_name} into {split_dir}")

    shard_files = []
    imgs, tins, touts = [], [], []

    worker_args = [
        (s, vocab.token_to_id, max_len, target_height, max_width, scale_factor)
        for s in samples
    ]

    shard_idx = 0
    with Pool(processes=num_workers) as pool:
        for img, tgt_in, tgt_out in tqdm(
            pool.imap(_preprocess_one, worker_args, chunksize=32),
            total=len(worker_args),
            desc=f"Preprocessing {split_name}",
            leave=True,
        ):
            imgs.append(img)
            tins.append(tgt_in)
            touts.append(tgt_out)

            if len(imgs) >= shard_size:
                shard_path = os.path.join(split_dir, f"shard_{shard_idx:05d}.npz")
                save_shard(shard_path, imgs, tins, touts)
                shard_files.append(shard_path)
                imgs, tins, touts = [], [], []
                shard_idx += 1

    if imgs:
        shard_path = os.path.join(split_dir, f"shard_{shard_idx:05d}.npz")
        save_shard(shard_path, imgs, tins, touts)
        shard_files.append(shard_path)

    manifest = {
        "split_name": split_name,
        "num_samples": len(samples),
        "num_shards": len(shard_files),
        "shard_files": shard_files,
        "target_height": target_height,
        "max_len_minus_1": max_len - 1,
        "pad_id": vocab.pad_id,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved manifest for split={split_name}: {manifest_path}")
    return manifest_path


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def shard_sample_generator(manifest_path, shuffle=False, seed=42):
    manifest = load_manifest(manifest_path)
    shard_files = list(manifest["shard_files"])

    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(shard_files)

    for shard_path in shard_files:
        data = np.load(shard_path, allow_pickle=True)
        images = data["images"]
        tgt_in = data["tgt_in"]
        tgt_out = data["tgt_out"]

        idxs = np.arange(len(images))
        if shuffle:
            rng.shuffle(idxs)

        for i in idxs:
            img = np.asarray(images[i], dtype=np.float32)
            ti = np.asarray(tgt_in[i], dtype=np.int32)
            to = np.asarray(tgt_out[i], dtype=np.int32)
            yield img, ti, to


def make_dataset_from_manifest(manifest_path, batch_size, shuffle=False, seed=42):
    manifest = load_manifest(manifest_path)
    target_height = manifest["target_height"]
    seq_len = manifest["max_len_minus_1"]
    pad_id = manifest["pad_id"]

    output_signature = (
        tf.TensorSpec(shape=(target_height, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: shard_sample_generator(manifest_path, shuffle=shuffle, seed=seed),
        output_signature=output_signature,
    )

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [target_height, None, 1],
            [seq_len],
            [seq_len],
        ),
        padding_values=(
            tf.constant(1.0, dtype=tf.float32),
            tf.constant(pad_id, dtype=tf.int32),
            tf.constant(pad_id, dtype=tf.int32),
        ),
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
