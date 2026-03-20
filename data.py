import os
import re
import cv2
import json
import numpy as np
import tensorflow as tf
from collections import Counter
from pathlib import Path
from loguru import logger


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize_latex(s: str):
    s = s.strip()
    pattern = r"(\\[a-zA-Z]+|\\.|[{}_^&%$#~]|\\\\|[0-9]+|[A-Za-z]+|\\s+|.)"
    raw = re.findall(pattern, s)
    tokens = []
    for tok in raw:
        if tok.isspace():
            continue
        tokens.append(tok)
    return tokens


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
    return [line.rstrip("\n") for line in lines]


def load_split(lst_path, formulas, image_dir):
    samples = []
    with open(lst_path, "r", encoding="utf-8") as f:
        for line in f:
            formula_idx, image_name, render_type = line.strip().split()
            formula = formulas[int(formula_idx)]
            image_path = os.path.join(image_dir, image_name + ".png")
            samples.append(
                {
                    "image_path": image_path,
                    "formula": formula,
                    "render_type": render_type,
                }
            )
    return samples


def crop_formula(img, pad=8, threshold=250):
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


def preprocess_image(path, target_height=128, max_width=512):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = crop_formula(img, pad=8)
    h, w = img.shape
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    new_w = min(new_w, max_width)
    img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, -1)
    return img


def generator(samples, vocab, max_len, target_height, max_width):
    for s in samples:
        img = preprocess_image(
            s["image_path"], target_height=target_height, max_width=max_width
        )
        tgt_in, tgt_out = vocab.encode(s["formula"], max_len=max_len)
        yield img, tgt_in, tgt_out


def make_dataset(
    samples,
    vocab,
    batch_size,
    max_len,
    target_height=128,
    max_width=512,
    shuffle=False,
    buffer_size=2048,
):
    output_signature = (
        tf.TensorSpec(shape=(target_height, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(max_len - 1,), dtype=tf.int32),
        tf.TensorSpec(shape=(max_len - 1,), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: generator(samples, vocab, max_len, target_height, max_width),
        output_signature=output_signature,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [target_height, None, 1],
            [max_len - 1],
            [max_len - 1],
        ),
        padding_values=(
            tf.constant(1.0, dtype=tf.float32),
            tf.constant(vocab.pad_id, dtype=tf.int32),
            tf.constant(vocab.pad_id, dtype=tf.int32),
        ),
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
