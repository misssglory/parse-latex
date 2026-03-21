import os
import argparse
import numpy as np
import tensorflow as tf
from loguru import logger

from config import TrainConfig
from utils import setup_logging, set_seed, setup_precision
from data import load_formulas, load_split, Vocab, make_dataset, preprocess_image
from model import Im2LatexModel
from metrics import char_diff, compile_latex_formula
from viz import overlay_attention, draw_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--min-freq", type=int, default=None)

    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--max-width", type=int, default=None)
    parser.add_argument("--scale-factor", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--emb-dim", type=int, default=None)
    parser.add_argument("--dec-dim", type=int, default=None)
    parser.add_argument("--attn-dim", type=int, default=None)

    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--num-visual-samples", type=int, default=None)

    parser.add_argument("--precision", type=str, choices=["fp32", "fp16"], default=None)
    parser.add_argument("--run-eagerly", action="store_true")

    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config:
        cfg = TrainConfig.load_json(args.config)

    for key, value in vars(args).items():
        if key == "config":
            continue
        if key == "run_eagerly":
            if value:
                cfg.run_eagerly = True
            continue
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)

    return cfg


def build_datasets(cfg):
    formulas = load_formulas(os.path.join(cfg.dataset_dir, "im2latex_formulas.lst"))
    train_samples = load_split(
        os.path.join(cfg.dataset_dir, "im2latex_train.lst"),
        formulas,
        os.path.join(cfg.dataset_dir, "formatted"),
    )
    val_samples = load_split(
        os.path.join(cfg.dataset_dir, "im2latex_validate.lst"),
        formulas,
        os.path.join(cfg.dataset_dir, "formatted"),
    )
    test_samples = load_split(
        os.path.join(cfg.dataset_dir, "im2latex_test.lst"),
        formulas,
        os.path.join(cfg.dataset_dir, "formatted"),
    )

    train_formulas = [s["formula"] for s in train_samples]
    vocab = Vocab.build(train_formulas, min_freq=cfg.min_freq, max_size=cfg.vocab_size)

    os.makedirs(cfg.output_dir, exist_ok=True)
    vocab.save(os.path.join(cfg.output_dir, "vocab.json"))

    train_ds = make_dataset(
        train_samples, vocab,
        batch_size=cfg.batch_size,
        max_len=cfg.max_len,
        target_height=cfg.target_height,
        max_width=cfg.max_width,
        scale_factor=cfg.scale_factor,
        shuffle=True,
    )
    val_ds = make_dataset(
        val_samples, vocab,
        batch_size=cfg.batch_size,
        max_len=cfg.max_len,
        target_height=cfg.target_height,
        max_width=cfg.max_width,
        scale_factor=cfg.scale_factor,
        shuffle=False,
    )
    test_ds = make_dataset(
        test_samples, vocab,
        batch_size=cfg.batch_size,
        max_len=cfg.max_len,
        target_height=cfg.target_height,
        max_width=cfg.max_width,
        scale_factor=cfg.scale_factor,
        shuffle=False,
    )
    return vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds


def sample_visualization(model, vocab, samples, cfg, epoch):
    count = min(cfg.num_visual_samples, len(samples))
    idxs = np.random.choice(len(samples), size=count, replace=False)

    rows = []
    for idx in idxs:
        sample = samples[idx]
        img = preprocess_image(
            sample["image_path"],
            target_height=cfg.target_height,
            max_width=cfg.max_width,
            scale_factor=cfg.scale_factor,
        )

        batch = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
        pred_ids, attn = model.greedy_decode(batch, max_len=cfg.max_len - 1)

        pred_formula = vocab.decode(pred_ids[0].numpy())
        gt_formula = sample["formula"]
        avg_attn = tf.reduce_mean(attn[0], axis=0).numpy()

        compiles, _ = compile_latex_formula(pred_formula)
        diff = char_diff(gt_formula, pred_formula)

        rows.append({
            "input_image": img.squeeze(),
            "attention_image": overlay_attention(img, avg_attn),
            "gt": gt_formula,
            "pred": pred_formula,
            "diff": diff,
            "compiles": compiles,
        })

    out_path = os.path.join(cfg.output_dir, "samples", f"epoch_{epoch:04d}.png")
    draw_samples(rows, out_path)
    logger.info(f"saved sample visualization: {out_path}")


def train_model(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    setup_logging(os.path.join(cfg.output_dir, cfg.log_file))
    set_seed(cfg.seed)
    setup_precision(cfg.precision)

    cfg.save_json(os.path.join(cfg.output_dir, "resolved_config.json"))

    vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds = build_datasets(cfg)

    model = Im2LatexModel(
        vocab_size=len(vocab.token_to_id),
        d_model=cfg.d_model,
        emb_dim=cfg.emb_dim,
        dec_dim=cfg.dec_dim,
        attn_dim=cfg.attn_dim,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        pad_id=vocab.pad_id,
        name="im2_latex_model",
    )

    dummy_images = tf.zeros([1, cfg.target_height, 128, 1], dtype=tf.float32)
    dummy_tgt = tf.zeros([1, cfg.max_len - 1], dtype=tf.int32)
    _ = model((dummy_images, dummy_tgt), training=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    model.compile(optimizer=optimizer, run_eagerly=cfg.run_eagerly)

    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"epoch {epoch}/{cfg.epochs}")

        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=1,
        ).history

        val_loss = float(hist["val_loss"][-1])

        last_path = os.path.join(cfg.output_dir, "checkpoints", "last.weights.h5")
        model.save_weights(last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.output_dir, "checkpoints", "best.weights.h5")
            model.save_weights(best_path)
            logger.info(f"new best checkpoint: {best_path}, val_loss={best_val:.6f}")

        if epoch % cfg.visualize_every == 0:
            sample_visualization(model, vocab, val_samples, cfg, epoch)

    return model, vocab, test_samples, test_ds


def evaluate_model(model, vocab, test_samples, test_ds, cfg):
    logger.info("running test evaluation")
    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"test metrics: {metrics}")
    sample_visualization(model, vocab, test_samples, cfg, epoch=9999)
    return metrics


if __name__ == "__main__":
    cfg = parse_args()
    model, vocab, test_samples, test_ds = train_model(cfg)
    evaluate_model(model, vocab, test_samples, test_ds, cfg)