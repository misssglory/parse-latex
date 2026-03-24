import os
import argparse
import numpy as np
import tensorflow as tf
from loguru import logger

from config import TrainConfig
from utils import setup_logging, set_seed, setup_precision
from data import (
    load_formulas, load_split, Vocab, preprocess_image,
    build_preprocessed_split, make_dataset_from_manifest
)
from model import Im2LatexModel
from metrics import char_diff, compile_latex_formula
from viz import overlay_attention, draw_samples
from history_utils import load_history, save_history, append_epoch, plot_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)

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

    parser.add_argument("--cache-preprocessed", action="store_true")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--num-preprocess-workers", type=int, default=None)

    args = parser.parse_args()

    cfg = TrainConfig()
    if args.config:
        cfg = TrainConfig.load_json(args.config)

    if args.resume_from:
        resume_dir = args.resume_from
        resolved_path = os.path.join(resume_dir, "resolved_config.json")
        cfg = TrainConfig.load_json(resolved_path)
        cfg.output_dir = resume_dir

    for key, value in vars(args).items():
        if key in ("config", "resume_from"):
            continue
        if key == "run_eagerly":
            if value:
                cfg.run_eagerly = True
            continue
        if key == "cache_preprocessed":
            if value:
                cfg.cache_preprocessed = True
            continue
        if value is not None:
            setattr(cfg, key.replace("-", "_"), value)

    return cfg, args.resume_from


def build_splits(cfg):
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
    return train_samples, val_samples, test_samples


def build_or_load_manifests(cfg, vocab, train_samples, val_samples, test_samples):
    train_manifest = build_preprocessed_split(
        train_samples, vocab, "train", cfg.output_dir,
        cfg.max_len, cfg.target_height, cfg.max_width, cfg.scale_factor,
        shard_size=cfg.shard_size,
        num_workers=cfg.num_preprocess_workers,
        dirname=cfg.preprocessed_dirname,
    )
    val_manifest = build_preprocessed_split(
        val_samples, vocab, "val", cfg.output_dir,
        cfg.max_len, cfg.target_height, cfg.max_width, cfg.scale_factor,
        shard_size=cfg.shard_size,
        num_workers=cfg.num_preprocess_workers,
        dirname=cfg.preprocessed_dirname,
    )
    test_manifest = build_preprocessed_split(
        test_samples, vocab, "test", cfg.output_dir,
        cfg.max_len, cfg.target_height, cfg.max_width, cfg.scale_factor,
        shard_size=cfg.shard_size,
        num_workers=cfg.num_preprocess_workers,
        dirname=cfg.preprocessed_dirname,
    )
    return train_manifest, val_manifest, test_manifest


def make_datasets_for(cfg, vocab, train_samples, val_samples, test_samples):
    train_manifest, val_manifest, test_manifest = build_or_load_manifests(
        cfg, vocab, train_samples, val_samples, test_samples
    )

    train_ds = make_dataset_from_manifest(
        train_manifest, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed
    )
    val_ds = make_dataset_from_manifest(
        val_manifest, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed
    )
    test_ds = make_dataset_from_manifest(
        test_manifest, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed
    )
    return train_ds, val_ds, test_ds


def prepare_model(cfg, vocab):
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
    return model


def prepare_new_training(cfg):
    train_samples, val_samples, test_samples = build_splits(cfg)

    train_formulas = [s["formula"] for s in train_samples]
    vocab = Vocab.build(train_formulas, min_freq=cfg.min_freq, max_size=cfg.vocab_size)

    os.makedirs(cfg.output_dir, exist_ok=True)
    vocab.save(os.path.join(cfg.output_dir, "vocab.json"))
    cfg.save_json(os.path.join(cfg.output_dir, "resolved_config.json"))

    train_ds, val_ds, test_ds = make_datasets_for(cfg, vocab, train_samples, val_samples, test_samples)
    model = prepare_model(cfg, vocab)

    history_path = os.path.join(cfg.output_dir, cfg.history_file)
    history = load_history(history_path)

    return model, vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds, history


def prepare_resume(cfg):
    train_samples, val_samples, test_samples = build_splits(cfg)
    vocab = Vocab.load(os.path.join(cfg.output_dir, "vocab.json"))

    train_ds, val_ds, test_ds = make_datasets_for(cfg, vocab, train_samples, val_samples, test_samples)
    model = prepare_model(cfg, vocab)

    best_ckpt = os.path.join(cfg.output_dir, "checkpoints", "best.weights.h5")
    if os.path.exists(best_ckpt):
        model.load_weights(best_ckpt)
        logger.info(f"Resuming from checkpoint: {best_ckpt}")
    else:
        logger.warning("Checkpoint not found. Training from scratch")

    history_path = os.path.join(cfg.output_dir, cfg.history_file)
    history = load_history(history_path)
    logger.info(f"Loaded history with {len(history.get('loss', []))} epochs")

    return model, vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds, history


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
        blended = overlay_attention(img, avg_attn)

        rows.append({
            "attention_image": blended,
            "gt": gt_formula,
            "pred": pred_formula,
            "diff": diff,
            "compiles": compiles,
        })

    out_path = os.path.join(cfg.output_dir, "samples", f"epoch_{epoch:04d}.png")
    draw_samples(rows, out_path)
    logger.info(f"saved sample visualization: {out_path}")


def train_model(cfg, model, vocab, train_samples, val_samples, train_ds, val_ds, history):
    os.makedirs(os.path.join(cfg.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "checkpoints"), exist_ok=True)

    best_val = min(history.get("val_loss", []) or [float("inf")])

    history_path = os.path.join(cfg.output_dir, cfg.history_file)
    hist_png = os.path.join(cfg.output_dir, "history.png")

    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"epoch {epoch}/{cfg.epochs}")

        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=1,
            verbose=1,
        ).history

        epoch_hist = {
            "loss": float(hist["loss"][-1]),
            "val_loss": float(hist["val_loss"][-1]),
            "token_acc": float(hist["token_acc"][-1]),
            "val_token_acc": float(hist["val_token_acc"][-1]),
        }
        history = append_epoch(history, epoch_hist)
        save_history(history, history_path)
        plot_history(history, hist_png)

        val_loss = epoch_hist["val_loss"]

        last_path = os.path.join(cfg.output_dir, "checkpoints", "last.weights.h5")
        model.save_weights(last_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(cfg.output_dir, "checkpoints", "best.weights.h5")
            model.save_weights(best_path)
            logger.info(f"new best checkpoint: {best_path}, val_loss={best_val:.6f}")

        if epoch % cfg.visualize_every == 0:
            sample_visualization(model, vocab, val_samples, cfg, epoch)

    return model, history


def evaluate_model(model, vocab, test_samples, test_ds, cfg):
    logger.info("running test evaluation")
    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"test metrics: {metrics}")
    sample_visualization(model, vocab, test_samples, cfg, epoch=9999)
    return metrics


if __name__ == "__main__":
    cfg, resume_from = parse_args()

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger, log_path = setup_logging(cfg.output_dir, base_name="train")
    logger.info(f"Log file: {log_path}")

    set_seed(cfg.seed)
    setup_precision(cfg.precision)

    if resume_from:
        model, vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds, history = \
            prepare_resume(cfg)
    else:
        model, vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds, history = \
            prepare_new_training(cfg)

    model, history = train_model(cfg, model, vocab, train_samples, val_samples, train_ds, val_ds, history)
    evaluate_model(model, vocab, test_samples, test_ds, cfg)