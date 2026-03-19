import os
import argparse
import numpy as np
import tensorflow as tf
from loguru import logger

from data import load_formulas, load_split, Vocab, make_dataset, preprocess_image
from model import Im2LatexModel
from utils import setup_logging, set_seed
from metrics import char_diff, compile_latex_formula
from viz import overlay_attention, draw_samples


def build_datasets(args):
    formulas = load_formulas(os.path.join(args.dataset_dir, "im2latex_formulas.lst"))
    train_samples = load_split(os.path.join(args.dataset_dir, "im2latex_train.lst"), formulas, os.path.join(args.dataset_dir, "formatted"))
    val_samples = load_split(os.path.join(args.dataset_dir, "im2latex_validate.lst"), formulas, os.path.join(args.dataset_dir, "formatted"))
    test_samples = load_split(os.path.join(args.dataset_dir, "im2latex_test.lst"), formulas, os.path.join(args.dataset_dir, "formatted"))

    train_formulas = [s["formula"] for s in train_samples]
    vocab = Vocab.build(train_formulas, min_freq=args.min_freq, max_size=args.vocab_size)
    vocab.save(os.path.join(args.output_dir, "vocab.json"))

    train_ds = make_dataset(
        train_samples, vocab, batch_size=args.batch_size, max_len=args.max_len,
        target_height=args.target_height, max_width=args.max_width, shuffle=True
    )
    val_ds = make_dataset(
        val_samples, vocab, batch_size=args.batch_size, max_len=args.max_len,
        target_height=args.target_height, max_width=args.max_width, shuffle=False
    )
    test_ds = make_dataset(
        test_samples, vocab, batch_size=args.batch_size, max_len=args.max_len,
        target_height=args.target_height, max_width=args.max_width, shuffle=False
    )
    return vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds


def sample_visualization(model, vocab, samples, args, epoch):
    idxs = np.random.choice(len(samples), size=min(5, len(samples)), replace=False)
    rows = []
    for idx in idxs:
        sample = samples[idx]
        img = preprocess_image(sample["image_path"], target_height=args.target_height, max_width=args.max_width)
        batch = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
        pred_ids, attn = model.greedy_decode(batch, max_len=args.max_len - 1)

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

    out_path = os.path.join(args.output_dir, "samples", f"epoch_{epoch:04d}.png")
    draw_samples(rows, out_path)
    logger.info(f"saved sample visualization: {out_path}")


def train_model(args):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    setup_logging(os.path.join(args.output_dir, "train.log"))
    set_seed(args.seed)

    vocab, train_samples, val_samples, test_samples, train_ds, val_ds, test_ds = build_datasets(args)

    model = Im2LatexModel(
        vocab_size=len(vocab.token_to_id),
        d_model=args.d_model,
        emb_dim=args.emb_dim,
        dec_dim=args.dec_dim,
        attn_dim=args.attn_dim,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        pad_id=vocab.pad_id,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        logger.info(f"epoch {epoch}/{args.epochs}")
        train_logs = model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1).history

        val_loss = float(train_logs["val_loss"][-1])
        ckpt_path = os.path.join(args.output_dir, "checkpoints", "last.weights.h5")
        model.save_weights(ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.output_dir, "checkpoints", "best.weights.h5")
            model.save_weights(best_path)
            logger.info(f"new best checkpoint: {best_path}, val_loss={best_val:.5f}")

        if epoch % args.visualize_every == 0:
            sample_visualization(model, vocab, val_samples, args, epoch)

    return model, vocab, test_samples, test_ds


def evaluate_model(model, vocab, test_samples, test_ds, args):
    logger.info("running test evaluation")
    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    logger.info(f"test metrics: {metrics}")

    sample_visualization(model, vocab, test_samples, args, epoch=9999)
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--vocab-size", type=int, default=800)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument("--target-height", type=int, default=128)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--dec-dim", type=int, default=256)
    p.add_argument("--attn-dim", type=int, default=256)
    p.add_argument("--visualize-every", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, vocab, test_samples, test_ds = train_model(args)
    evaluate_model(model, vocab, test_samples, test_ds, args)