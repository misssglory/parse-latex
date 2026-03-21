import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers


class CNNEncoder(layers.Layer):
    def __init__(self, d_model=256, dropout=0.1, name="cnn_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv_blocks = [
            keras.Sequential(
                [
                    layers.Conv2D(64, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                ]
            ),
            keras.Sequential(
                [
                    layers.Conv2D(128, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                ]
            ),
            keras.Sequential(
                [
                    layers.Conv2D(256, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.Conv2D(256, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D(pool_size=(2, 1)),
                ]
            ),
            keras.Sequential(
                [
                    layers.Conv2D(256, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.Conv2D(d_model, 3, padding="same", activation="relu"),
                    layers.BatchNormalization(),
                    layers.MaxPooling2D(pool_size=(1, 2)),
                ]
            ),
        ]
        self.dropout = layers.Dropout(dropout)

    def call(self, images, training=False):
        x = images
        for block in self.conv_blocks:
            x = block(x, training=training)
        x = self.dropout(x, training=training)
        return x


class RowEncoder(layers.Layer):
    def __init__(self, d_model=256, name="row_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.proj = layers.Dense(d_model)
        self.row_rnn = layers.Bidirectional(
            layers.GRU(d_model // 2, return_sequences=True),
            merge_mode="concat",
        )

    def call(self, feat_map, training=False):
        x = self.proj(feat_map)
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = tf.shape(x)[3]

        x = tf.reshape(x, [b * h, w, c])
        x = self.row_rnn(x, training=training)
        c2 = tf.shape(x)[-1]
        x = tf.reshape(x, [b, h, w, c2])
        return x


class BahdanauAttention(layers.Layer):
    def __init__(self, attn_dim, name="bahdanau_attention", **kwargs):
        super().__init__(name=name, **kwargs)
        self.W1 = layers.Dense(attn_dim)
        self.W2 = layers.Dense(attn_dim)
        self.V = layers.Dense(1)

    def call(self, features, hidden, mask=None):
        hidden_exp = tf.expand_dims(tf.expand_dims(hidden, 1), 1)
        score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_exp)))
        score = tf.squeeze(score, axis=-1)

        if mask is not None:
            minus_inf = tf.constant(-1e9, dtype=score.dtype)
            score = tf.where(mask > 0, score, minus_inf)

        flat_score = tf.reshape(score, [tf.shape(score)[0], -1])
        flat_alpha = tf.nn.softmax(flat_score, axis=-1)
        alpha = tf.reshape(flat_alpha, tf.shape(score))
        context = tf.reduce_sum(features * tf.expand_dims(alpha, -1), axis=[1, 2])
        return context, alpha


class TokenEmbedding(layers.Layer):
    def __init__(self, vocab_size, emb_dim, name="token_embedding", **kwargs):
        super().__init__(name=name, **kwargs)
        self.embed = layers.Embedding(vocab_size, emb_dim, mask_zero=True)

    def call(self, x):
        return self.embed(x)


class AttentionDecoder(layers.Layer):
    def __init__(
        self,
        vocab_size,
        emb_dim=128,
        dec_dim=256,
        attn_dim=256,
        name="decoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.embedding = TokenEmbedding(vocab_size, emb_dim)
        self.gru = layers.GRU(dec_dim, return_state=True, return_sequences=False)
        self.attn = BahdanauAttention(attn_dim)
        self.fc1 = layers.Dense(dec_dim, activation="tanh")
        self.fc2 = layers.Dense(vocab_size, dtype="float32")

    def call_step(
        self, prev_token, prev_hidden, enc_features, image_mask=None, training=False
    ):
        token_emb = self.embedding(prev_token)
        context, alpha = self.attn(enc_features, prev_hidden, mask=image_mask)
        gru_in = tf.concat([token_emb[:, 0, :], context], axis=-1)
        gru_in = tf.expand_dims(gru_in, 1)
        out, hidden = self.gru(gru_in, initial_state=prev_hidden, training=training)
        logits = self.fc2(self.fc1(tf.concat([out, context], axis=-1)))
        return logits, hidden, alpha

    def init_hidden(self, batch_size, dec_dim):
        return tf.zeros([batch_size, dec_dim], dtype=tf.float32)


class Im2LatexModel(keras.Model):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        emb_dim=128,
        dec_dim=256,
        attn_dim=256,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dec_dim = dec_dim
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        self.encoder_cnn = CNNEncoder(d_model=d_model)
        self.row_encoder = RowEncoder(d_model=d_model)
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
        )

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.token_acc = keras.metrics.Mean(name="token_acc")

    @property
    def metrics(self):
        return [self.loss_tracker, self.token_acc]

    def encode(self, images, training=False):
        x = self.encoder_cnn(images, training=training)
        x = self.row_encoder(x, training=training)
        return x

    def call(self, inputs, training=False):
        images, tgt_in = inputs
        enc = self.encode(images, training=training)
        hidden = self.decoder.init_hidden(tf.shape(images)[0], self.dec_dim)

        seq_len = tgt_in.shape[1]
        if seq_len is None:
            raise ValueError("Target sequence length must be static.")

        logits_all = []
        for t in range(seq_len):
            prev_tok = tgt_in[:, t : t + 1]
            logits, hidden, _ = self.decoder.call_step(
                prev_tok, hidden, enc, image_mask=None, training=training
            )
            logits_all.append(logits)

        return tf.stack(logits_all, axis=1)

    def _compute_loss_and_acc(self, tgt_out, logits):
        loss_per_tok = tf.keras.losses.sparse_categorical_crossentropy(
            tgt_out, logits, from_logits=True
        )
        loss_per_tok = tf.cast(loss_per_tok, tf.float32)

        mask = tf.cast(tf.not_equal(tgt_out, self.pad_id), tf.float32)
        loss = tf.reduce_sum(loss_per_tok * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

        y_hat = tf.argmax(logits, axis=-1, output_type=tgt_out.dtype)
        correct = tf.cast(tf.equal(y_hat, tgt_out), tf.float32) * mask
        acc = tf.reduce_sum(correct) / tf.maximum(tf.reduce_sum(mask), 1.0)
        return loss, acc

    def train_step(self, data):
        images, tgt_in, tgt_out = data

        with tf.GradientTape() as tape:
            logits = self((images, tgt_in), training=True)
            loss, acc = self._compute_loss_and_acc(tgt_out, logits)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.token_acc.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, tgt_in, tgt_out = data
        logits = self((images, tgt_in), training=False)
        loss, acc = self._compute_loss_and_acc(tgt_out, logits)

        self.loss_tracker.update_state(loss)
        self.token_acc.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def greedy_decode(self, images, max_len):
        enc = self.encode(images, training=False)
        batch_size = tf.shape(images)[0]
        hidden = self.decoder.init_hidden(batch_size, self.dec_dim)
        prev = tf.fill([batch_size, 1], tf.cast(self.bos_id, tf.int32))

        tokens = []
        attentions = []

        for _ in range(max_len):
            logits, hidden, alpha = self.decoder.call_step(
                prev, hidden, enc, image_mask=None, training=False
            )
            next_tok = tf.argmax(logits, axis=-1, output_type=tf.int32)
            tokens.append(next_tok)
            attentions.append(alpha)
            prev = tf.expand_dims(next_tok, axis=1)

        token_tensor = tf.stack(tokens, axis=1)
        attn_tensor = tf.stack(attentions, axis=1)
        return token_tensor, attn_tensor
