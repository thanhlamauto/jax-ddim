"""Conditional DDIM training with Classifier-Free Guidance on Kaggle TPU v5e-8.

Dataset: NIST SD 302a fingerprint images
  - 8 sensor classes: A, B, C, D, E, F, G, H
  - Images: 512x512 grayscale PNG, resized to IMAGE_SIZE x IMAGE_SIZE

TPU: Kaggle TPU v5e-8 (8 chips)
  - Uses jax.pmap for data-parallel training across 8 devices
  - Batch is sharded: (BATCH_SIZE, H, W, 1) -> (8, BATCH_SIZE//8, H, W, 1)
"""

import argparse
import functools
import os
from pathlib import Path
from typing import Tuple, Any
from datetime import datetime

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import wandb

from model import DiffusionModel


SENSORS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
NUM_CLASSES = len(SENSORS)
NULL_CLASS = NUM_CLASSES  # index 8 = unconditional token for CFG


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_file_label_lists(data_dir: str):
    """Collect all PNG file paths and corresponding integer class labels."""
    data_dir = Path(data_dir)
    paths, labels = [], []
    for label_idx, sensor in enumerate(SENSORS):
        sensor_dir = data_dir / 'challengers' / sensor / 'roll' / 'png'
        if not sensor_dir.exists():
            continue
        for p in sorted(sensor_dir.glob('*.png')):
            paths.append(str(p))
            labels.append(label_idx)
    return paths, labels


def preprocess_image(image_path: tf.Tensor, label: tf.Tensor,
                     image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw, channels=1)  # (H, W, 1) uint8
    image = tf.cast(image, tf.float32)

    # Center crop to square then resize
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    crop = tf.minimum(h, w)
    image = tf.image.crop_to_bounding_box(image,
                                          (h - crop) // 2,
                                          (w - crop) // 2,
                                          crop, crop)
    image = tf.image.resize(image, (image_size, image_size), antialias=True)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)  # (image_size, image_size, 1)
    return image, label


def build_dataset(data_dir: str,
                  image_size: int = 128,
                  batch_size: int = 512,
                  val_fraction: float = 0.1,
                  seed: int = 42):
    paths, labels = build_file_label_lists(data_dir)
    n = len(paths)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    paths = [paths[i] for i in idx]
    labels = [labels[i] for i in idx]

    n_val = max(1, int(n * val_fraction))
    train_paths, train_labels = paths[n_val:], labels[n_val:]
    val_paths, val_labels = paths[:n_val], labels[:n_val]

    preprocess_fn = functools.partial(preprocess_image, image_size=image_size)

    def make_split(split_paths, split_labels, shuffle: bool):
        ds = tf.data.Dataset.from_tensor_slices(
            (split_paths, tf.cast(split_labels, tf.int32))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(split_paths), 10000), seed=seed)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(split_paths), 5000))
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = make_split(train_paths, train_labels, shuffle=True)
    ds_val = make_split(val_paths, val_labels, shuffle=False)
    return ds_train, ds_val


# ---------------------------------------------------------------------------
# Train state
# ---------------------------------------------------------------------------

class TrainState(train_state.TrainState):
    batch_stats: Any


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def l1_loss(predictions, targets):
    return jnp.abs(predictions - targets)


def update_ema(p_cur, p_new, momentum: float = 0.999):
    return momentum * p_cur + (1 - momentum) * p_new


# ---------------------------------------------------------------------------
# Train / eval steps (pmap-compatible)
# ---------------------------------------------------------------------------

@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(4,))
def train_step(state, images, labels, rng, p_uncond: float = 0.1):
    # Randomly drop labels to null class for CFG training
    rng, rng_drop = jax.random.split(rng)
    drop_mask = jax.random.bernoulli(rng_drop, p_uncond, labels.shape)
    labels_in = jnp.where(drop_mask, NULL_CLASS, labels)

    def loss_fn(params):
        outputs, mutated_vars = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            images, labels_in, rng, train=True,
            mutable=['batch_stats']
        )
        noises, imgs, pred_noises, pred_images = outputs
        noise_loss = l1_loss(pred_noises, noises).mean()
        image_loss = l1_loss(pred_images, imgs).mean()
        loss = noise_loss + image_loss
        return loss, mutated_vars

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, mutated_vars), grads = grad_fn(state.params)

    # Average gradients and loss across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')

    state = state.apply_gradients(
        grads=grads,
        batch_stats=mutated_vars['batch_stats']
    )
    return state, loss


@functools.partial(jax.pmap, axis_name='batch')
def val_step(state, images, labels, rng):
    """Forward pass without gradient for validation loss."""
    outputs, _ = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        images, labels, rng, train=False,
        mutable=['batch_stats']
    )
    noises, imgs, pred_noises, pred_images = outputs
    noise_loss = l1_loss(pred_noises, noises).mean()
    image_loss = l1_loss(pred_images, imgs).mean()
    loss = noise_loss + image_loss
    noise_loss = jax.lax.pmean(noise_loss, axis_name='batch')
    image_loss = jax.lax.pmean(image_loss, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    return loss, noise_loss, image_loss


@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(4, 5))
def generate_cfg_step(state, ema_params, rng, class_labels,
                      diffusion_steps: int, guidance_scale: float):
    """Generate images with CFG on a single device shard."""
    variables = {'params': ema_params, 'batch_stats': state.batch_stats}
    generated = state.apply_fn(
        variables,
        rng,
        (class_labels.shape[0], 128, 128, 1),  # image_shape per device
        class_labels,
        diffusion_steps,
        guidance_scale,
        method=DiffusionModel.generate_cfg
    )
    return generated


# ---------------------------------------------------------------------------
# Checkpoint helpers (orbax)
# ---------------------------------------------------------------------------

def save_ckpt(ckpt_manager, state, ema_params, step: int):
    unreplicated_state = jax_utils.unreplicate(state)
    unreplicated_ema = jax_utils.unreplicate(ema_params)
    ckpt_manager.save(step, args=ocp.args.StandardSave({
        'state': unreplicated_state,
        'ema_params': unreplicated_ema,
    }))


# ---------------------------------------------------------------------------
# Output / logging helpers
# ---------------------------------------------------------------------------

def create_output_dir(output_dir: Path):
    ckpt_dir = output_dir / 'models'
    log_dir = output_dir / 'logs'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    return output_dir, ckpt_dir, log_dir


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run(data_dir: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        val_diffusion_steps: int,
        guidance_scale: float,
        p_uncond: float,
        log_image_every: int,
        output_dir: Path):

    # Hide GPUs from TF so it doesn't grab VRAM (not needed on TPU but keeps things clean)
    tf.config.experimental.set_visible_devices([], 'GPU')

    # W&B init — API key read from environment variable WANDB_API_KEY
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(
        project="ddim-fingerprint",
        config={
            "data_dir": data_dir,
            "epochs": epochs,
            "image_size": image_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "val_diffusion_steps": val_diffusion_steps,
            "guidance_scale": guidance_scale,
            "p_uncond": p_uncond,
            "num_classes": NUM_CLASSES,
            "feature_stages": [64, 128, 256, 512],
            "embedding_dims": 64,
        },
    )

    n_devices = jax.device_count()
    print(f"JAX devices: {n_devices} x {jax.devices()[0].device_kind}")
    assert batch_size % n_devices == 0, \
        f"batch_size ({batch_size}) must be divisible by n_devices ({n_devices})"

    output_dir, ckpt_dir, log_dir = create_output_dir(output_dir)
    summary_writer = tf.summary.create_file_writer(str(log_dir))

    # ---- Data ----
    ds_train, ds_val = build_dataset(data_dir, image_size, batch_size)

    # ---- Model & optimizer ----
    rng = jax.random.PRNGKey(0)
    rng, key_init, key_diffusion = jax.random.split(rng, 3)

    # Total steps for cosine decay schedule
    steps_per_epoch = sum(1 for _ in ds_train)
    total_steps = epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=1e-2  # final lr = learning_rate * alpha
    )
    tx = optax.adamw(schedule, weight_decay=weight_decay)

    model = DiffusionModel(
        feature_stages=[64, 128, 256, 512],
        blocks=2,
        num_classes=NUM_CLASSES,
        embedding_dims=64,
        class_embed_dims=64,
    )

    # Init with dummy grayscale batch
    dummy_images = jnp.ones((1, image_size, image_size, 1), dtype=jnp.float32)
    dummy_labels = jnp.zeros((1,), dtype=jnp.int32)
    variables = model.init(key_init, dummy_images, dummy_labels, key_diffusion,
                           train=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=tx,
    )

    # Replicate state across all devices
    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(variables['params'])

    # Orbax checkpoint manager
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=1)
    ckpt_manager = ocp.CheckpointManager(str(ckpt_dir), options=ckpt_options)

    steps_per_val = sum(1 for _ in ds_val)

    # ---- Training loop ----
    rng, rng_train, rng_val, rng_val_step = jax.random.split(rng, 4)

    for epoch in range(epochs):
        losses = []
        pbar = tqdm(ds_train.as_numpy_iterator(), desc=f'Epoch {epoch+1}/{epochs}',
                    total=steps_per_epoch)

        for images, labels in pbar:
            # Shard batch across devices: (B, H, W, C) -> (n_devices, B//n, H, W, C)
            images = images.reshape(n_devices, -1, *images.shape[1:])
            labels = labels.reshape(n_devices, -1)

            rng_train, key = jax.random.split(rng_train)
            # Replicate per-step rng across devices (each device gets same key — split inside pmap is fine)
            keys = jax.random.split(key, n_devices)

            state, loss = train_step(state, images, labels, keys, p_uncond)

            loss_val = float(jax_utils.unreplicate(loss))
            pbar.set_postfix({'loss': f'{loss_val:.5f}'})
            losses.append(loss_val)

            # EMA update on all replicas
            ema_params = jax.tree_util.tree_map(update_ema, ema_params, state.params)

        mean_loss = np.mean(losses)

        # Current LR from schedule (use unreplicated step count)
        current_step = int(jax_utils.unreplicate(state.step))
        current_lr = float(schedule(current_step))

        # ---- Validation loss on ds_val ----
        val_losses, val_noise_losses, val_image_losses = [], [], []
        for val_images, val_labels_batch in ds_val.as_numpy_iterator():
            val_images = val_images.reshape(n_devices, -1, *val_images.shape[1:])
            val_labels_batch = val_labels_batch.reshape(n_devices, -1)
            rng_val_step, vkey = jax.random.split(rng_val_step)
            vkeys = jax.random.split(vkey, n_devices)
            v_loss, v_noise_loss, v_image_loss = val_step(
                state, val_images, val_labels_batch, vkeys
            )
            val_losses.append(float(jax_utils.unreplicate(v_loss)))
            val_noise_losses.append(float(jax_utils.unreplicate(v_noise_loss)))
            val_image_losses.append(float(jax_utils.unreplicate(v_image_loss)))

        mean_val_loss = np.mean(val_losses)
        mean_val_noise_loss = np.mean(val_noise_losses)
        mean_val_image_loss = np.mean(val_image_losses)

        print(f'Epoch {epoch+1}: train_loss={mean_loss:.5f}  val_loss={mean_val_loss:.5f}  lr={current_lr:.2e}')

        # ---- Validation: generate one image per class ----
        rng_val, key_gen = jax.random.split(rng_val)
        # Generate 1 image per class (8 total), distribute evenly across devices
        val_labels = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        val_labels = val_labels.reshape(n_devices, -1)  # (8, 1) if n_devices==8
        key_gen_devices = jax.random.split(key_gen, n_devices)

        generated = generate_cfg_step(
            state, ema_params, key_gen_devices, val_labels,
            val_diffusion_steps, guidance_scale
        )
        # Collect from devices: (n_devices, B_per_device, H, W, 1) -> (N, H, W, 1)
        generated = generated.reshape(-1, image_size, image_size, 1)
        generated_np = np.array(generated)

        with summary_writer.as_default():
            tf.summary.scalar('train/loss', mean_loss, step=epoch)
            tf.summary.scalar('val/loss', mean_val_loss, step=epoch)
            tf.summary.scalar('val/noise_loss', mean_val_noise_loss, step=epoch)
            tf.summary.scalar('val/image_loss', mean_val_image_loss, step=epoch)
            gen_rgb = np.repeat(generated_np, 3, axis=-1)
            tf.summary.image('generated/per_class', gen_rgb,
                             step=epoch, max_outputs=NUM_CLASSES)

        # ---- W&B logging (per epoch, images every log_image_every epochs) ----
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": mean_loss,
            "train/lr": current_lr,
            "val/loss": mean_val_loss,
            "val/noise_loss": mean_val_noise_loss,
            "val/image_loss": mean_val_image_loss,
        }
        if (epoch + 1) % log_image_every == 0:
            log_dict["generated/per_class"] = [
                wandb.Image(generated_np[i, :, :, 0],
                            caption=f"sensor_{SENSORS[i]}")
                for i in range(NUM_CLASSES)
            ]
        wandb.log(log_dict, step=epoch + 1)

        save_ckpt(ckpt_manager, state, ema_params, step=epoch)

    ckpt_manager.wait_until_finished()
    wandb.finish()
    print('Training complete. Checkpoints saved to:', ckpt_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional DDIM with CFG - Fingerprint')

    # On Kaggle, dataset is mounted at /kaggle/input/<dataset-name>/images
    parser.add_argument('--data-dir', type=str,
                        default='/kaggle/input/nist-sd302a/images',
                        help='Path to the images/ directory of NIST SD 302a')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('-b', '--batch-size', type=int, default=512,
                        help='Total batch size across all devices (must be divisible by n_devices)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-diffusion-steps', type=int, default=80)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--p-uncond', type=float, default=0.1,
                        help='Probability of dropping class label to null during training')
    parser.add_argument('--log-image-every', type=int, default=5,
                        help='Log generated images to W&B every N epochs')
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('-o', '--output-dir', type=Path,
                        default=f'/kaggle/working/outputs/{now}')

    args = parser.parse_args()
    run(**vars(args))
