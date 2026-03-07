"""Conditional DDIM with Patch-Diffusion + CFG on Kaggle TPU v5e-8.

Dataset: NIST SD 302a fingerprint images (8 sensor classes, grayscale)
Training: multi-scale random patches [64, 128, 256] with position conditioning
TPU: 8-chip data-parallel via jax.pmap
"""

import argparse
import functools
import os
from pathlib import Path
from typing import Tuple
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
NULL_CLASS = NUM_CLASSES

PATCH_SIZES = [64, 128, 256]
PATCH_PROBS = [0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Patch-Diffusion helpers
# ---------------------------------------------------------------------------

def make_pos_grid(patch_size: int, offset_y: int, offset_x: int,
                  full_res: int) -> np.ndarray:
    """Position grid (ps, ps, 2) with (x, y) in [-1, 1]."""
    y = np.arange(patch_size, dtype=np.float32) + offset_y
    x = np.arange(patch_size, dtype=np.float32) + offset_x
    y = (y / (full_res - 1) - 0.5) * 2.0
    x = (x / (full_res - 1) - 0.5) * 2.0
    xx, yy = np.meshgrid(x, y)
    return np.stack([xx, yy], axis=-1)


def pachify_numpy(images: np.ndarray, patch_size: int,
                  full_res: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random-crop patches + position grids on CPU."""
    B, H, W, C = images.shape
    if patch_size >= H:
        pos_single = make_pos_grid(H, 0, 0, full_res)
        pos = np.broadcast_to(pos_single[None], (B, H, W, 2)).copy()
        return images, pos
    patches = np.empty((B, patch_size, patch_size, C), dtype=images.dtype)
    pos = np.empty((B, patch_size, patch_size, 2), dtype=np.float32)
    for i in range(B):
        oy = np.random.randint(0, H - patch_size + 1)
        ox = np.random.randint(0, W - patch_size + 1)
        patches[i] = images[i, oy:oy + patch_size, ox:ox + patch_size, :]
        pos[i] = make_pos_grid(patch_size, oy, ox, full_res)
    return patches, pos


def make_full_pos(image_size: int, batch_size: int) -> np.ndarray:
    """Full coordinate grid for inference / validation."""
    pos_single = make_pos_grid(image_size, 0, 0, image_size)
    return np.tile(pos_single[None], (batch_size, 1, 1, 1))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_file_label_lists(data_dir: str):
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
    image = tf.image.decode_png(raw, channels=1)
    image = tf.cast(image, tf.float32)
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    crop = tf.minimum(h, w)
    image = tf.image.crop_to_bounding_box(image,
                                          (h - crop) // 2,
                                          (w - crop) // 2,
                                          crop, crop)
    image = tf.image.resize(image, (image_size, image_size), antialias=True)
    image = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    return image, label


def _augment_flip(image, label):
    return tf.image.random_flip_left_right(image), label


def build_dataset(data_dir: str,
                  image_size: int = 256,
                  batch_size: int = 64,
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

    def make_split(split_paths, split_labels, shuffle: bool, augment: bool):
        ds = tf.data.Dataset.from_tensor_slices(
            (split_paths, tf.cast(split_labels, tf.int32))
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(split_paths), 10000), seed=seed)
        ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(split_paths), 5000))
        if augment:
            ds = ds.map(_augment_flip, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = make_split(train_paths, train_labels,
                          shuffle=True, augment=True)
    ds_val = make_split(val_paths, val_labels,
                        shuffle=False, augment=False)
    return ds_train, ds_val


# ---------------------------------------------------------------------------
# Loss & EMA
# ---------------------------------------------------------------------------

def l1_loss(predictions, targets):
    return jnp.abs(predictions - targets)


def update_ema(p_cur, p_new, momentum: float = 0.999):
    return momentum * p_cur + (1 - momentum) * p_new


# ---------------------------------------------------------------------------
# Train / eval steps (pmap)
# ---------------------------------------------------------------------------

@functools.partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(5,))
def train_step(state, images, pos, labels, rng, p_uncond: float = 0.1):
    rng, rng_drop = jax.random.split(rng)
    drop_mask = jax.random.bernoulli(rng_drop, p_uncond, labels.shape)
    labels_in = jnp.where(drop_mask, NULL_CLASS, labels)

    def loss_fn(params):
        outputs = state.apply_fn(
            {'params': params},
            images, labels_in, pos, rng,
        )
        noises, imgs, pred_noises, pred_images = outputs
        noise_loss = l1_loss(pred_noises, noises).mean()
        image_loss = l1_loss(pred_images, imgs).mean()
        return noise_loss + image_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss


@functools.partial(jax.pmap, axis_name='batch')
def val_step(state, images, pos, labels, rng):
    outputs = state.apply_fn(
        {'params': state.params},
        images, labels, pos, rng,
    )
    noises, imgs, pred_noises, pred_images = outputs
    noise_loss = l1_loss(pred_noises, noises).mean()
    image_loss = l1_loss(pred_images, imgs).mean()
    loss = noise_loss + image_loss
    return (jax.lax.pmean(loss, axis_name='batch'),
            jax.lax.pmean(noise_loss, axis_name='batch'),
            jax.lax.pmean(image_loss, axis_name='batch'))


@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(5, 6))
def generate_cfg_step(state, ema_params, rng, class_labels, pos,
                      diffusion_steps: int, guidance_scale: float):
    variables = {'params': ema_params}
    image_shape = (class_labels.shape[0], pos.shape[1], pos.shape[2], 1)
    generated = state.apply_fn(
        variables, rng, image_shape, class_labels, pos,
        diffusion_steps, guidance_scale,
        method=DiffusionModel.generate_cfg,
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
# Output helpers
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

    tf.config.experimental.set_visible_devices([], 'GPU')

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
            "cond_dims": 256,
            "patch_sizes": PATCH_SIZES,
            "patch_probs": PATCH_PROBS,
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

    steps_per_epoch = sum(1 for _ in ds_train)
    total_steps = epochs * steps_per_epoch

    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=total_steps,
        alpha=1e-2,
    )
    tx = optax.adamw(schedule, weight_decay=weight_decay)

    model = DiffusionModel(
        feature_stages=[64, 128, 256, 512],
        blocks=2,
        num_classes=NUM_CLASSES,
        embedding_dims=64,
        cond_dims=256,
    )

    dummy_images = jnp.ones((1, image_size, image_size, 1), jnp.float32)
    dummy_labels = jnp.zeros((1,), jnp.int32)
    dummy_pos = jnp.zeros((1, image_size, image_size, 2), jnp.float32)
    variables = model.init(key_init, dummy_images, dummy_labels,
                           dummy_pos, key_diffusion)

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )

    state = jax_utils.replicate(state)
    ema_params = jax_utils.replicate(variables['params'])

    ckpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=3, save_interval_steps=1)
    ckpt_manager = ocp.CheckpointManager(str(ckpt_dir), options=ckpt_options)

    # Pre-compute full pos grid for validation / generation
    full_pos_val = make_full_pos(image_size, batch_size)
    full_pos_gen = make_full_pos(image_size, NUM_CLASSES)

    # ---- Training loop ----
    rng, rng_train, rng_val, rng_val_step = jax.random.split(rng, 4)

    for epoch in range(epochs):
        losses = []
        pbar = tqdm(ds_train.as_numpy_iterator(),
                    desc=f'Epoch {epoch + 1}/{epochs}',
                    total=steps_per_epoch)

        for images, labels in pbar:
            # Multi-scale patch crop + position grid
            patch_size = int(np.random.choice(PATCH_SIZES, p=PATCH_PROBS))
            patches, pos = pachify_numpy(images, patch_size,
                                         full_res=image_size)

            patches = patches.reshape(
                n_devices, -1, patch_size, patch_size, 1)
            pos = pos.reshape(
                n_devices, -1, patch_size, patch_size, 2)
            labels_shard = labels.reshape(n_devices, -1)

            rng_train, key = jax.random.split(rng_train)
            keys = jax.random.split(key, n_devices)

            state, loss = train_step(state, patches, pos,
                                     labels_shard, keys, p_uncond)

            loss_val = float(jax_utils.unreplicate(loss))
            pbar.set_postfix({'loss': f'{loss_val:.5f}', 'ps': patch_size})
            losses.append(loss_val)

            ema_params = jax.tree_util.tree_map(
                update_ema, ema_params, state.params)

        mean_loss = np.mean(losses)
        current_step = int(jax_utils.unreplicate(state.step))
        current_lr = float(schedule(current_step))

        # ---- Validation loss (full images) ----
        val_losses, val_noise_losses, val_image_losses = [], [], []
        for val_images, val_labels_batch in ds_val.as_numpy_iterator():
            bs_actual = val_images.shape[0]
            val_pos = full_pos_val[:bs_actual]
            val_images = val_images.reshape(
                n_devices, -1, image_size, image_size, 1)
            val_pos = val_pos.reshape(
                n_devices, -1, image_size, image_size, 2)
            val_labels_batch = val_labels_batch.reshape(n_devices, -1)

            rng_val_step, vkey = jax.random.split(rng_val_step)
            vkeys = jax.random.split(vkey, n_devices)
            v_loss, v_noise, v_image = val_step(
                state, val_images, val_pos, val_labels_batch, vkeys)
            val_losses.append(float(jax_utils.unreplicate(v_loss)))
            val_noise_losses.append(float(jax_utils.unreplicate(v_noise)))
            val_image_losses.append(float(jax_utils.unreplicate(v_image)))

        mean_val = np.mean(val_losses) if val_losses else 0.0
        mean_val_noise = np.mean(val_noise_losses) if val_noise_losses else 0.0
        mean_val_image = np.mean(val_image_losses) if val_image_losses else 0.0

        print(f'Epoch {epoch + 1}: train={mean_loss:.5f}  '
              f'val={mean_val:.5f}  lr={current_lr:.2e}')

        # ---- Generate one image per class (full resolution) ----
        rng_val, key_gen = jax.random.split(rng_val)
        gen_labels = jnp.arange(NUM_CLASSES, dtype=jnp.int32)
        gen_labels = gen_labels.reshape(n_devices, -1)
        gen_pos = full_pos_gen.reshape(
            n_devices, -1, image_size, image_size, 2)
        key_gen_devices = jax.random.split(key_gen, n_devices)

        generated = generate_cfg_step(
            state, ema_params, key_gen_devices,
            gen_labels, gen_pos,
            val_diffusion_steps, guidance_scale,
        )
        generated = generated.reshape(-1, image_size, image_size, 1)
        generated_np = np.array(generated)

        # ---- TensorBoard ----
        with summary_writer.as_default():
            tf.summary.scalar('train/loss', mean_loss, step=epoch)
            tf.summary.scalar('val/loss', mean_val, step=epoch)
            tf.summary.scalar('val/noise_loss', mean_val_noise, step=epoch)
            tf.summary.scalar('val/image_loss', mean_val_image, step=epoch)
            gen_rgb = np.repeat(generated_np, 3, axis=-1)
            tf.summary.image('generated/per_class', gen_rgb,
                             step=epoch, max_outputs=NUM_CLASSES)

        # ---- W&B (sparse image logging) ----
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": mean_loss,
            "train/lr": current_lr,
            "val/loss": mean_val,
            "val/noise_loss": mean_val_noise,
            "val/image_loss": mean_val_image,
        }
        if (epoch + 1) % log_image_every == 0:
            log_dict["generated/per_class"] = [
                wandb.Image(generated_np[i, :, :, 0],
                            caption=f"sensor_{SENSORS[i]}")
                for i in range(min(NUM_CLASSES, generated_np.shape[0]))
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
    parser = argparse.ArgumentParser(
        description='Conditional DDIM + Patch-Diffusion + CFG — Fingerprint')

    parser.add_argument('--data-dir', type=str,
                        default='/kaggle/input/nist-sd302a/images')
    parser.add_argument('-e', '--epochs', type=int, default=500)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Total batch across all devices')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-diffusion-steps', type=int, default=80)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--p-uncond', type=float, default=0.1)
    parser.add_argument('--log-image-every', type=int, default=5)
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    parser.add_argument('-o', '--output-dir', type=Path,
                        default=f'/kaggle/working/outputs/{now}')

    args = parser.parse_args()
    run(**vars(args))
