"""FID4K evaluation for fingerprint DDIM.

Generates 4000 images (500 per sensor class) with CFG, then computes FID
against the real NIST SD 302a training images.

Usage on Kaggle:
    python infer_fid.py \
        --ckpt-dir /kaggle/input/ddim-ckpt/499 \
        --data-dir /kaggle/input/nist-sd302a/images \
        --image-size 256 \
        --n-samples 4000 \
        --diffusion-steps 80 \
        --guidance-scale 3.0 \
        --out-dir /kaggle/working/fid_samples

Install deps (in addition to train deps):
    pip install -q torch torchvision  # needed for pytorch-fid
    pip install -q pytorch-fid
"""

import argparse
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import jax_utils
from flax.training import train_state
import optax
import orbax.checkpoint as ocp
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
import sys

from model import DiffusionModel
from train import make_full_pos, SENSORS, NUM_CLASSES


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_ema_params(ckpt_dir: str, model: DiffusionModel,
                    image_size: int):
    """Restore only ema_params from an orbax checkpoint directory."""
    dummy_images = jnp.ones((1, image_size, image_size, 1), jnp.float32)
    dummy_labels = jnp.zeros((1,), jnp.int32)
    dummy_pos = jnp.zeros((1, image_size, image_size, 2), jnp.float32)
    dummy_rng = jax.random.PRNGKey(0)

    variables = model.init(dummy_rng, dummy_images, dummy_labels,
                           dummy_pos, dummy_rng)
    dummy_ema = variables['params']

    ckpt_manager = ocp.CheckpointManager(
        str(ckpt_dir),
        options=ocp.CheckpointManagerOptions(),
    )
    step = ckpt_manager.latest_step()
    print(f"Restoring checkpoint step {step} from {ckpt_dir}")

    # Use partial_restore=True so orbax ignores extra keys on disk (state, opt_state)
    restored = ckpt_manager.restore(
        step,
        args=ocp.args.StandardRestore({'ema_params': dummy_ema}),
        partial_restore=True,
    )
    return restored['ema_params']


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(5, 6))
def generate_batch(state, ema_params, rng, class_labels, pos,
                   diffusion_steps: int, guidance_scale: float):
    variables = {'params': ema_params}
    image_shape = (class_labels.shape[0], pos.shape[1], pos.shape[2], 1)
    return state.apply_fn(
        variables, rng, image_shape, class_labels, pos,
        diffusion_steps, guidance_scale,
        method=DiffusionModel.generate_cfg,
    )


def generate_samples(ema_params, model, image_size: int,
                     n_per_class: int, diffusion_steps: int,
                     guidance_scale: float, out_dir: Path,
                     seed: int = 0):
    """Generate n_per_class images for each of the 8 sensor classes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sensor in SENSORS:
        (out_dir / sensor).mkdir(exist_ok=True)

    n_devices = jax.device_count()
    # batch_size must be divisible by n_devices and num_classes
    # generate per-class in rounds
    batch_per_device = max(1, 8 // n_devices)  # images per device per step
    batch_size = batch_per_device * n_devices

    full_pos = make_full_pos(image_size, batch_size)
    full_pos_sharded = full_pos.reshape(
        n_devices, batch_per_device, image_size, image_size, 2)

    # Replicate state (we only need ema_params + apply_fn via a dummy state)
    tx = optax.adamw(1e-4)
    dummy_images = jnp.ones((1, image_size, image_size, 1), jnp.float32)
    dummy_labels = jnp.zeros((1,), jnp.int32)
    dummy_pos = jnp.zeros((1, image_size, image_size, 2), jnp.float32)
    dummy_rng = jax.random.PRNGKey(seed)
    variables = model.init(dummy_rng, dummy_images, dummy_labels,
                           dummy_pos, dummy_rng)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    state = jax_utils.replicate(state)
    ema_params_rep = jax_utils.replicate(ema_params)

    rng = jax.random.PRNGKey(seed)
    total_saved = {s: 0 for s in SENSORS}
    total_target = n_per_class

    pbar = tqdm(total=NUM_CLASSES * total_target, desc='Generating')

    while any(v < total_target for v in total_saved.values()):
        for class_idx, sensor in enumerate(SENSORS):
            if total_saved[sensor] >= total_target:
                continue

            rng, key = jax.random.split(rng)
            keys = jax.random.split(key, n_devices)

            labels = np.full((batch_size,), class_idx, dtype=np.int32)
            labels_sharded = labels.reshape(n_devices, batch_per_device)

            generated = generate_batch(
                state, ema_params_rep, keys,
                labels_sharded, full_pos_sharded,
                diffusion_steps, guidance_scale,
            )
            # (n_devices, batch_per_device, H, W, 1) -> (batch_size, H, W, 1)
            generated = np.array(generated).reshape(
                batch_size, image_size, image_size, 1)

            for img_arr in generated:
                if total_saved[sensor] >= total_target:
                    break
                img = Image.fromarray(
                    (img_arr[:, :, 0] * 255).clip(0, 255).astype(np.uint8),
                    mode='L'
                )
                fname = out_dir / sensor / f'{total_saved[sensor]:05d}.png'
                img.save(fname)
                total_saved[sensor] += 1
                pbar.update(1)

    pbar.close()
    print(f"Generated images saved to {out_dir}")


# ---------------------------------------------------------------------------
# Real images: collect and save for FID reference
# ---------------------------------------------------------------------------

def save_real_images(data_dir: str, out_dir: Path, image_size: int,
                     n_per_class: int, seed: int = 0):
    """Resize real images to image_size and save as grayscale PNG for FID."""
    import random
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    for sensor in SENSORS:
        sensor_dir = data_dir / 'challengers' / sensor / 'roll' / 'png'
        if not sensor_dir.exists():
            print(f"WARNING: {sensor_dir} not found, skipping.")
            continue
        all_files = sorted(sensor_dir.glob('*.png'))
        sampled = random.sample(all_files, min(n_per_class, len(all_files)))

        out_sensor = out_dir / sensor
        out_sensor.mkdir(exist_ok=True)
        for i, fpath in enumerate(tqdm(sampled, desc=f'Real {sensor}',
                                        leave=False)):
            img = Image.open(fpath).convert('L')
            img = img.resize((image_size, image_size), Image.LANCZOS)
            img.save(out_sensor / f'{i:05d}.png')

    print(f"Real reference images saved to {out_dir}")


# ---------------------------------------------------------------------------
# FID computation via pytorch-fid
# ---------------------------------------------------------------------------

def compute_fid(real_dir: Path, fake_dir: Path, batch_size: int = 64,
                device: str = 'cpu'):
    """Run pytorch-fid between two flat directories of images.

    pytorch-fid expects two directories; we merge all class subdirs into
    flat dirs for a single FID score over the full dataset.
    """
    import shutil, tempfile

    # Flatten class subdirs into two temp flat dirs
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        flat_real = tmp / 'real'
        flat_fake = tmp / 'fake'
        flat_real.mkdir()
        flat_fake.mkdir()

        for sensor in SENSORS:
            for src, dst in [(real_dir / sensor, flat_real),
                             (fake_dir / sensor, flat_fake)]:
                if not src.exists():
                    continue
                for f in src.glob('*.png'):
                    shutil.copy(f, dst / f'{sensor}_{f.name}')

        cmd = [
            sys.executable, '-m', 'pytorch_fid',
            str(flat_real), str(flat_fake),
            '--batch-size', str(batch_size),
            '--device', device,
            '--dims', '2048',
            '--num-workers', '4',
        ]
        print('Running:', ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print('STDERR:', result.stderr)
            raise RuntimeError('pytorch-fid failed')

        # Parse FID value from output
        for line in result.stdout.splitlines():
            if 'FID' in line:
                print(f"\n>>> {line.strip()}")


# ---------------------------------------------------------------------------
# Per-class FID
# ---------------------------------------------------------------------------

def compute_fid_per_class(real_dir: Path, fake_dir: Path,
                           batch_size: int = 64, device: str = 'cpu'):
    """Compute FID separately for each sensor class."""
    fid_scores = {}
    for sensor in SENSORS:
        r = real_dir / sensor
        f = fake_dir / sensor
        if not r.exists() or not f.exists():
            print(f"Skipping {sensor}: directory missing")
            continue
        cmd = [
            sys.executable, '-m', 'pytorch_fid',
            str(r), str(f),
            '--batch-size', str(batch_size),
            '--device', device,
            '--dims', '2048',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if 'FID' in line:
                try:
                    score = float(line.strip().split()[-1])
                    fid_scores[sensor] = score
                    print(f"  Sensor {sensor}: FID = {score:.2f}")
                except ValueError:
                    pass

    if fid_scores:
        mean_fid = np.mean(list(fid_scores.values()))
        print(f"\n>>> Mean per-class FID: {mean_fid:.2f}")
    return fid_scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 4K samples + compute FID vs NIST SD 302a')

    parser.add_argument('--ckpt-dir', type=str, required=True,
                        help='Path to orbax checkpoint directory (e.g. .../models/499)')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to images/ directory of NIST SD 302a')
    parser.add_argument('--out-dir', type=Path,
                        default='/kaggle/working/fid_eval')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=4000,
                        help='Total samples to generate (split evenly across 8 classes)')
    parser.add_argument('--diffusion-steps', type=int, default=80)
    parser.add_argument('--guidance-scale', type=float, default=3.0)
    parser.add_argument('--fid-batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for pytorch-fid InceptionV3')
    parser.add_argument('--per-class-fid', action='store_true',
                        help='Also compute FID per sensor class')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    n_per_class = args.n_samples // NUM_CLASSES
    print(f"Generating {n_per_class} images per class "
          f"({n_per_class * NUM_CLASSES} total)")

    fake_dir = args.out_dir / 'fake'
    real_dir = args.out_dir / 'real'

    model = DiffusionModel(
        feature_stages=[64, 128, 256, 512],
        blocks=2,
        num_classes=NUM_CLASSES,
        embedding_dims=64,
        cond_dims=256,
    )

    # Load checkpoint
    ema_params = load_ema_params(args.ckpt_dir, model, args.image_size)

    # Generate fake images
    generate_samples(
        ema_params=ema_params,
        model=model,
        image_size=args.image_size,
        n_per_class=n_per_class,
        diffusion_steps=args.diffusion_steps,
        guidance_scale=args.guidance_scale,
        out_dir=fake_dir,
        seed=args.seed,
    )

    # Save reference real images
    save_real_images(
        data_dir=args.data_dir,
        out_dir=real_dir,
        image_size=args.image_size,
        n_per_class=n_per_class,
        seed=args.seed,
    )

    # Compute FID
    print("\n=== Overall FID4K ===")
    compute_fid(real_dir, fake_dir,
                batch_size=args.fid_batch_size,
                device=args.device)

    if args.per_class_fid:
        print("\n=== Per-class FID ===")
        compute_fid_per_class(real_dir, fake_dir,
                              batch_size=args.fid_batch_size,
                              device=args.device)
