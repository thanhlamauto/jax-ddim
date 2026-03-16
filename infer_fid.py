"""FID evaluation for fingerprint DDIM — unconditional or CFG.

Two modes
---------
Unconditional (default, recommended for FID):
    Generates n_samples images using NULL_CLASS token (no class guidance),
    then computes FID against a random sample of real images from all sensors.

    python infer_fid.py \
        --ckpt-dir /kaggle/input/models/nhibui916/patch-diffusion-fingerprint/flax/default/1 \
        --data-dir /kaggle/input/fingerprint-challengers/images \
        --image-size 256 \
        --n-samples 8000 \
        --diffusion-steps 80 \
        --batch-per-device 16 \
        --unconditional \
        --out-dir /kaggle/working/fid_eval

CFG / class-conditional (legacy):
    Generates n_samples/8 images per sensor class with classifier-free guidance.

    python infer_fid.py \
        --ckpt-dir /kaggle/input/models/nhibui916/patch-diffusion-fingerprint/flax/default/1 \
        --data-dir /kaggle/input/fingerprint-challengers/images \
        --n-samples 8000 \
        --guidance-scale 3.0 \
        --out-dir /kaggle/working/fid_eval

Notes
-----
- --ckpt-dir must point to the *parent* directory that contains numbered
  step folders (e.g. .../1  which contains  .../1/499/).
  Orbax CheckpointManager will automatically pick the latest step (499).
- Install deps: pip install -q torch torchvision pytorch-fid
- TPU v5e-8 has 8 devices; use --batch-per-device 16 (128 imgs/step).
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

# Mirror constants from train.py to avoid importing wandb as a side-effect
SENSORS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
NUM_CLASSES = len(SENSORS)
NULL_CLASS = NUM_CLASSES  # index 8 = unconditional token


def make_full_pos(image_size: int, batch_size: int) -> np.ndarray:
    """Full coordinate grid (batch_size, H, W, 2) for inference."""
    y = np.arange(image_size, dtype=np.float32)
    x = np.arange(image_size, dtype=np.float32)
    y = (y / (image_size - 1) - 0.5) * 2.0
    x = (x / (image_size - 1) - 0.5) * 2.0
    xx, yy = np.meshgrid(x, y)
    pos_single = np.stack([xx, yy], axis=-1)   # (H, W, 2)
    return np.tile(pos_single[None], (batch_size, 1, 1, 1))


﻿# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_ema_params(ckpt_dir: str, model: DiffusionModel,
                    image_size: int):
    """Restore EMA params from an Orbax checkpoint directory.

    This implementation is deliberately conservative for maximum compatibility:
    it lets Orbax restore the full checkpoint with its default handler and then
    extracts EMA parameters from the restored structure, instead of trying to
    configure partial-restore arguments (which vary a lot between versions).
    """
    dummy_images = jnp.ones((1, image_size, image_size, 1), jnp.float32)
    dummy_labels = jnp.zeros((1,), jnp.int32)
    dummy_pos = jnp.zeros((1, image_size, image_size, 2), jnp.float32)
    dummy_rng = jax.random.PRNGKey(0)

    # Initialise model once so shapes are known (not strictly needed for restore,
    # but kept for consistency and potential future use).
    _ = model.init(dummy_rng, dummy_images, dummy_labels, dummy_pos, dummy_rng)

    ckpt_manager = ocp.CheckpointManager(
        str(ckpt_dir),
        options=ocp.CheckpointManagerOptions(),
    )
    step = ckpt_manager.latest_step()
    if step is None:
        raise RuntimeError(f"No checkpoint steps found under {ckpt_dir}")
    print(f"Restoring checkpoint step {step} from {ckpt_dir}")

    # Let Orbax decide how to restore; do not pass args/items so that this works
    # across different versions and composite handlers.
    restored = ckpt_manager.restore(step)

    # Common patterns:
    # 1) {'state': TrainState(...), 'ema_params': params}
    # 2) TrainState(...) with .params already being EMA
    # 3) {'ema_params': params}
    if isinstance(restored, dict):
        if 'ema_params' in restored:
            return restored['ema_params']
        if 'state' in restored:
            state = restored['state']
            if hasattr(state, 'params'):
                return state.params
        # Some handlers may wrap a single item under 'default'
        if 'default' in restored:
            default_item = restored['default']
            if isinstance(default_item, dict) and 'ema_params' in default_item:
                return default_item['ema_params']
            if hasattr(default_item, 'params'):
                return default_item.params

    # If it's not a dict, it might be a TrainState-like object.
    if hasattr(restored, 'params'):
        return restored.params

    raise ValueError(
        f"Could not find EMA parameters in restored checkpoint structure. "
        f"Keys/attributes available: {getattr(restored, 'keys', lambda: list(restored.__dict__.keys()))()}"
    )


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


@functools.partial(jax.pmap, axis_name='batch',
                   static_broadcasted_argnums=(4,))
def generate_batch_uncond(state, ema_params, rng, pos, diffusion_steps: int):
    """Unconditional generation: uses NULL_CLASS token, no CFG."""
    variables = {'params': ema_params}
    batch_per_device = pos.shape[0]
    image_shape = (batch_per_device, pos.shape[1], pos.shape[2], 1)
    null_labels = jnp.full((batch_per_device,), NULL_CLASS, dtype=jnp.int32)
    return state.apply_fn(
        variables, rng, image_shape, null_labels, pos,
        diffusion_steps,
        method=DiffusionModel.generate,
    )


def generate_samples(ema_params, model, image_size: int,
                     n_per_class: int, diffusion_steps: int,
                     guidance_scale: float, out_dir: Path,
                     seed: int = 0):
    """Generate n_per_class images for each of the 8 sensor classes (CFG)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sensor in SENSORS:
        (out_dir / sensor).mkdir(exist_ok=True)

    n_devices = jax.device_count()
    batch_per_device = max(1, 8 // n_devices)  # images per device per step
    batch_size = batch_per_device * n_devices

    full_pos = make_full_pos(image_size, batch_size)
    full_pos_sharded = full_pos.reshape(
        n_devices, batch_per_device, image_size, image_size, 2)

    state = _build_dummy_state(model, image_size, seed)
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


def _build_dummy_state(model, image_size, seed):
    """Shared helper: init model + replicate a dummy TrainState."""
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
    return jax_utils.replicate(state)


def generate_samples_uncond(ema_params, model, image_size: int,
                             n_samples: int, diffusion_steps: int,
                             out_dir: Path, batch_per_device: int = 16,
                             seed: int = 0):
    """Generate n_samples unconditional images (NULL_CLASS, no CFG) into out_dir.

    All images are saved flat (no sensor subdirectories).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_devices = jax.device_count()
    batch_size = batch_per_device * n_devices
    print(f"Unconditional generation: {n_samples} images, "
          f"{n_devices} devices × {batch_per_device} = {batch_size} imgs/step")

    full_pos = make_full_pos(image_size, batch_size)
    full_pos_sharded = full_pos.reshape(
        n_devices, batch_per_device, image_size, image_size, 2)

    state = _build_dummy_state(model, image_size, seed)
    ema_params_rep = jax_utils.replicate(ema_params)

    rng = jax.random.PRNGKey(seed)
    saved = 0
    pbar = tqdm(total=n_samples, desc='Generating (uncond)')

    while saved < n_samples:
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, n_devices)

        generated = generate_batch_uncond(
            state, ema_params_rep, keys,
            full_pos_sharded, diffusion_steps,
        )
        generated = np.array(generated).reshape(
            batch_size, image_size, image_size, 1)

        for img_arr in generated:
            if saved >= n_samples:
                break
            img = Image.fromarray(
                (img_arr[:, :, 0] * 255).clip(0, 255).astype(np.uint8),
                mode='L'
            )
            img.save(out_dir / f'{saved:05d}.png')
            saved += 1
            pbar.update(1)

    pbar.close()
    print(f"Unconditional images saved to {out_dir}")


# ---------------------------------------------------------------------------
# Real images: collect and save for FID reference
# ---------------------------------------------------------------------------

def save_real_images(data_dir: str, out_dir: Path, image_size: int,
                     n_per_class: int, seed: int = 0):
    """Resize real images to image_size and save as grayscale PNG for FID.

    Saves into per-sensor subdirectories (used with CFG / conditional mode).
    """
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


def save_real_images_flat(data_dir: str, out_dir: Path, image_size: int,
                          n_total: int, seed: int = 0):
    """Sample n_total real images from ALL sensor classes into a flat directory.

    Used for unconditional FID where generated images have no class structure.
    """
    import random
    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    all_files = []
    for sensor in SENSORS:
        sensor_dir = data_dir / 'challengers' / sensor / 'roll' / 'png'
        if not sensor_dir.exists():
            print(f"WARNING: {sensor_dir} not found, skipping.")
            continue
        all_files.extend(sorted(sensor_dir.glob('*.png')))

    if len(all_files) == 0:
        raise RuntimeError(f"No real images found under {data_dir}/challengers/")

    sampled = random.sample(all_files, min(n_total, len(all_files)))
    print(f"Sampling {len(sampled)}/{len(all_files)} real images for FID reference")

    for i, fpath in enumerate(tqdm(sampled, desc='Real images (flat)')):
        img = Image.open(fpath).convert('L')
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img.save(out_dir / f'{i:05d}.png')

    print(f"Real reference images saved to {out_dir}")


# ---------------------------------------------------------------------------
# FID computation via pytorch-fid
# ---------------------------------------------------------------------------

def _run_pytorch_fid(real_dir: Path, fake_dir: Path,
                     batch_size: int = 64, device: str = 'cpu'):
    """Low-level: run pytorch-fid on two *flat* image directories."""
    cmd = [
        sys.executable, '-m', 'pytorch_fid',
        str(real_dir), str(fake_dir),
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
    for line in result.stdout.splitlines():
        if 'FID' in line:
            print(f"\n>>> {line.strip()}")


def compute_fid(real_dir: Path, fake_dir: Path, batch_size: int = 64,
                device: str = 'cpu'):
    """Run pytorch-fid between real and fake image sets.

    Handles two layouts:
    - Per-class subdirs (CFG mode): flattens sensor/* into temporary flat dirs.
    - Flat dirs (unconditional mode): passes them directly to pytorch-fid.
    """
    import shutil, tempfile

    # Check if dirs contain sensor subdirectories
    has_subdirs = any((fake_dir / s).exists() for s in SENSORS)

    if has_subdirs:
        # CFG mode: flatten sensor subdirs into temporary flat dirs
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

            _run_pytorch_fid(flat_real, flat_fake, batch_size, device)
    else:
        # Unconditional mode: dirs are already flat
        _run_pytorch_fid(real_dir, fake_dir, batch_size, device)


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
        description='Generate samples + compute FID vs real fingerprint images')

    parser.add_argument('--ckpt-dir', type=str, required=True,
                        help='Parent orbax checkpoint dir containing numbered '
                             'step folders, e.g. '
                             '.../patch-diffusion-fingerprint/flax/default/1')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to directory that contains challengers/ '
                             'subdirectory with real images')
    parser.add_argument('--out-dir', type=Path,
                        default='/kaggle/working/fid_eval')
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--n-samples', type=int, default=8000,
                        help='Total samples to generate')
    parser.add_argument('--diffusion-steps', type=int, default=80)
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='CFG scale (only used without --unconditional)')
    parser.add_argument('--batch-per-device', type=int, default=16,
                        help='Images per TPU chip per step. '
                             'TPU v5e-8 has 8 chips → default 16 = 128 imgs/step')
    parser.add_argument('--unconditional', action='store_true',
                        help='Generate unconditionally (NULL_CLASS, no CFG). '
                             'Recommended for FID evaluation.')
    parser.add_argument('--fid-batch-size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device for pytorch-fid InceptionV3')
    parser.add_argument('--per-class-fid', action='store_true',
                        help='Also compute per-sensor FID (CFG mode only)')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

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

    if args.unconditional:
        print(f"\n=== Unconditional generation: {args.n_samples} images ===")
        generate_samples_uncond(
            ema_params=ema_params,
            model=model,
            image_size=args.image_size,
            n_samples=args.n_samples,
            diffusion_steps=args.diffusion_steps,
            out_dir=fake_dir,
            batch_per_device=args.batch_per_device,
            seed=args.seed,
        )
        save_real_images_flat(
            data_dir=args.data_dir,
            out_dir=real_dir,
            image_size=args.image_size,
            n_total=args.n_samples,
            seed=args.seed,
        )
    else:
        n_per_class = args.n_samples // NUM_CLASSES
        print(f"\n=== CFG generation: {n_per_class} images × {NUM_CLASSES} classes "
              f"= {n_per_class * NUM_CLASSES} total ===")
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
        save_real_images(
            data_dir=args.data_dir,
            out_dir=real_dir,
            image_size=args.image_size,
            n_per_class=n_per_class,
            seed=args.seed,
        )

    # Compute FID
    print("\n=== Overall FID ===")
    compute_fid(real_dir, fake_dir,
                batch_size=args.fid_batch_size,
                device=args.device)

    if args.per_class_fid and not args.unconditional:
        print("\n=== Per-class FID ===")
        compute_fid_per_class(real_dir, fake_dir,
                              batch_size=args.fid_batch_size,
                              device=args.device)
