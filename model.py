from dataclasses import field
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from flax import linen as nn


def sinusoidal_embedding(x,
                         min_freq: float = 1.0,
                         max_freq: float = 1000.0,
                         embedding_dims: int = 64):
    frequencies = jnp.exp(jnp.linspace(jnp.log(min_freq),
                                       jnp.log(max_freq),
                                       embedding_dims // 2))
    angular_speeds = 2.0 * jnp.pi * frequencies
    embeddings = jnp.concatenate([jnp.sin(angular_speeds * x),
                                  jnp.cos(angular_speeds * x)], axis=3)
    return embeddings  # (batch, 1, 1, embedding_dims)


class ResidualBlock(nn.Module):
    features: int
    cond_dims: int = 256

    @nn.compact
    def __call__(self, x, cond_emb):
        # AdaGN: conditioning -> per-channel scale & shift
        scale_shift = nn.Dense(self.features * 2)(nn.swish(cond_emb))
        scale, shift = jnp.split(scale_shift, 2, axis=-1)
        scale = scale[:, None, None, :]  # (B, 1, 1, features)
        shift = shift[:, None, None, :]  # (B, 1, 1, features)

        # Project input channels to self.features if needed (shared for residual)
        if x.shape[-1] != self.features:
            x = nn.Conv(self.features, kernel_size=(1, 1))(x)
        residual = x

        x = nn.GroupNorm(num_groups=min(8, self.features))(x)
        x = x * (1 + scale) + shift
        x = nn.Conv(self.features, (3, 3))(x)
        x = nn.swish(x)
        x = nn.Conv(self.features, (3, 3))(x)
        return x + residual


class DownBlock(nn.Module):
    features: int
    blocks: int
    cond_dims: int = 256

    @nn.compact
    def __call__(self, x, cond_emb) -> Tuple:
        skips = []
        for _ in range(self.blocks):
            x = ResidualBlock(self.features, self.cond_dims)(x, cond_emb)
            skips.append(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips


def upsample2d(x,
               scale: Union[int, Tuple[int, int]],
               method: str = 'bilinear'):
    b, h, w, c = x.shape
    if isinstance(scale, int):
        h_out, w_out = scale * h, scale * w
    elif len(scale) == 2:
        h_out, w_out = scale[0] * h, scale[1] * w
    else:
        raise ValueError('scale must be int or Tuple[int, int]')
    return jax.image.resize(x, shape=(b, h_out, w_out, c), method=method)


class UpBlock(nn.Module):
    features: int
    blocks: int
    cond_dims: int = 256

    @nn.compact
    def __call__(self, x, skips: List, cond_emb):
        x = upsample2d(x, scale=2, method='bilinear')
        for _ in range(self.blocks):
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = ResidualBlock(self.features, self.cond_dims)(x, cond_emb)
        return x


class UNet(nn.Module):
    feature_stages: List[int]
    blocks: int
    num_classes: int = 8
    min_freq: float = 1.0
    max_freq: float = 1000.0
    embedding_dims: int = 64
    cond_dims: int = 256

    @nn.compact
    def __call__(self, noisy_images, noise_variances, class_labels, pos):
        # Position conditioning: concat (x,y) coordinates to noisy image
        x_in = jnp.concatenate([noisy_images, pos], axis=-1)  # (B, H, W, 1+2)
        x = nn.Conv(self.feature_stages[0], (1, 1))(x_in)

        # Time embedding
        time_emb = sinusoidal_embedding(noise_variances,
                                        min_freq=self.min_freq,
                                        max_freq=self.max_freq,
                                        embedding_dims=self.embedding_dims)
        time_flat = time_emb.reshape(time_emb.shape[0], -1)  # (B, emb_dims)

        # Class embedding (num_classes index = null/unconditional token)
        class_emb = nn.Embed(self.num_classes + 1,
                             self.embedding_dims)(class_labels)  # (B, emb_dims)

        # Conditioning MLP: time + class -> cond vector
        cond = nn.Dense(self.cond_dims)(nn.swish(time_flat + class_emb))
        cond = nn.Dense(self.cond_dims)(nn.swish(cond))  # (B, cond_dims)

        # Encoder
        skip_stages = []
        for features in self.feature_stages[:-1]:
            x, skips = DownBlock(features, self.blocks, self.cond_dims)(x, cond)
            skip_stages.append(skips)

        # Bottleneck
        for _ in range(self.blocks):
            x = ResidualBlock(self.feature_stages[-1], self.cond_dims)(x, cond)

        # Decoder
        for features in reversed(self.feature_stages[:-1]):
            skips = skip_stages.pop()
            x = UpBlock(features, self.blocks, self.cond_dims)(x, skips, cond)

        # Grayscale output
        x = nn.Conv(1, (1, 1), kernel_init=nn.initializers.zeros)(x)
        return x


class DiffusionModel(nn.Module):
    feature_stages: List[int] = field(default_factory=lambda:
                                      [64, 128, 256, 512])
    blocks: int = 2
    num_classes: int = 8
    min_freq: float = 1.0
    max_freq: float = 1000.0
    embedding_dims: int = 64
    cond_dims: int = 256

    min_signal_rate: float = 0.02
    max_signal_rate: float = 0.95

    def setup(self):
        self.network = UNet(feature_stages=self.feature_stages,
                            blocks=self.blocks,
                            num_classes=self.num_classes,
                            min_freq=self.min_freq,
                            max_freq=self.max_freq,
                            embedding_dims=self.embedding_dims,
                            cond_dims=self.cond_dims)

    def __call__(self, images, class_labels, pos, rng):
        images = (images - 0.5) / 0.5  # [0,1] -> [-1,1]

        rng_noises, rng_times = jax.random.split(rng)
        noises = jax.random.normal(rng_noises, images.shape, images.dtype)
        diffusion_times = jax.random.uniform(rng_times,
                                             (images.shape[0], 1, 1, 1),
                                             images.dtype)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates,
                                                signal_rates, class_labels, pos)
        return noises, images, pred_noises, pred_images

    def diffusion_schedule(self, diffusion_times):
        start_angle = jnp.arccos(self.max_signal_rate)
        end_angle = jnp.arccos(self.min_signal_rate)
        diffusion_angles = start_angle \
            + diffusion_times * (end_angle - start_angle)
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)
        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates,
                class_labels, pos):
        pred_noises = self.network(noisy_images, noise_rates ** 2,
                                   class_labels, pos)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, class_labels, pos,
                          diffusion_steps):
        n_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            ones = jnp.ones((n_images, 1, 1, 1), dtype=initial_noise.dtype)
            diffusion_times = ones - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, class_labels, pos)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates \
                = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images \
                + next_noise_rates * pred_noises

        return pred_images

    def reverse_diffusion_cfg(self, initial_noise, class_labels, pos,
                               diffusion_steps, guidance_scale: float = 3.0):
        n_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        null_labels = jnp.full_like(class_labels, self.num_classes)

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images
            ones = jnp.ones((n_images, 1, 1, 1), dtype=initial_noise.dtype)
            diffusion_times = ones - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            pred_noises_cond, _ = self.denoise(
                noisy_images, noise_rates, signal_rates, class_labels, pos)
            pred_noises_uncond, _ = self.denoise(
                noisy_images, noise_rates, signal_rates, null_labels, pos)

            pred_noises = pred_noises_uncond \
                + guidance_scale * (pred_noises_cond - pred_noises_uncond)
            pred_images = (noisy_images - noise_rates * pred_noises) \
                / signal_rates

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates \
                = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images \
                + next_noise_rates * pred_noises

        return pred_images

    def denormalize(self, x):
        return x * 0.5 + 0.5

    def generate(self, rng, image_shape, class_labels, pos,
                 diffusion_steps: int):
        initial_noise = jax.random.normal(rng, image_shape)
        generated_images = self.reverse_diffusion(
            initial_noise, class_labels, pos, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return jnp.clip(generated_images, 0.0, 1.0)

    def generate_cfg(self, rng, image_shape, class_labels, pos,
                     diffusion_steps: int, guidance_scale: float = 3.0):
        initial_noise = jax.random.normal(rng, image_shape)
        generated_images = self.reverse_diffusion_cfg(
            initial_noise, class_labels, pos,
            diffusion_steps, guidance_scale)
        generated_images = self.denormalize(generated_images)
        return jnp.clip(generated_images, 0.0, 1.0)
