# encoding/jax_base_converter.py

from flax import linen as nn
import jax.numpy as jnp
from typing import Any


class JaxBaseConverter(nn.Module):

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Any:
        raise NotImplementedError("__call__ must be implemented in subclass")

    def encode(self, signal: jnp.ndarray) -> Any:
        raise NotImplementedError("encode() must be implemented in subclass")

    def decode(self, spikes: jnp.ndarray) -> Any:
        raise NotImplementedError("decode() must be implemented in subclass")

    def optimize(self, data: jnp.ndarray) -> Any:
        raise NotImplementedError("optimize() must be implemented in subclass")