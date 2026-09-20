"""Maintainer reference copied into generated notebook cells; never imported by students.

Functional JAX implementation of the pinned Hugging Face DeiT checkpoint.

Weights retain their original names and layout. The first eleven transformer
blocks are frozen; the final block can be differentiated without rebuilding or
reinitializing the pretrained backbone. No PyTorch dependency is needed at run
time. GELU and LayerNorm match the original ViT implementation.
"""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

MODEL_ID = "facebook/deit-tiny-patch16-224"
MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"


def dense(params, prefix, x):
    return jnp.matmul(x, params[prefix + ".weight"].T,
                      precision=jax.lax.Precision.HIGHEST) + params[prefix + ".bias"]


def layer_norm(params, prefix, x, epsilon):
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * jax.lax.rsqrt(variance + epsilon) * params[prefix + ".weight"] + params[prefix + ".bias"]


def transformer_block(params, index, x, config):
    prefix = f"vit.encoder.layer.{index}"
    normalized = layer_norm(params, prefix + ".layernorm_before", x, config["layer_norm_eps"])
    batch, tokens, hidden = normalized.shape
    heads = config["num_attention_heads"]
    head_dim = hidden // heads

    def project(name):
        value = dense(params, prefix + ".attention.attention." + name, normalized)
        return value.reshape(batch, tokens, heads, head_dim).transpose(0, 2, 1, 3)

    query, key, value = (project(name) for name in ("query", "key", "value"))
    scores = jnp.matmul(query, key.swapaxes(-1, -2), precision=jax.lax.Precision.HIGHEST)
    probabilities = jax.nn.softmax(scores / jnp.sqrt(jnp.float32(head_dim)), axis=-1)
    context = jnp.matmul(probabilities, value, precision=jax.lax.Precision.HIGHEST)
    context = context.transpose(0, 2, 1, 3).reshape(batch, tokens, hidden)
    x = x + dense(params, prefix + ".attention.output.dense", context)
    normalized = layer_norm(params, prefix + ".layernorm_after", x, config["layer_norm_eps"])
    intermediate = jax.nn.gelu(dense(params, prefix + ".intermediate.dense", normalized), approximate=False)
    return x + dense(params, prefix + ".output.dense", intermediate)


def prefix_tokens(params, images, config):
    """Patch embedding plus blocks 0..10; this output is safe to cache."""
    if images.ndim != 4 or images.shape[1:] != (224, 224, 3):
        raise ValueError("Expected preprocessed NHWC images [N, 224, 224, 3]")
    x = jax.lax.conv_general_dilated(
        images.astype(jnp.float32),
        params["vit.embeddings.patch_embeddings.projection.weight"].transpose(2, 3, 1, 0),
        window_strides=(config["patch_size"], config["patch_size"]),
        padding="VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=jax.lax.Precision.HIGHEST,
    ) + params["vit.embeddings.patch_embeddings.projection.bias"]
    x = x.reshape(x.shape[0], -1, config["hidden_size"])
    cls = jnp.broadcast_to(params["vit.embeddings.cls_token"], (x.shape[0], 1, config["hidden_size"]))
    x = jnp.concatenate([cls, x], axis=1) + params["vit.embeddings.position_embeddings"]
    for index in range(config["num_hidden_layers"] - 1):
        x = transformer_block(params, index, x, config)
    return x


def tail_features(params, tokens, config):
    x = transformer_block(params, config["num_hidden_layers"] - 1, tokens, config)
    return layer_norm(params, "vit.layernorm", x, config["layer_norm_eps"])[:, 0]


def tail_logits(params, tokens, config):
    return dense(params, "classifier", tail_features(params, tokens, config))


def original_logits(params, images, config):
    return tail_logits(params, prefix_tokens(params, images, config), config)


def is_tail_parameter(name, config):
    return (name.startswith(f"vit.encoder.layer.{config['num_hidden_layers'] - 1}.")
            or name.startswith("vit.layernorm.") or name.startswith("classifier."))


def extract_tail(params, config):
    return {key: value for key, value in params.items() if is_tail_parameter(key, config)}


def initialize_head(class_count, hidden_size, seed=42, device=None):
    if class_count < 2:
        raise ValueError("Fine-tuning requires at least two classes")
    weight = 0.02 * jax.random.normal(jax.random.PRNGKey(seed), (class_count, hidden_size))
    return {"classifier.weight": jax.device_put(weight, device),
            "classifier.bias": jax.device_put(jnp.zeros(class_count, jnp.float32), device)}


def parameter_digest(params):
    digest = hashlib.sha256()
    for key in sorted(params):
        value = np.ascontiguousarray(jax.device_get(params[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(path, tail, metadata):
    """Store only the adapted final block, norm and head; source stays unchanged."""
    arrays = {key: np.asarray(jax.device_get(value)) for key, value in tail.items()}
    arrays["__metadata__"] = np.array(json.dumps(metadata, ensure_ascii=False))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path, source_params, config, device=None):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["__metadata__"]))
        if metadata.get("model_revision") != MODEL_REVISION:
            raise ValueError("Checkpoint revision does not match the pinned source")
        expected = set(extract_tail(source_params, config))
        actual = set(archive.files) - {"__metadata__"}
        if actual != expected:
            raise ValueError("Checkpoint tail parameter keys do not match the architecture")
        tail = {key: jax.device_put(archive[key], device) for key in actual}
    classes = metadata.get("classes", [])
    for key in expected:
        shape = ((len(classes), config["hidden_size"]) if key == "classifier.weight"
                 else (len(classes),) if key == "classifier.bias" else source_params[key].shape)
        if tail[key].shape != shape:
            raise ValueError(f"Checkpoint parameter shape mismatch: {key}")
    return {**source_params, **tail}, metadata
