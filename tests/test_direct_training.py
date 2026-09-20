"""Verify padding semantics in the exact visible training notebook cells."""
import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="JAX 심화 실습용 라이브러리입니다.")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax", reason="JAX 심화 실습용 라이브러리입니다.")

from notebook_sources.training import BATCHING, HEAD_INITIALIZATION


def notebook_namespace():
    namespace = {
        "jax": jax, "jnp": jnp, "np": np, "optax": optax,
        "device": jax.devices("cpu")[0], "classes": ["one", "two", "three"],
        "BATCH_SIZE": 4, "SEED": 42, "HEAD_LEARNING_RATE": 1e-3,
        "config": {"hidden_size": 4},
    }
    exec(BATCHING, namespace)
    return namespace


def test_visible_prediction_cells_drop_padding_rows():
    namespace = notebook_namespace()
    values = np.arange(20, dtype=np.float32).reshape(5, 4)
    observed = namespace["predict_batches"](lambda _, batch: batch * 2, {}, values)
    np.testing.assert_array_equal(observed, values * 2)
    assert observed.shape == (5, 4)


def test_visible_adam_step_ignores_masked_examples():
    namespace = notebook_namespace()

    def dense(weights, prefix, values):
        return values @ weights[prefix + ".weight"].T + weights[prefix + ".bias"]

    namespace["dense"] = dense
    exec(HEAD_INITIALIZATION, namespace)
    weights = namespace["head"]
    state = namespace["head_optimizer_state"]
    step = namespace["head_step"]
    features = jnp.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.1, 0.0]])
    labels = jnp.array([0, 2])
    clean_weights, _, clean_loss = step(weights, state, features, labels, jnp.ones(2))

    # Intentionally give padded rows very different inputs and labels. They must
    # not change the optimizer update, even though they produce finite losses.
    padded_features = jnp.concatenate([features, jnp.array([[9., 8., 7., 6.], [-9., -8., -7., -6.]])])
    padded_labels = jnp.array([0, 2, 1, 1])
    padded_weights, _, padded_loss = step(weights, state, padded_features, padded_labels,
                                          jnp.array([1., 1., 0., 0.]))
    assert float(padded_loss) == pytest.approx(float(clean_loss), rel=1e-6)
    for key in weights:
        np.testing.assert_allclose(padded_weights[key], clean_weights[key], atol=1e-7, rtol=1e-6)
