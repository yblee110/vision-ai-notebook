"""Authoring source for the two visible, direct-library training notebooks.

These functions assemble cells at build time. The exported notebooks contain
the full Python code and do not import this module or ``vision_lab``.
"""

from .common import (
    boot_cells,
    code,
    device_cells,
    make_notebook,
    md,
    model_definition_cells,
    prepared_data_cells,
    preprocessing_cells,
    weight_loading_cells,
)


SETTINGS = r'''
import csv
import time
import optax  # 교차 엔트로피와 Adam 업데이트

SEED = 42
BATCH_SIZE = 16
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 3
HEAD_LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-4
if len(classes) < 2:
    raise ValueError("분류할 클래스가 2개 이상 필요합니다.")

# 새 실험을 할 때는 OUTPUT_DIR을 다른 경로로 바꾸세요.
# VISION_OUTPUT_DIR은 강사의 자동 검증에서도 같은 설정을 지정하는 방법입니다.
OUTPUT_DIR = Path(os.environ.get("VISION_OUTPUT_DIR", str(ROOT / "results" / device.platform)))
if (OUTPUT_DIR / "report.json").exists():
    raise FileExistsError(f"완료된 결과가 있습니다. OUTPUT_DIR을 바꾸세요: {OUTPUT_DIR}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CPU 검증은 강사가 명시적으로 선택했을 때만 가능합니다.
# 기본 GPU/TPU 수업에서는 데이터를 줄이지 않습니다.
CPU_SMOKE_PER_CLASS = int(os.environ.get("VISION_SMOKE_PER_CLASS", "0"))
if CPU_SMOKE_PER_CLASS < 0:
    raise ValueError("VISION_SMOKE_PER_CLASS는 0 이상이어야 합니다.")
if CPU_SMOKE_PER_CLASS and device.platform != "cpu":
    raise ValueError("작은 데이터 검증은 CPU에서만 허용합니다.")
if CPU_SMOKE_PER_CLASS:
    for name, split in splits.items():
        selected = np.concatenate([
            np.flatnonzero(split["labels"] == label)[:CPU_SMOKE_PER_CLASS]
            for label in range(len(classes))
        ])
        splits[name] = {
            "images": split["images"][selected],
            "labels": split["labels"][selected],
            "ids": [split["ids"][int(index)] for index in selected],
        }
print("실제 장치:", device.platform, "| 결과:", OUTPUT_DIR)
print("분할별 이미지 수:", {name: len(split["labels"]) for name, split in splits.items()})
print("CPU 검증은 GPU·TPU 실습 완료로 인정하지 않습니다." if device.platform == "cpu" else "가속기 실습")
'''

BATCH_ITERATOR = r'''
def batches(values, batch_size, indices=None):
    """마지막 배치도 같은 크기로 만들고, 추가한 행은 mask=0으로 표시합니다."""
    if batch_size < 1 or len(values) < 1:
        raise ValueError("배치와 데이터 크기는 양수여야 합니다.")
    order = np.arange(len(values)) if indices is None else np.asarray(indices)
    for start in range(0, len(order), batch_size):
        actual = order[start:start + batch_size]
        padded = np.pad(actual, (0, batch_size - len(actual)), mode="edge")
        mask = (np.arange(batch_size) < len(actual)).astype(np.float32)
        yield values[padded], padded, mask, len(actual)
'''

PREDICT_BATCHES = r'''
def predict_batches(forward, weights, values):
    outputs = []
    for batch, _, _, count in batches(values, BATCH_SIZE):
        logits = forward(weights, jax.device_put(batch, device))
        outputs.append(np.asarray(jax.device_get(logits))[:count])
    return np.concatenate(outputs)
'''

CLASSIFICATION_METRICS = r'''
def classification_metrics(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0 or logits.shape != (len(labels), len(classes)):
        raise ValueError("예측과 정답의 크기가 다릅니다.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    predictions = logits.argmax(axis=1)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * np.diag(confusion).astype(float), denominator,
                   out=np.zeros(len(classes)), where=denominator != 0)
    return {
        "loss": float(-log_probs[np.arange(len(labels)), labels].mean()),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(f1.mean()),
        "count": int(len(labels)),
        "confusion_matrix": confusion.tolist(),
    }
'''

# The test imports this concatenation; the student notebook uses three cells.
BATCHING = "\n".join([BATCH_ITERATOR, PREDICT_BATCHES, CLASSIFICATION_METRICS])

CACHE = r'''
# 학습하지 않는 부분: patch embedding + Transformer block 0~10
frozen_prefix = {name: value for name, value in params.items()
                 if not is_tail_parameter(name, config)}
prefix_hash_before = parameter_digest(frozen_prefix)
prefix_forward = jax.jit(lambda weights, pixels: prefix_tokens(weights, pixels, config))
feature_forward = jax.jit(lambda weights, tokens: tail_features(weights, tokens, config))

# 같은 입력에 같은 출력을 내는 부분만 캐시합니다. 데이터 증강은 사용하지 않습니다.
# test에서도 입력 특징만 계산하며, 평가 결과로 모델을 선택하지 않습니다.
cache = {}
for split_name, split in splits.items():
    started = time.perf_counter()
    token_parts = []
    for raw_images, _, _, count in batches(split["images"], BATCH_SIZE):
        pixels = preprocess(raw_images)  # Pillow: RGB/224x224, NumPy: 정규화
        tokens = prefix_forward(params, jax.device_put(pixels, device))
        token_parts.append(np.asarray(jax.device_get(tokens))[:count])
    tokens = np.concatenate(token_parts)
    features = predict_batches(feature_forward, params, tokens)
    cache[split_name] = {
        "tokens": tokens, "features": features,
        "labels": split["labels"], "ids": split["ids"],
    }
    print(f"{split_name}: tokens {tokens.shape}, features {features.shape}, "
          f"{time.perf_counter() - started:.1f}초", flush=True)
'''

ORIGINAL_PREDICTIONS = r'''
# 원래 1000개 ImageNet 클래스의 추론 결과를 살펴봅니다.
# 이 값은 새 데이터의 클래스 정확도와 비교하는 기준선이 아닙니다.
example_indices = [int(np.flatnonzero(cache["train"]["labels"] == label)[0])
                   for label in range(len(classes))]
original_head = {name: value for name, value in params.items() if name.startswith("classifier.")}
imagenet_logits = dense(original_head, "classifier",
                       jax.device_put(cache["train"]["features"][example_indices], device))
imagenet_probabilities = np.asarray(jax.device_get(jax.nn.softmax(imagenet_logits, axis=-1)))
original_predictions = []
for index, probabilities in zip(example_indices, imagenet_probabilities):
    top5 = probabilities.argsort()[-5:][::-1]
    row = {
        "image_id": str(cache["train"]["ids"][index]),
        "workbench_label": classes[int(cache["train"]["labels"][index])],
        "top5_imagenet": [{"label": config["id2label"][str(int(label))],
                            "probability": float(probabilities[label])} for label in top5],
    }
    original_predictions.append(row)
    print(row["workbench_label"], "→", row["top5_imagenet"][0])
'''

HEAD_INITIALIZATION = r'''
# 기존 1000개 클래스 head 대신, manifest의 클래스 수에 맞게 새 head를 만듭니다.
with jax.default_device(device):
    head = {
        "classifier.weight": 0.02 * jax.random.normal(
            jax.random.PRNGKey(SEED), (len(classes), config["hidden_size"])),
        "classifier.bias": jnp.zeros(len(classes), dtype=jnp.float32),
    }
head_optimizer = optax.adam(HEAD_LEARNING_RATE)
head_optimizer_state = head_optimizer.init(head)
head_forward = jax.jit(lambda weights, features: dense(weights, "classifier", features))


@jax.jit
def head_step(weights, optimizer_state, features, labels, mask):
    def loss_fn(candidate_weights):
        logits = dense(candidate_weights, "classifier", features)
        per_image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        # 패딩 행을 제외한 실제 이미지 수로 나눕니다.
        return jnp.sum(per_image_loss * mask) / jnp.sum(mask)

    loss, gradients = jax.value_and_grad(loss_fn)(weights)
    updates, optimizer_state = head_optimizer.update(gradients, optimizer_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, optimizer_state, loss

print("head에서 학습할 가중치 수:", sum(value.size for value in head.values()))
'''

HEAD_LOOP = r'''
head_rows = []
best_head = None
best_head_loss = float("inf")
best_head_epoch = None
head_rng = np.random.default_rng(SEED)

for epoch in range(1, HEAD_EPOCHS + 1):
    started = time.perf_counter()
    order = head_rng.permutation(len(cache["train"]["labels"]))
    total_loss = 0.0
    for features, indices, mask, count in batches(cache["train"]["features"], BATCH_SIZE, order):
        head, head_optimizer_state, loss = head_step(
            head, head_optimizer_state,
            jax.device_put(features, device),
            jax.device_put(cache["train"]["labels"][indices], device),
            jax.device_put(mask, device),
        )
        total_loss += float(loss) * count

    validation_logits = predict_batches(head_forward, head, cache["validation"]["features"])
    validation_metrics = classification_metrics(validation_logits, cache["validation"]["labels"])
    if not np.isfinite(total_loss) or not np.isfinite(validation_metrics["loss"]):
        raise FloatingPointError("head 학습 손실이 유한한 값이 아닙니다.")
    if validation_metrics["loss"] < best_head_loss:
        best_head_loss = validation_metrics["loss"]
        best_head_epoch = epoch
        best_head = dict(head)  # JAX 배열은 불변이므로 이후 업데이트가 이 값을 바꾸지 않습니다.

    row = {
        "stage": "head", "epoch": epoch,
        "train_loss": total_loss / len(cache["train"]["labels"]),
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    head_rows.append(row)
    print(f"head {epoch}/{HEAD_EPOCHS}: train loss {row['train_loss']:.4f}, "
          f"validation loss {row['validation_loss']:.4f}, "
          f"accuracy {row['validation_accuracy']:.3f}", flush=True)

head_info = {
    "best_epoch": best_head_epoch, "validation_loss": best_head_loss,
    "selection_rule": "lowest validation cross entropy after each epoch",
}
print("검증 손실로 선택한 head epoch:", best_head_epoch)
'''

FINETUNE_INITIALIZATION = r'''
# 마지막 Transformer block + 최종 LayerNorm + 학습된 head만 업데이트합니다.
baseline_tail = {**extract_tail(params, config), **best_head}
tuned_tail = dict(baseline_tail)
trainable_count = sum(int(value.size) for value in tuned_tail.values())
backbone_tail_count = sum(int(value.size) for name, value in baseline_tail.items()
                          if not name.startswith("classifier."))
expected_trainable_count = backbone_tail_count + len(classes) * (config["hidden_size"] + 1)
if trainable_count != expected_trainable_count:
    raise ValueError("마지막 블록·LayerNorm·head의 가중치 수가 맞지 않습니다.")

finetune_optimizer = optax.adam(FINETUNE_LEARNING_RATE)
finetune_optimizer_state = finetune_optimizer.init(tuned_tail)
finetune_forward = jax.jit(lambda weights, tokens: tail_logits(weights, tokens, config))


@jax.jit
def finetune_step(weights, optimizer_state, tokens, labels, mask):
    def loss_fn(candidate_weights):
        logits = tail_logits(candidate_weights, tokens, config)
        per_image_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return jnp.sum(per_image_loss * mask) / jnp.sum(mask)

    loss, gradients = jax.value_and_grad(loss_fn)(weights)
    updates, optimizer_state = finetune_optimizer.update(gradients, optimizer_state, weights)
    weights = optax.apply_updates(weights, updates)
    return weights, optimizer_state, loss

print("파인튜닝할 가중치 수:", trainable_count)
print("고정할 가중치 수:", sum(value.size for value in frozen_prefix.values()))
'''

FINETUNE_LOOP = r'''
finetune_rows = []
best_tuned_tail = None
best_finetune_loss = float("inf")
best_finetune_epoch = None
finetune_rng = np.random.default_rng(SEED + 1)

for epoch in range(1, FINETUNE_EPOCHS + 1):
    started = time.perf_counter()
    order = finetune_rng.permutation(len(cache["train"]["labels"]))
    total_loss = 0.0
    for tokens, indices, mask, count in batches(cache["train"]["tokens"], BATCH_SIZE, order):
        tuned_tail, finetune_optimizer_state, loss = finetune_step(
            tuned_tail, finetune_optimizer_state,
            jax.device_put(tokens, device),
            jax.device_put(cache["train"]["labels"][indices], device),
            jax.device_put(mask, device),
        )
        total_loss += float(loss) * count

    validation_logits = predict_batches(finetune_forward, tuned_tail, cache["validation"]["tokens"])
    validation_metrics = classification_metrics(validation_logits, cache["validation"]["labels"])
    if not np.isfinite(total_loss) or not np.isfinite(validation_metrics["loss"]):
        raise FloatingPointError("파인튜닝 손실이 유한한 값이 아닙니다.")
    if validation_metrics["loss"] < best_finetune_loss:
        best_finetune_loss = validation_metrics["loss"]
        best_finetune_epoch = epoch
        best_tuned_tail = dict(tuned_tail)

    row = {
        "stage": "finetune", "epoch": epoch,
        "train_loss": total_loss / len(cache["train"]["labels"]),
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    finetune_rows.append(row)
    print(f"finetune {epoch}/{FINETUNE_EPOCHS}: train loss {row['train_loss']:.4f}, "
          f"validation loss {row['validation_loss']:.4f}, "
          f"accuracy {row['validation_accuracy']:.3f}", flush=True)

finetune_info = {
    "best_epoch": best_finetune_epoch, "validation_loss": best_finetune_loss,
    "selection_rule": "lowest validation cross entropy after each epoch",
}
print("검증 손실로 선택한 파인튜닝 epoch:", best_finetune_epoch)
'''

TEST_AND_VERIFY = r'''
# 모델 선택이 모두 끝난 뒤, 같은 test 이미지로 두 모델을 평가합니다.
head_test_logits = predict_batches(head_forward, best_head, cache["test"]["features"])
finetune_test_logits = predict_batches(finetune_forward, best_tuned_tail, cache["test"]["tokens"])
head_info["test"] = classification_metrics(head_test_logits, cache["test"]["labels"])
finetune_info["test"] = classification_metrics(finetune_test_logits, cache["test"]["labels"])

# 첫 11개 블록이 그대로인지, 마지막 블록은 실제로 바뀌었는지 검사합니다.
merged_params = {**params, **best_tuned_tail}
prefix_after = {name: value for name, value in merged_params.items()
                if not is_tail_parameter(name, config)}
prefix_hash_after = parameter_digest(prefix_after)
last_block_prefix = f"vit.encoder.layer.{config['num_hidden_layers'] - 1}."
last_block_delta_l2 = float(np.sqrt(sum(
    np.sum(np.square(np.asarray(jax.device_get(best_tuned_tail[name]))
                     - np.asarray(jax.device_get(value))), dtype=np.float64)
    for name, value in params.items() if name.startswith(last_block_prefix)
)))
if prefix_hash_before != prefix_hash_after:
    raise RuntimeError("고정해야 할 앞부분의 가중치가 바뀌었습니다.")
if not np.isfinite(last_block_delta_l2) or last_block_delta_l2 <= 0:
    raise RuntimeError("마지막 블록의 가중치가 갱신되지 않았습니다.")

for name, info in [("head 학습 후", head_info), ("마지막 블록 파인튜닝 후", finetune_info)]:
    print(f"{name}: test accuracy {info['test']['accuracy']:.1%}, "
          f"macro F1 {info['test']['macro_f1']:.4f}")
print("고정 가중치 동일:", prefix_hash_before == prefix_hash_after)
print("마지막 블록의 변화량 L2:", last_block_delta_l2)
'''

PLOT_RESULTS = r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
for axis, title, info in zip(axes, ["Trained head (frozen backbone)", "Last-block fine-tuning"],
                           [head_info, finetune_info]):
    confusion = np.asarray(info["test"]["confusion_matrix"])
    axis.imshow(confusion, cmap="Blues", vmin=0, vmax=max(1, confusion.sum(axis=1).max()))
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center",
                      color="white" if confusion[row, column] > confusion.sum(axis=1).max() / 2 else "black")
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)),
             xticklabels=classes, yticklabels=classes,
             xlabel="Predicted", ylabel="Actual", title=f"{title}\nAccuracy {info['test']['accuracy']:.1%}")
    axis.tick_params(axis="x", rotation=35)
fig.savefig(OUTPUT_DIR / "confusion_comparison.png", dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
for axis, title, rows in zip(axes, ["Head training", "Last-block fine-tuning"],
                           [head_rows, finetune_rows]):
    epochs = [row["epoch"] for row in rows]
    axis.plot(epochs, [row["train_loss"] for row in rows], "o-", label="train loss")
    axis.plot(epochs, [row["validation_loss"] for row in rows], "o-", label="validation loss")
    axis.set(title=title, xlabel="Epoch", ylabel="Cross entropy", xticks=epochs)
    axis.legend()
fig.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
plt.show()
'''

SAVE_CHECKPOINTS = r'''
weights_sha256 = file_sha256(ROOT / "assets/pretrained/model.safetensors")
manifest_sha256 = file_sha256(DATA_PATH / "manifest.json")
checkpoint_metadata = {
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "weights_sha256": weights_sha256, "classes": classes,
    "manifest_sha256": manifest_sha256,
    "preprocessing": "pinned preprocessor_config.json; RGB NHWC 224x224 float32",
}

# np.savez_compressed: 학습한 마지막 부분과 클래스 순서를 함께 저장합니다.
# head_checkpoint에도 마지막 블록을 넣어 추론할 때 복원 형식을 맞춥니다.
for filename, stage, selected_weights in [
    ("head_checkpoint.npz", "head", baseline_tail),
    ("finetuned_checkpoint.npz", "finetune", best_tuned_tail),
]:
    arrays = {name: np.asarray(jax.device_get(value)) for name, value in selected_weights.items()}
    arrays["__metadata__"] = np.array(json.dumps({**checkpoint_metadata, "stage": stage}, ensure_ascii=False))
    np.savez_compressed(OUTPUT_DIR / filename, **arrays)
'''

SAVE_CSV = r'''
with (OUTPUT_DIR / "training.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=["stage", "epoch", "train_loss", "validation_loss",
                                               "validation_accuracy", "elapsed_seconds"])
    writer.writeheader()
    writer.writerows(head_rows + finetune_rows)

head_probabilities = np.asarray(jax.nn.softmax(head_test_logits, axis=-1))
finetune_probabilities = np.asarray(jax.nn.softmax(finetune_test_logits, axis=-1))
prediction_rows = []
for index, true_label in enumerate(cache["test"]["labels"]):
    before = int(head_test_logits[index].argmax())
    after = int(finetune_test_logits[index].argmax())
    prediction_rows.append({
        "image_id": str(cache["test"]["ids"][index]), "true_label": classes[int(true_label)],
        "head_prediction": classes[before], "finetune_prediction": classes[after],
        "head_confidence": float(head_probabilities[index, before]),
        "finetune_confidence": float(finetune_probabilities[index, after]),
    })
with (OUTPUT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
    writer.writeheader()
    writer.writerows(prediction_rows)

for name, info in [("head", head_info), ("finetune", finetune_info)]:
    with (OUTPUT_DIR / f"confusion_{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true/predicted"] + classes)
        writer.writerows([[label] + row for label, row in zip(classes, info["test"]["confusion_matrix"])])

(OUTPUT_DIR / "original_predictions.json").write_text(
    json.dumps(original_predictions, ensure_ascii=False, indent=2), encoding="utf-8")
print("모델·학습 이력·예측을 저장했습니다:", OUTPUT_DIR)
'''

REPORT = r'''
# 실행 상태를 JSON으로 남깁니다. CLI 종료 코드만으로 성공을 판단하지 않습니다.
accelerator_verified = device.platform in {"gpu", "tpu"}
report = {
    "schema_version": 2,
    "status": "completed" if accelerator_verified else ("cpu_smoke_only" if CPU_SMOKE_PER_CLASS else "cpu_validation"),
    "pipeline_completed": True,
    "accelerator_verified": accelerator_verified,
    "classes": classes,
    "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weights_sha256": weights_sha256},
    "dataset": {
        "manifest_sha256": manifest_sha256, "classes": classes,
        "split_counts": {name: len(split["labels"]) for name, split in splits.items()},
        "source": manifest.get("source"),
        "split_ids": {name: [str(value) for value in split["ids"]] for name, split in splits.items()},
    },
    "config": {
        "seed": SEED, "head_epochs": HEAD_EPOCHS, "finetune_epochs": FINETUNE_EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": FINETUNE_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE, "cpu_smoke_per_class": CPU_SMOKE_PER_CLASS,
        "dtype": "float32", "optimizer": "Adam",
        "augmentation": "none: deterministic frozen-prefix cache",
        "checkpoint_selection": "validation_loss_only",
    },
    "device": device_info,
    "stages": {"head": head_info, "finetune": finetune_info},
    "verification": {
        "frozen_prefix_sha256_before": prefix_hash_before,
        "frozen_prefix_sha256_after": prefix_hash_after,
        "frozen_prefix_unchanged": prefix_hash_before == prefix_hash_after,
        "last_block_delta_l2": last_block_delta_l2,
        "last_block_changed": last_block_delta_l2 > 0,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": sum(int(value.size) for value in frozen_prefix.values()),
    },
    "artifacts": {
        "head_checkpoint": "head_checkpoint.npz", "finetuned_checkpoint": "finetuned_checkpoint.npz",
        "training": "training.csv", "predictions": "predictions.csv",
        "original_predictions": "original_predictions.json",
        "confusion_head": "confusion_head.csv", "confusion_finetune": "confusion_finetune.csv",
        "confusion_plot": "confusion_comparison.png", "learning_curves": "learning_curves.png",
    },
    "interpretation": "Compare the trained frozen-backbone head with last-block fine-tuning on the same held-out test split. "
                      "Original ImageNet predictions are qualitative, not a target-class baseline. "
                      "Fine-tuning is not guaranteed to improve accuracy.",
}
'''

VERIFY_AND_SAVE_REPORT = r'''
# 저장한 체크포인트의 형식과 메타데이터도 확인합니다.
for filename in ["head_checkpoint.npz", "finetuned_checkpoint.npz"]:
    with np.load(OUTPUT_DIR / filename, allow_pickle=False) as archive:
        restored_metadata = json.loads(str(archive["__metadata__"]))
        assert restored_metadata["classes"] == classes
        assert restored_metadata["weights_sha256"] == weights_sha256
        assert archive["classifier.weight"].shape == (len(classes), config["hidden_size"])
        assert set(archive.files) - {"__metadata__"} == set(best_tuned_tail)
        assert all(np.all(np.isfinite(archive[name])) for name in best_tuned_tail)
for relative_path in report["artifacts"].values():
    path = OUTPUT_DIR / relative_path
    assert path.is_file() and path.stat().st_size > 0, path

report["artifact_sha256"] = {
    relative_path: file_sha256(OUTPUT_DIR / relative_path)
    for relative_path in report["artifacts"].values()
}
(OUTPUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
saved_report = json.loads((OUTPUT_DIR / "report.json").read_text(encoding="utf-8"))
assert saved_report["verification"]["frozen_prefix_unchanged"]
assert saved_report["verification"]["last_block_changed"]
print("VISION_NOTEBOOK_COMPLETE", json.dumps({
    "status": saved_report["status"], "actual_device": device.platform,
    "accelerator_verified": accelerator_verified, "report": str(OUTPUT_DIR / "report.json"),
}, ensure_ascii=False))
'''

COMPARE_GPU = r'''
# GPU에서 회수한 report.json을 TPU 업로드 전에 results/gpu에 둡니다.
gpu_report_path = ROOT / "results/gpu/report.json"
if device.platform != "tpu":
    print("CPU 검증에서는 GPU/TPU를 비교하지 않습니다.")
elif not gpu_report_path.exists():
    print("GPU 결과가 없습니다. TPU 학습은 완료했습니다. GPU 결과를 업로드한 뒤 이 셀을 다시 실행할 수 있습니다.")
else:
    gpu_report = json.loads(gpu_report_path.read_text(encoding="utf-8"))
    checks = {
        "GPU 실행 완료": gpu_report.get("status") == "completed" and gpu_report["device"]["platform"] == "gpu",
        "모델 일치": gpu_report["model"] == report["model"],
        "데이터 일치": gpu_report["dataset"]["manifest_sha256"] == report["dataset"]["manifest_sha256"],
        "클래스 순서 일치": gpu_report["dataset"]["classes"] == report["dataset"]["classes"],
        "분할 ID 일치": gpu_report["dataset"]["split_ids"] == report["dataset"]["split_ids"],
    }
    comparison_keys = ["seed", "head_epochs", "finetune_epochs", "batch_size", "learning_rate",
                       "head_learning_rate", "dtype", "optimizer", "augmentation", "checkpoint_selection"]
    checks["학습 조건 일치"] = all(gpu_report["config"].get(key) == report["config"].get(key)
                                for key in comparison_keys)
    print(checks)
    if not all(checks.values()):
        print("조건이 달라 성능을 비교하지 않습니다. 원본 보고서를 확인하세요.")
    else:
        names = ["Trained head", "Last-block fine-tuning"]
        gpu_values = [gpu_report["stages"][stage]["test"]["accuracy"] for stage in ["head", "finetune"]]
        tpu_values = [report["stages"][stage]["test"]["accuracy"] for stage in ["head", "finetune"]]
        x = np.arange(2)
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(x - 0.17, gpu_values, width=0.34, label="GPU")
        axis.bar(x + 0.17, tpu_values, width=0.34, label="TPU")
        axis.set(xticks=x, xticklabels=names, ylim=(0, 1), ylabel="Test accuracy",
                 title="Same split and training configuration")
        axis.legend()
        plt.show()
        for name, gpu_value, tpu_value in zip(names, gpu_values, tpu_values):
            print(f"{name}: GPU {gpu_value:.1%} / TPU {tpu_value:.1%}")
        print("소수점 단위 차이는 있을 수 있습니다. 시간은 컴파일·캐시 조건에도 영향을 받으므로 여기서 속도의 우열을 정하지 않습니다.")
'''


def _training_notebook(device):
    label = device.upper()
    number = "02" if device == "gpu" else "03"
    cells = [
        md(f"""# {number}. {label}에서 분류기 학습과 파인튜닝

이 노트북은 **Colab {label} 런타임에서 실행**합니다. Codespaces에서는 코드를 읽고 수정한 뒤
공식 Colab CLI로 이 파일을 전송합니다. Codespaces의 CPU 커널에서 실행하면 장치 검사에서 멈춥니다.
실행 절차와 의존성 설치 명령은 [Colab CLI 직접 실행 안내](../docs/direct-colab-cli.md)를 따르세요.

오늘 살펴볼 순서는 **라이브러리 가져오기 → 데이터·모델 읽기 → 특징 계산 → head 학습 → 마지막 블록 파인튜닝 → 평가·저장**입니다.
각 단계의 Python 코드를 셀에 그대로 넣었습니다. `vision_lab` 패키지나 통합 실행 함수를 부르지 않습니다.

| 라이브러리 | 이 노트북에서 하는 일 |
|---|---|
| JAX / `jax.numpy` | GPU·TPU 배열 연산, `jit` 컴파일, `value_and_grad` 자동미분 |
| Optax | 교차 엔트로피, Adam 업데이트 |
| NumPy | NPZ 데이터 읽기, 배치 인덱스, 지표 계산, 체크포인트 저장 |
| Pillow (`PIL.Image`) | RGB 변환과 이미지 크기 조정 |
| Safetensors | 사전학습 가중치 읽기 |
| Matplotlib | 학습 곡선과 혼동 행렬 그리기 |

가상환경은 라이브러리 버전을 분리하는 공간입니다. 라이브러리의 역할은 아래 `import`와 사용 코드를 따라 확인할 수 있습니다.
"""),
        *boot_cells(),
        *device_cells(device),
        *model_definition_cells(),
        *weight_loading_cells(),
        *preprocessing_cells(),
        *prepared_data_cells(),
        md("""## 1. 학습 조건과 결과 폴더

같은 클래스 순서와 train/validation/test 분할로 진행합니다. 기본값은 head 3 epoch, 파인튜닝 3 epoch,
배치 16, seed 42입니다. 완성된 결과가 있으면 덮어쓰지 않고 멈춥니다.
직접 준비한 데이터는 앞의 `DATA_PATH` 셀을 해당 폴더로 바꿔 읽으세요. 클래스 수와 순서는 manifest를 따릅니다.
"""),
        code(SETTINGS),
        md("""## 2. 배치와 평가 함수

마지막 배치에 이미지가 16장보다 적으면 끝 이미지를 복제해 모양을 맞춥니다.
복제한 행은 `mask=0`으로 표시해 학습 손실에서 제외하고, 평가할 때도 잘라냅니다.
"""),
        code(BATCH_ITERATOR),
        md("배치별 추론 결과에서 패딩 행을 잘라 실제 이미지의 출력만 모읍니다."),
        code(PREDICT_BATCHES),
        md("정확도·macro F1·혼동 행렬을 계산합니다. 클래스 순서는 앞에서 읽은 manifest를 따릅니다."),
        code(CLASSIFICATION_METRICS),
        md("""## 3. 학습하지 않는 앞부분의 특징 계산

DeiT의 첫 11개 블록은 고정합니다. 같은 이미지를 넣으면 같은 토큰이 나오므로 한 번만 계산해 둡니다.
head 학습에는 마지막 블록까지 거친 192차원 특징을, 파인튜닝에는 마지막 블록 직전 토큰을 사용합니다.
이 실습에는 데이터 증강이 없습니다.
"""),
        code(CACHE),
        md("""## 4. 원래 모델의 추론 결과 확인

사전학습 모델은 ImageNet의 1000개 클래스를 구분합니다. 아래 출력은 모델이 이미 알고 있는 것을 살펴보는 예시입니다.
이 결과를 물체 5종 분류의 정확도 기준선으로 쓰지는 않습니다. 학습 전후 비교 기준은 다음 단계에서 학습하는 **5종 head**입니다.
"""),
        code(ORIGINAL_PREDICTIONS),
        md("""## 5. 새 head와 한 번의 학습 단계

기본 데이터에서는 5개 클래스의 점수(logits)를 내는 head를 만듭니다. 직접 준비한 데이터는 해당 클래스 수에 맞춥니다.
`loss_fn`이 교차 엔트로피를 계산하고,
`jax.value_and_grad`가 손실과 기울기를 구합니다. Optax의 Adam이 가중치 업데이트를 계산합니다.
"""),
        code(HEAD_INITIALIZATION),
        md("""## 6. head 학습 반복문

epoch마다 train 순서를 섞고 head만 업데이트합니다. **validation 손실이 가장 낮은 epoch**를 선택합니다.
이 반복문에는 test 평가가 없습니다.
"""),
        code(HEAD_LOOP),
        md("""## 7. 마지막 블록까지 학습하도록 범위 변경

이제 학습된 head와 함께 마지막 Transformer 블록, 최종 LayerNorm도 업데이트합니다.
기본 5개 클래스의 학습 대상은 총 **446,213개 가중치**입니다. 직접 준비한 데이터는 클래스 수에 따라 head 크기가 달라집니다.
학습률은 `1e-4`이며 앞의 11개 블록은 계속 고정합니다.
"""),
        code(FINETUNE_INITIALIZATION),
        md("""## 8. 파인튜닝 반복문

이번에는 캐시한 토큰을 마지막 블록에 넣습니다. head와 마찬가지로 validation 손실만 보고 epoch를 선택합니다.
검증 결과가 좋지 않더라도 test 결과를 보고 학습 횟수를 다시 고르지 않습니다.
"""),
        code(FINETUNE_LOOP),
        md("""## 9. 같은 test 분할로 평가

선택을 마친 두 모델을 비교합니다. 파인튜닝으로 정확도가 반드시 높아지는 것은 아닙니다.
정확도와 macro F1을 확인하고, 고정 가중치는 그대로인지, 마지막 블록 가중치는 바뀌었는지도 검사합니다.
"""),
        code(TEST_AND_VERIFY),
        md("""## 10. 어느 물체를 혼동하는지 확인

혼동 행렬의 행은 정답, 열은 예측입니다. 같은 test 이미지에서 head 학습 후와 파인튜닝 후를 비교합니다.
아래 학습 곡선은 모델을 선택할 때 사용한 train/validation 손실입니다.
"""),
        code(PLOT_RESULTS),
        md("""## 11. 결과 저장

`np.savez_compressed`로 마지막 블록·LayerNorm·head와 클래스 순서를 저장합니다.
`csv`는 학습 이력과 개별 예측을, `json`은 실험 조건과 검사 결과를 기록합니다.
사전학습 원본 파일은 그대로 두고 학습한 부분만 저장합니다.
"""),
        code(SAVE_CHECKPOINTS),
        md("학습 이력과 개별 이미지의 예측을 CSV로 저장합니다."),
        code(SAVE_CSV),
        md("보고서에는 모델·데이터·학습 조건과 실제 장치를 함께 기록합니다."),
        code(REPORT),
        md("체크포인트와 결과 파일을 다시 확인한 뒤 완료 보고서를 저장합니다."),
        code(VERIFY_AND_SAVE_REPORT),
        md("""## 실행 완료 확인과 결과 회수

마지막 출력에 `VISION_NOTEBOOK_COMPLETE`가 있고 `report.json`의 `status`가 `completed`인지 확인하세요.
`device.platform`도 요청한 GPU 또는 TPU여야 합니다. 명시적으로 CPU 검증을 했다면 `cpu_validation`,
일부 데이터만 검증했다면 `cpu_smoke_only`로 남습니다.
Colab CLI의 종료 코드만으로 성공을 판단하지 마세요.

결과를 내려받고 세션을 종료하는 명령은 [Colab CLI 직접 실행 안내](../docs/direct-colab-cli.md)에 있습니다.
질문: 정확도가 달라졌다면 혼동 행렬의 어떤 칸에서 변화가 나타났나요?
"""),
    ]
    if device == "tpu":
        cells.extend([
            md("""## 12. 같은 조건으로 실행한 GPU 결과와 비교

이 노트북도 TPU에서 head 학습과 파인튜닝 전체를 실행했습니다.
GPU의 `report.json`을 `results/gpu/report.json`에 업로드했다면 아래 셀에서 결과를 비교할 수 있습니다.
모델·데이터·클래스 순서·분할 ID·학습 조건이 같을 때만 그래프를 그립니다.
GPU 결과가 없어도 앞에서 저장한 TPU 학습 결과는 사용할 수 있습니다.
"""),
            code(COMPARE_GPU),
        ])
    return make_notebook(cells)


def make_notebooks():
    return {
        "02_gpu_finetuning.ipynb": _training_notebook("gpu"),
        "03_tpu_and_compare.ipynb": _training_notebook("tpu"),
    }
