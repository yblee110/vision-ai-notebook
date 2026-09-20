"""Build-time cell fragments; none of this module is imported by student notebooks."""
from pathlib import Path
import ast
import inspect
import textwrap
import nbformat as nbf


def md(text):
    # cleandoc handles an unindented first line plus indented source continuation
    # lines while retaining additional indentation inside lists and code fences.
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def make_notebook(cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Vision AI (Codespaces CPU)", "language": "python", "name": "codespaces-vision-ai"},
        "language_info": {"name": "python", "version": "3.12"},
        "vision_ai": {"public_library_imports": True, "course_scope": "pretrained inference and last-block fine-tuning"},
    }
    nbf.validate(notebook)
    return notebook


def boot_cells(device=None):
    cells = [md('''## 라이브러리를 직접 불러오기
    가상환경은 라이브러리 버전을 구분하는 공간입니다. 아래 `import`가 이 노트북에서 실제로 사용하는 라이브러리입니다.

    | 가져오는 이름 | 설치할 패키지 | 하는 일 |
    |---|---|---|
    | `numpy` | `numpy` | 이미지 배열, 라벨, `.npz` 파일 |
    | `PIL.Image` | `Pillow` | 이미지 읽기와 크기 조절 |
    | `matplotlib.pyplot` | `matplotlib` | 이미지와 그래프 표시 |
    | `IPython` | `ipykernel`과 함께 설치 | 노트북 안에 그래프 표시 |

    `pathlib`, `os`, `sys`, `json`, `hashlib`, `importlib.metadata`는 Python 표준 라이브러리이므로 따로 설치하지 않습니다. 필요한 추가 라이브러리는 사용하는 셀에서 직접 불러옵니다.'''), code('''
    from pathlib import Path
    import os
    import sys
    import json
    import hashlib
    from importlib.metadata import version

    candidates = [Path.cwd(), *Path.cwd().parents, Path("/content/vision-ai")]
    ROOT = next((path for path in candidates if (path / ".vision-lab-root").is_file()), None)
    if ROOT is None:
        raise RuntimeError(".vision-lab-root가 있는 수업 폴더에서 열거나 Colab에 실습 파일을 먼저 업로드하세요.")
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    print("Project:", ROOT)
    print("Python:", sys.executable)
    '''), code('''
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    ipython = get_ipython()
    if ipython is not None:
        ipython.run_line_magic("matplotlib", "inline")
    plt.rcParams.update({"figure.figsize": (9, 4), "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    for package in ("numpy", "Pillow", "matplotlib", "ipykernel"):
        print(f"{package}: {version(package)}")
    ''')]
    if device is not None:
        cells.extend(device_cells(device))
    return cells


def device_cells(device="cpu"):
    return [md('''### JAX와 실행 장치 확인
    `jax`는 자동 미분과 컴파일을, `jax.numpy`는 배열 연산을 제공합니다. 설치 패키지는 CPU에서 `jax`, GPU에서 `jax[cuda12]`, TPU에서 `jax[tpu]`입니다. 같은 `import`를 사용하지만 실행 환경에 맞는 패키지가 필요합니다.

    GPU·TPU를 요청했는데 장치가 없으면 오류로 중단합니다. CPU로 몰래 바꾸지 않습니다. 강사용 CPU 검증 때만 `VISION_DEVICE=cpu`를 명시할 수 있으며 출력에도 CPU로 기록됩니다.'''), code(f'''
    import jax
    import jax.numpy as jnp

    REQUESTED_DEVICE = os.environ.get("VISION_DEVICE", {device!r})
    if REQUESTED_DEVICE not in {{"cpu", "gpu", "tpu"}}:
        raise ValueError("VISION_DEVICE는 cpu, gpu, tpu 중 하나여야 합니다.")
    try:
        devices = jax.devices(REQUESTED_DEVICE)
    except RuntimeError as error:
        raise RuntimeError(f"{{REQUESTED_DEVICE.upper()}}를 찾지 못했습니다. 해당 Colab 런타임에서 실행하세요.") from error
    if not devices or devices[0].platform != REQUESTED_DEVICE:
        raise RuntimeError("요청한 장치가 없습니다. Codespaces 자체에는 Colab GPU·TPU가 연결되지 않습니다.")
    device = devices[0]
    device_info = {{"requested": REQUESTED_DEVICE, "platform": device.platform,
                   "device_kind": device.device_kind, "available_count": len(devices),
                   "used_count": 1, "jax_version": jax.__version__}}
    print("Device:", device_info)
    ''')]


def _model_functions(names):
    """Copy reviewed arithmetic into visible notebook cells at build time only."""
    path = Path(__file__).resolve().parent / "model_reference.py"
    source = path.read_text(encoding="utf-8")
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return "\n\n".join(functions[name] for name in names)


def model_definition_cells():
    # The student output contains every function body, not this authoring helper.
    groups = [
        ("### 모델 계산 1 · 행렬 곱과 정규화", "`jnp.matmul`이 가중치와 입력을 곱합니다. LayerNorm은 마지막 축의 평균과 분산으로 값을 정규화합니다.", ["dense", "layer_norm"]),
        ("### 모델 계산 2 · Attention과 Transformer 블록", "Query·Key의 내적으로 토큰 사이의 점수를 계산하고 `jax.nn.softmax`로 가중치를 만듭니다. 잔차 연결과 GELU까지 아래 함수에서 확인할 수 있습니다.", ["transformer_block"]),
        ("### 모델 계산 3 · 이미지를 패치 토큰으로 바꾸기", "224×224 이미지를 16×16 패치로 바꾼 뒤 첫 11개 블록을 통과시킵니다. 이 부분은 추가 학습에서 고정하므로 출력을 캐시할 수 있습니다.", ["prefix_tokens"]),
        ("### 모델 계산 4 · 마지막 블록과 분류기", "마지막 블록의 CLS 토큰으로 분류합니다. `original_logits`는 앞서 정의한 함수들을 순서대로 호출해 원본 1,000개 클래스 점수를 계산합니다.", ["tail_features", "tail_logits", "original_logits"]),
        ("### 학습할 가중치와 파일 지문", "아래 함수는 마지막 블록·LayerNorm·분류기의 이름을 고르고 가중치가 바뀌었는지 확인합니다. `hashlib`의 SHA256은 파일과 배열의 지문을 만드는 표준 라이브러리 함수입니다.", ["is_tail_parameter", "extract_tail", "initialize_head", "parameter_digest", "file_sha256"]),
        ("### 체크포인트 저장과 복원", "`np.savez_compressed`와 `np.load`로 갱신한 마지막 블록·분류기를 저장하고 읽습니다. 복원할 때 원본 모델 revision과 각 배열 크기를 검사합니다.", ["save_checkpoint", "load_checkpoint"]),
    ]
    cells = []
    for heading, explanation, names in groups:
        cells.extend([md(heading + "\n" + explanation), code(_model_functions(names))])
    return cells


def weight_loading_cells():
    return [md('''### safetensors로 사전학습 가중치 읽기
    `safetensors.numpy.load_file`은 저장된 텐서를 NumPy 배열로 읽습니다. `jax.device_put`으로 이 배열들을 선택한 장치로 옮깁니다. 새 무작위 모델이 아니라 고정된 DeiT 가중치를 불러옵니다.'''), code('''
    from safetensors.numpy import load_file

    MODEL_ID = "facebook/deit-tiny-patch16-224"
    MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
    MODEL_SHA256 = "056550dbe6c439dddb35e1800a48aaef86cfdcf8e566ba73f0d1ce41bc7fa1b3"
    ASSETS = ROOT / "assets/pretrained"
    config = json.loads((ASSETS / "config.json").read_text(encoding="utf-8"))
    if (config["hidden_act"] != "gelu" or config["num_hidden_layers"] != 12
            or config["hidden_size"] != 192 or config["image_size"] != 224
            or config["patch_size"] != 16 or config["num_attention_heads"] != 3):
        raise ValueError("고정된 DeiT tiny 모델 구조와 다릅니다.")
    weights_path = ASSETS / "model.safetensors"
    if file_sha256(weights_path) != MODEL_SHA256:
        raise ValueError("사전학습 파일 SHA256이 다릅니다.")
    numpy_weights = load_file(str(weights_path))
    if numpy_weights["classifier.weight"].shape != (1000, 192):
        raise ValueError("원본 1,000개 클래스 분류기가 필요합니다.")
    params = {name: jax.device_put(np.asarray(value, np.float32), device)
              for name, value in numpy_weights.items()}
    del numpy_weights
    print("Model:", MODEL_ID, "revision:", MODEL_REVISION)
    print("safetensors:", version("safetensors"), "tensors:", len(params))
    for name in ["vit.embeddings.patch_embeddings.projection.weight",
                 "vit.encoder.layer.11.attention.attention.query.weight", "classifier.weight"]:
        print(name, params[name].shape, params[name].dtype)
    ''')]


def preprocessing_cells():
    return [md('''### Pillow와 NumPy로 입력 전처리
    RGB 이미지를 224×224로 바꾸고 픽셀 값을 `(pixel / 255 - 0.5) / 0.5`로 변환합니다. 모델의 `preprocessor_config.json`과 같은 설정입니다. `preprocess`는 외부 패키지가 아닌 아래 셀에서 직접 정의하는 함수입니다.'''), code('''
    def preprocess(images):
        if not len(images):
            return np.empty((0, 224, 224, 3), dtype=np.float32)
        resized = []
        for pixels in images:
            image = Image.fromarray(np.asarray(pixels, dtype=np.uint8)).convert("RGB")
            image = image.resize((224, 224), Image.Resampling.BILINEAR)
            resized.append(np.asarray(image, dtype=np.float32))
        return (np.stack(resized) / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    ''')]


def data_validation_cells():
    return [md('''### 저장된 데이터 읽기와 무결성 확인
    `np.load(..., allow_pickle=False)`로 이미지·라벨·ID를 읽습니다. SHA256과 분할별 ID를 검사해 다른 데이터가 섞이거나 손상된 경우 중단합니다. 이 함수의 본문도 아래에 모두 표시합니다.'''), code('''
    def digest(path):
        hasher = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        return hasher.hexdigest()
    '''), code('def load_prepared(path):\n    path = Path(path)\n    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))\n    classes = manifest["classes"]\n    if len(classes) < 2 or len(set(classes)) != len(classes):\n        raise ValueError("클래스 목록이 잘못됐습니다.")\n    splits, all_ids = {}, set()\n    for name in ("train", "validation", "test"):\n        record = manifest["splits"][name]\n        if record["file"] != f"{name}.npz":\n            raise ValueError("데이터 파일 경로가 예상 형식과 다릅니다.")\n        file = path / record["file"]\n        if digest(file) != record["sha256"]:\n            raise ValueError(f"{name} 데이터 체크섬 불일치")\n        with np.load(file, allow_pickle=False) as a:\n            images, labels, ids = a["images"], a["labels"], a["ids"].tolist()\n        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:\n            raise ValueError("images는 uint8 NHWC RGB여야 합니다.")\n        if len(images) != len(labels) or len(ids) != len(labels) or len(labels) != record["count"]:\n            raise ValueError("이미지·라벨·식별자 개수가 다릅니다.")\n        if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or len(labels) == 0 or labels.min() < 0 or labels.max() >= len(classes):\n            raise ValueError("라벨 범위가 잘못됐습니다.")\n        if len(set(ids)) != len(ids) or all_ids.intersection(ids):\n            raise ValueError("분할 간 이미지 ID 중복")\n        all_ids.update(ids)\n        splits[name] = {"images": images, "labels": labels, "ids": ids}\n    identity = {k: manifest[k] for k in ("classes", "splits", "seed", "preprocess")}\n    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != manifest["dataset_sha256"]:\n        raise ValueError("데이터 manifest 지문 불일치")\n    return splits, manifest')]


def prepared_data_cells(data_path="data/prepared"):
    return [*data_validation_cells(), code(f'''
    DATA_PATH = ROOT / {data_path!r}
    if not (DATA_PATH / "manifest.json").is_file():
        raise RuntimeError("00_setup_and_data.ipynb에서 데이터를 먼저 준비하세요.")
    splits, manifest = load_prepared(DATA_PATH)
    classes = manifest["classes"]
    print("Classes:", classes)
    print("Split counts:", {{name: len(split["labels"]) for name, split in splits.items()}})
    print("Dataset fingerprint:", manifest["dataset_sha256"])
    ''')]


def data_saving_cells():
    return [md('''### 분할을 NPZ와 manifest로 저장하기
    이미지 배열은 압축한 `.npz`, 출처·클래스·파일 지문은 `manifest.json`으로 저장합니다. 같은 픽셀이 두 분할에 있으면 데이터 누출을 막기 위해 중단합니다.'''), code('''
    PREPROCESS = {"size": 224, "mean": [0.5] * 3, "std": [0.5] * 3,
                  "resize": "bilinear", "layout": "NHWC"}
    '''), code('def save_prepared(output, splits, classes, source, seed):\n    output = Path(output)\n    output.mkdir(parents=True, exist_ok=True)\n    # Exact duplicate pixels across splits are excluded before this point.\n    seen = {}\n    for name, split in splits.items():\n        for image in split["images"]:\n            key = hashlib.sha256(image.tobytes()).hexdigest()\n            if key in seen and seen[key] != name:\n                raise ValueError(f"분할 간 중복 이미지: {seen[key]} / {name}")\n            seen[key] = name\n    records = {}\n    for name, split in splits.items():\n        path = output / f"{name}.npz"\n        with path.with_suffix(".tmp").open("wb") as stream:\n            np.savez_compressed(stream, images=split["images"], labels=split["labels"], ids=np.asarray(split["ids"], dtype=str))\n        path.with_suffix(".tmp").replace(path)\n        records[name] = {"file": path.name, "sha256": digest(path), "count": len(split["labels"]),\n                         "per_class": {c: int(np.sum(split["labels"] == i)) for i, c in enumerate(classes)}}\n    identity = {"classes": classes, "splits": records, "seed": seed, "preprocess": PREPROCESS}\n    manifest = {"schema_version": 1, "dataset_name": source["name"], **identity, "source": source,\n                "dataset_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}\n    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")\n    return manifest')]
