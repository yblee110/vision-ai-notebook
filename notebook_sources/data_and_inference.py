"""Author the data, pretrained inference and custom image notebooks."""
from .common import (md, code, make_notebook, boot_cells, device_cells,
                     model_definition_cells, weight_loading_cells, preprocessing_cells,
                     prepared_data_cells, data_validation_cells, data_saving_cells)


def setup_notebook():
    cells = [md('''# 00 · 라이브러리로 직접 준비하는 물체 이미지 데이터
    **목표:** 어떤 라이브러리가 무엇을 하는지 확인하고 이미지 분류에 쓸 데이터를 직접 준비합니다.

    Codespaces는 편집·데이터 준비·CPU 추론을 맡고 Colab은 GPU·TPU 학습을 맡습니다. 이 노트북은 원격 연산을 시작하지 않습니다. CIFAR-100의 32×32 공개 이미지로 흐름을 익힙니다. 실제 작업대 사진의 성능은 별도로 확인해야 합니다.

    커널은 **Vision AI (Codespaces CPU)**를 선택하세요. 처음에는 `scripts/setup.sh`로 개별 라이브러리를 설치합니다. 설치 과정에서는 데이터를 준비하지 않으므로 아래 셀에서 다운로드·선별·저장 과정을 직접 실행합니다.'''), *boot_cells(),
    md('''## 1. Parquet를 읽는 pyarrow 불러오기
    `pyarrow.parquet`는 표 형태의 Parquet 파일을 읽습니다. `urllib.request`는 파일을 내려받고 `io.BytesIO`는 압축된 이미지 바이트를 Pillow에 전달합니다. `io`, `shutil`, `urllib.request`는 표준 라이브러리입니다.'''),
    code('''
    import io
    import shutil
    import urllib.request
    import pyarrow.parquet as pq

    print("pyarrow:", version("pyarrow"))
    CLASSES = ["bottle", "bowl", "can", "cup", "plate"]
    FINE_LABEL_IDS = [9, 10, 16, 28, 61]
    DATASET_ID = "uoft-cs/cifar100"
    DATASET_REVISION = "aadb3af77e9048adbea6b47c21a81e47dd092ae5"
    SOURCE_FILES = {
        "train": "694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23",
        "test": "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68",
    }
    SEED = 42
    TRAIN_PER_CLASS, VALIDATION_PER_CLASS, TEST_PER_CLASS = 100, 20, 40
    CACHE = ROOT / ".cache/data"
    DATA_PATH = ROOT / "data/prepared"
    print("Classes:", dict(zip(FINE_LABEL_IDS, CLASSES)))
    '''), *data_validation_cells(),
    md('''## 2. 기존 데이터가 있으면 먼저 검증하기
    같은 조건으로 만든 파일은 재사용합니다. 다른 조건의 실험 폴더는 덮어쓰지 않습니다. 데이터 수나 seed를 바꾸려면 `DATA_PATH`를 새로운 폴더로 지정하세요.'''),
    code('''
    selection = {"train_per_class": TRAIN_PER_CLASS, "val_per_class": VALIDATION_PER_CLASS,
                 "test_per_class": TEST_PER_CLASS}
    REUSE_PREPARED = (DATA_PATH / "manifest.json").is_file()
    if REUSE_PREPARED:
        splits, manifest = load_prepared(DATA_PATH)
        source = manifest.get("source", {})
        if (manifest["dataset_name"] != "cifar100_food_containers" or manifest["seed"] != SEED
                or manifest["classes"] != CLASSES or source.get("selection") != selection
                or source.get("revision") != DATASET_REVISION or source.get("source_sha256") != SOURCE_FILES):
            raise ValueError("다른 조건의 데이터가 있습니다. 새 DATA_PATH를 지정하세요.")
        print("기존 데이터의 SHA256과 실험 조건을 확인했습니다.")
    elif DATA_PATH.exists() and any(DATA_PATH.iterdir()):
        raise FileExistsError("완성되지 않은 데이터 폴더가 있습니다. 내용을 확인하거나 새 DATA_PATH를 지정하세요.")
    '''),
    md('''## 3. 원본 다운로드와 SHA256 확인
    처음 실행할 때 원본 Parquet 두 개를 약 142 MB 내려받습니다. 이미 있는 파일도 SHA256을 확인합니다. 주소에 고정 revision을 넣어 다른 시점의 데이터를 받지 않도록 합니다.'''),
    code('''
    CACHE.mkdir(parents=True, exist_ok=True)
    source_paths = {}
    if not REUSE_PREPARED:
        for split_name in ("train", "test"):
            filename = f"{split_name}-00000-of-00001.parquet"
            target = CACHE / filename
            if not target.is_file():
                partial = target.with_suffix(".part")
                url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/cifar100/{filename}"
                print("Download:", filename)
                try:
                    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                        shutil.copyfileobj(response, stream)
                    if digest(partial) != SOURCE_FILES[split_name]:
                        raise ValueError("다운로드한 파일의 SHA256이 다릅니다.")
                    partial.replace(target)
                finally:
                    partial.unlink(missing_ok=True)
            if digest(target) != SOURCE_FILES[split_name]:
                raise ValueError(f"캐시 SHA256 불일치: {target}")
            source_paths[split_name] = target
            print("Verified:", filename)
    else:
        print("검증한 준비 데이터를 재사용하므로 원본 다운로드를 생략합니다.")
    '''),
    md('''## 4. pyarrow로 표 읽기
    필요한 `img`, `fine_label` 두 열만 읽습니다. 원본 학습 분할에는 50,000장, 테스트 분할에는 10,000장이 있어야 합니다.'''),
    code('''
    if not REUSE_PREPARED:
        train_table = pq.read_table(source_paths["train"], columns=["img", "fine_label"])
        test_table = pq.read_table(source_paths["test"], columns=["img", "fine_label"])
        train_labels = train_table["fine_label"].to_numpy()
        test_labels = test_table["fine_label"].to_numpy()
        assert len(train_labels) == 50000 and len(test_labels) == 10000
        print(train_table.schema)
        print("Original split sizes:", len(train_labels), len(test_labels))
    '''),
    md('''## 5. NumPy로 클래스별 이미지를 고르기
    원본 학습 분할에서 클래스마다 학습 100장·검증 20장을 서로 겹치지 않게 선택합니다. 테스트는 원본 테스트 분할에서 클래스마다 40장을 고릅니다. 검증 이미지는 epoch 선택에, 테스트 이미지는 최종 평가에 사용합니다.'''),
    code('''
    if not REUSE_PREPARED:
        if min(TRAIN_PER_CLASS, VALIDATION_PER_CLASS, TEST_PER_CLASS) < 1:
            raise ValueError("클래스별 선택 수는 양수여야 합니다.")
        rng = np.random.default_rng(SEED)
        train_indices, validation_indices = [], []
        for label in FINE_LABEL_IDS:
            candidates = np.flatnonzero(train_labels == label)
            if TRAIN_PER_CLASS + VALIDATION_PER_CLASS > len(candidates):
                raise ValueError("학습·검증 요청 수가 원본 클래스 수를 넘습니다.")
            shuffled = rng.permutation(candidates)
            train_indices.extend(shuffled[:TRAIN_PER_CLASS])
            validation_indices.extend(shuffled[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VALIDATION_PER_CLASS])
        rng = np.random.default_rng(SEED + 1)
        if TEST_PER_CLASS > 100:
            raise ValueError("원본 테스트에는 클래스마다 100장이 있습니다.")
        test_indices = np.concatenate([rng.permutation(np.flatnonzero(test_labels == label))[:TEST_PER_CLASS]
                                       for label in FINE_LABEL_IDS])
        assert not set(train_indices).intersection(validation_indices)
        print("Selected:", len(train_indices), len(validation_indices), len(test_indices))
    '''),
    md('''## 6. Pillow로 이미지 바이트를 RGB 배열로 바꾸기
    이 단계에서는 원본 32×32 픽셀을 보존합니다. 모델에 넣을 224×224 전처리는 다음 노트북에서 직접 수행합니다.'''),
    code('''
    if not REUSE_PREPARED:
        label_map = {old: new for new, old in enumerate(FINE_LABEL_IDS)}
        splits = {}
        for name, table, labels, indices, origin in [
            ("train", train_table, train_labels, train_indices, "train"),
            ("validation", train_table, train_labels, validation_indices, "train"),
            ("test", test_table, test_labels, test_indices, "test"),
        ]:
            pixels = []
            for row in table.take([int(index) for index in indices])["img"].to_pylist():
                with Image.open(io.BytesIO(row["bytes"])) as image:
                    pixels.append(np.asarray(image.convert("RGB")))
            splits[name] = {"images": np.stack(pixels),
                            "labels": np.asarray([label_map[int(labels[index])] for index in indices], dtype=np.int64),
                            "ids": [f"cifar100/{origin}/{index:05d}" for index in indices]}
        del train_table, test_table
        print("Train pixels:", splits["train"]["images"].shape, splits["train"]["images"].dtype)
    '''), *data_saving_cells(),
    code('''
    if not REUSE_PREPARED:
        source = {"name": "cifar100_food_containers", "dataset_id": DATASET_ID,
                  "revision": DATASET_REVISION, "source_sha256": SOURCE_FILES,
                  "original_url": "https://www.cs.toronto.edu/~kriz/cifar.html",
                  "original_resolution": [32, 32], "selection": selection}
        manifest = save_prepared(DATA_PATH, splits, CLASSES, source, SEED)
    splits, manifest = load_prepared(DATA_PATH)
    classes = manifest["classes"]
    for name, split in splits.items():
        print(name, dict(zip(classes, np.bincount(split["labels"], minlength=len(classes)).tolist())))
    print("Dataset fingerprint:", manifest["dataset_sha256"])
    '''), md('''## 7. Matplotlib으로 학습 이미지 확인
    각 클래스에서 3장을 표시합니다. 작은 이미지를 확대해도 새로운 세부 정보가 생기지는 않습니다. 물체와 배경을 함께 살펴보세요.'''),
    code('''
    fig, axes = plt.subplots(len(classes), 3, figsize=(8, 2 * len(classes)), squeeze=False)
    for label, name in enumerate(classes):
        indices = np.flatnonzero(splits["train"]["labels"] == label)[:3]
        for column, index in enumerate(indices):
            axes[label, column].imshow(splits["train"]["images"][index], interpolation="nearest")
            axes[label, column].set_title(f"{name} · sample {column + 1}")
            axes[label, column].axis("off")
    fig.suptitle("Training images · CIFAR-100 food containers · native 32 × 32", y=1.01)
    fig.tight_layout()
    plt.show()
    '''), md('''## 확인 질문과 다음 단계
    - 컵과 그릇을 구분하기 어려운 사진에는 어떤 특징이 있나요?
    - 같은 촬영 장면을 학습과 테스트에 복사하면 평가가 어떻게 달라질까요?
    - 어떤 라이브러리가 파일 다운로드·이미지 변환·배열 저장을 각각 맡았나요?

    다음은 `01_pretrained_inference.ipynb`입니다. 데이터 출처는 [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)와 [고정 데이터 저장소](https://huggingface.co/datasets/uoft-cs/cifar100/tree/aadb3af77e9048adbea6b47c21a81e47dd092ae5)입니다.''')]
    return make_notebook(cells)


def inference_notebook():
    cells = [md('''# 01 · 사전학습 모델을 직접 읽고 한 장 추론하기
    **목표:** `safetensors`로 DeiT 가중치를 불러오고 Pillow·NumPy로 이미지를 준비한 뒤 JAX로 원본 1,000개 클래스의 Top-5를 계산합니다.

    Codespaces CPU에서 실제 추론합니다. GPU·TPU 실습 완료를 뜻하지 않습니다. 이 노트북은 모델 계산도 셀 안에 모두 보여 줍니다. 함수 정의를 읽는 데 오래 걸리면 제목과 라이브러리 호출부터 확인하고 각 셀을 순서대로 실행하세요.'''), *boot_cells(), *prepared_data_cells(),
    md('''## 1. 예제 이미지 선택
    기본 예시는 테스트 분할의 `cup`입니다. 여기서는 원본 모델의 라벨을 관찰할 뿐 최적 epoch를 고르거나 새 분류기의 정확도를 계산하지 않습니다.'''),
    code('''
    SAMPLE_CLASS = "cup"
    label = classes.index(SAMPLE_CLASS)
    sample_index = int(np.flatnonzero(splits["test"]["labels"] == label)[0])
    raw = splits["test"]["images"][sample_index]
    print("Image ID:", splits["test"]["ids"][sample_index])
    plt.figure(figsize=(3, 3))
    plt.imshow(raw, interpolation="nearest")
    plt.title(f"Workbench label: {SAMPLE_CLASS}")
    plt.axis("off")
    plt.show()
    '''), *device_cells("cpu"), *model_definition_cells(), *weight_loading_cells(), *preprocessing_cells(),
    md('''## 2. 전처리한 배열을 CPU에 올리기
    출력 크기는 `(1, 224, 224, 3)`, 자료형은 `float32`입니다. `jax.device_put`은 계산에 사용할 장치로 배열을 보냅니다.'''),
    code('''
    pixels = jax.device_put(preprocess(raw[None, ...]), device)
    print("Input:", pixels.shape, pixels.dtype)
    print("Original classifier:", params["classifier.weight"].shape)
    '''),
    md('''## 3. jax.jit으로 컴파일하고 추론하기
    `jax.jit`은 위에서 정의한 모델 계산을 컴파일합니다. 첫 호출에는 컴파일 시간이 포함됩니다. softmax 값은 원본 라벨 체계 안의 상대적인 점수이며 실제 작업대 환경에서 보정한 신뢰도가 아닙니다.'''),
    code('''
    forward = jax.jit(lambda weights, images: original_logits(weights, images, config))
    logits = forward(params, pixels)
    probabilities = np.asarray(jax.device_get(jax.nn.softmax(logits[0])))
    top = np.argsort(probabilities)[-5:][::-1]
    top_labels = [config["id2label"][str(int(index))] for index in top]
    assert logits.shape == (1, 1000) and np.isfinite(probabilities).all()
    for name, score in zip(top_labels, probabilities[top]):
        print(f"{float(score):.3f}  {name}")
    '''), md('''## 4. Top-5 표시하기'''),
    code('''
    fig, ax = plt.subplots(figsize=(9, 3.8))
    positions = np.arange(len(top))
    ax.barh(positions, probabilities[top], color="#4285F4")
    ax.set_yticks(positions, top_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Softmax probability")
    ax.set_title("Original pretrained model · top 5 ImageNet labels · one test image")
    for y, score in zip(positions, probabilities[top]):
        ax.text(min(float(score) + 0.01, 0.94), y, f"{float(score):.3f}", va="center")
    fig.tight_layout()
    plt.show()
    '''), md('''## 확인 질문과 다음 단계
    - `load_file`, `Image.resize`, `jax.jit`, `jax.nn.softmax`는 각각 어느 단계에서 사용했나요?
    - 우리 라벨 `cup`과 원본 모델의 `coffee mug`는 어떻게 비교해야 할까요?
    - 새 5개 클래스 분류기를 만들면 그 분류기는 이미 학습된 상태일까요?

    다음에는 새 분류기를 먼저 학습해 기준 모델을 만들고 마지막 사전학습 블록도 업데이트합니다. 원본 1,000개 클래스의 결과를 새 5개 클래스 정확도와 같은 지표로 비교하지 않습니다.

    모델 출처: [DeiT](https://huggingface.co/facebook/deit-tiny-patch16-224/tree/b3428f18dcc7b543470d07f14b4a4157815d1880).''')]
    return make_notebook(cells)


def custom_notebook():
    cells = [md('''# 04 · 직접 촬영한 이미지로 바꾸기
    **목표:** Pillow로 사진을 읽어 NumPy 데이터로 만들고 회수한 파인튜닝 체크포인트로 추론합니다.

    사진과 체크포인트를 아직 준비하지 않았다면 해당 단계는 안내만 표시합니다. 원격 작업은 이 노트북에서 자동 시작하지 않습니다.'''), *boot_cells(),
    md('''## 1. 촬영과 분할
    `custom_images/train/cup`, `custom_images/validation/cup`, `custom_images/test/cup`처럼 분할마다 같은 클래스 폴더를 만드세요. 분류할 클래스는 2개 이상이어야 합니다.

    같은 영상의 비슷한 프레임을 섞지 말고 촬영 날짜·배경·조명 단위로 분리합니다. 아래 코드는 완전히 동일한 이미지를 찾지만 비슷한 연속 프레임까지 자동으로 걸러내지는 않습니다.'''),
    code('''
    CUSTOM_INPUT = ROOT / "custom_images"
    CUSTOM_DATA = ROOT / "data/my-workbench"
    PREPARE_CUSTOM = False
    IMAGE_PATH = CUSTOM_INPUT / "test/cup/example.jpg"
    CHECKPOINT = ROOT / "results/gpu/finetuned_checkpoint.npz"
    print("Input exists:", CUSTOM_INPUT.is_dir())
    print("Checkpoint exists:", CHECKPOINT.is_file())
    '''), *data_validation_cells(), *data_saving_cells(),
    md('''## 2. 폴더와 클래스 확인
    사진을 넣은 뒤 `PREPARE_CUSTOM=True`로 바꾸세요. 새 출력 폴더에만 저장합니다. 기존 데이터를 다시 보려면 `False`로 두면 됩니다.'''),
    code('''
    custom_splits = custom_manifest = None
    if PREPARE_CUSTOM:
        if not (CUSTOM_INPUT / "train").is_dir():
            raise FileNotFoundError("custom_images/train/<클래스> 폴더가 필요합니다.")
        if CUSTOM_DATA.exists() and any(CUSTOM_DATA.iterdir()):
            raise FileExistsError("기존 데이터를 덮어쓰지 않습니다. 새 CUSTOM_DATA를 지정하세요.")
        custom_classes = sorted(path.name for path in (CUSTOM_INPUT / "train").iterdir() if path.is_dir())
        if len(custom_classes) < 2:
            raise ValueError("클래스가 2개 이상 필요합니다.")
        for split_name in ("train", "validation", "test"):
            folder = CUSTOM_INPUT / split_name
            if not folder.is_dir() or sorted(path.name for path in folder.iterdir() if path.is_dir()) != custom_classes:
                raise ValueError(f"{split_name}의 클래스 폴더가 train과 다릅니다.")
        print("Classes:", custom_classes)
    else:
        print("사진을 준비했다면 PREPARE_CUSTOM=True로 바꾸세요.")
    '''), md('''## 3. Pillow로 사진을 읽고 중복 확인
    읽은 RGB 픽셀의 SHA256으로 중복 사진을 확인합니다. 서로 다른 크기로 저장한 복사본도 잡을 수 있도록 아래 저장 단계에서 224×224 픽셀을 다시 검사합니다.'''),
    code('''
    if PREPARE_CUSTOM:
        custom_splits, seen_pixels = {}, {}
        for split_name in ("train", "validation", "test"):
            images, labels, image_ids = [], [], []
            for label, class_name in enumerate(custom_classes):
                files = sorted(path for path in (CUSTOM_INPUT / split_name / class_name).iterdir()
                               if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
                if not files:
                    raise ValueError(f"이미지가 없습니다: {split_name}/{class_name}")
                for path in files:
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        fingerprint = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        if fingerprint in seen_pixels:
                            raise ValueError(f"중복 이미지: {seen_pixels[fingerprint]} / {path}")
                        seen_pixels[fingerprint] = str(path)
                        images.append(np.asarray(rgb.resize((224, 224), Image.Resampling.BILINEAR)))
                    labels.append(label)
                    image_ids.append(path.relative_to(CUSTOM_INPUT).as_posix())
            custom_splits[split_name] = {"images": np.stack(images),
                                         "labels": np.asarray(labels, dtype=np.int64), "ids": image_ids}
    '''), md('''## 4. 직접 저장하고 다시 읽기'''),
    code('''
    if PREPARE_CUSTOM:
        custom_manifest = save_prepared(CUSTOM_DATA, custom_splits, custom_classes,
                                       {"name": "custom_workbench", "input_layout": "train,validation,test/class/image"}, 0)
    if (CUSTOM_DATA / "manifest.json").is_file():
        custom_splits, custom_manifest = load_prepared(CUSTOM_DATA)
        print("Classes:", custom_manifest["classes"])
        print("Counts:", {name: len(split["labels"]) for name, split in custom_splits.items()})
    else:
        print("준비한 개인 데이터가 없어 저장·읽기를 생략합니다.")
    '''), md('''## 5. 학습 사진 확인'''),
    code('''
    if custom_manifest is not None:
        fig, axes = plt.subplots(1, len(custom_manifest["classes"]),
                                 figsize=(3 * len(custom_manifest["classes"]), 3), squeeze=False)
        for label, name in enumerate(custom_manifest["classes"]):
            index = int(np.flatnonzero(custom_splits["train"]["labels"] == label)[0])
            axes[0, label].imshow(custom_splits["train"]["images"][index])
            axes[0, label].set_title(name)
            axes[0, label].axis("off")
        fig.suptitle("Your training images · inspect class and background clues")
        fig.tight_layout()
        plt.show()
    else:
        print("개인 사진 갤러리를 생략합니다.")
    '''), md('''## 6. 새 데이터로 추가 학습
    `CUSTOM_DATA`에 만든 `train.npz`, `validation.npz`, `test.npz`, `manifest.json` 네 파일을 Colab CLI의 `upload` 명령으로 하나씩 올립니다. 업로드 대상은 각각 `/content/vision-ai/data/prepared/train.npz`처럼 원격의 `data/prepared` 폴더로 유지합니다. 따라서 `02_gpu_finetuning.ipynb`와 `03_tpu_and_compare.ipynb`의 `DATA_PATH`를 바꾸지 않아도 됩니다. 학습 코드는 manifest에 저장된 클래스 이름과 개수를 읽습니다.

    Colab CLI로 노트북과 이 데이터를 업로드한 뒤 각 장치에서 셀을 실행하세요. 두 장치에서는 같은 학습 설정을 사용합니다. CLI의 생성·업로드·실행·다운로드·종료 명령은 `docs/direct-colab-cli.md`에 있습니다. 다운로드한 GPU 체크포인트는 `results/gpu/finetuned_checkpoint.npz`에 둡니다. 기존 실험 결과를 보존하려면 다운로드 전에 별도 폴더로 옮기세요. 원격 실행을 선택하면 직접 찍은 이미지가 본인의 Colab 런타임으로 전송됩니다.

    아래 단계는 결과를 회수한 뒤 CPU에서 한 장을 추론하는 코드입니다. 경로가 없으면 연산을 생략합니다.'''),
    *device_cells("cpu"), *model_definition_cells(), *preprocessing_cells(),
    md('''## 7. 원본 가중치와 파인튜닝 체크포인트를 직접 읽기
    `safetensors`로 원본 가중치를 읽고 앞에서 정의한 `load_checkpoint`로 업데이트한 마지막 블록·분류기를 덮어씁니다. 원본 모델 파일 자체는 변경하지 않습니다.'''),
    code('''
    from safetensors.numpy import load_file

    MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
    MODEL_SHA256 = "056550dbe6c439dddb35e1800a48aaef86cfdcf8e566ba73f0d1ce41bc7fa1b3"
    READY_TO_INFER = IMAGE_PATH.is_file() and CHECKPOINT.is_file()
    if READY_TO_INFER:
        weights_path = ROOT / "assets/pretrained/model.safetensors"
        if file_sha256(weights_path) != MODEL_SHA256:
            raise ValueError("원본 가중치 SHA256이 다릅니다.")
        config = json.loads((ROOT / "assets/pretrained/config.json").read_text(encoding="utf-8"))
        source_params = {name: jax.device_put(np.asarray(array, np.float32), device)
                         for name, array in load_file(str(weights_path)).items()}
        params, checkpoint_metadata = load_checkpoint(CHECKPOINT, source_params, config, device)
        print("Checkpoint classes:", checkpoint_metadata["classes"])
    else:
        print("IMAGE_PATH와 CHECKPOINT를 실제 파일 경로로 설정하세요. 추론은 아직 실행하지 않습니다.")
    '''), md('''## 8. Pillow 전처리 → JAX 추론 → Matplotlib 표시'''),
    code('''
    if READY_TO_INFER:
        with Image.open(IMAGE_PATH) as image:
            raw = np.asarray(image.convert("RGB"))
        pixels = jax.device_put(preprocess([raw]), device)
        forward = jax.jit(lambda weights, images: original_logits(weights, images, config))
        logits = forward(params, pixels)
        scores = np.asarray(jax.device_get(jax.nn.softmax(logits[0])))
        predicted = int(np.argmax(scores))
        prediction = checkpoint_metadata["classes"][predicted]
        assert len(scores) == len(checkpoint_metadata["classes"]) and np.isfinite(scores).all()
        print("Prediction:", prediction, "softmax:", float(scores[predicted]))
        plt.figure(figsize=(5, 4))
        plt.imshow(raw)
        plt.title(f"{prediction} · softmax {scores[predicted]:.3f}")
        plt.axis("off")
        plt.show()
    else:
        print("사진과 체크포인트가 준비된 뒤 이 셀을 다시 실행하세요.")
    '''), md('''## 제출할 내용
    1. 클래스 구분 기준과 촬영·분할 방식
    2. 같은 테스트 이미지에서 분류기 학습과 파인튜닝의 결과
    3. 맞힌 사진과 놓친 사진, 다음 실험에서 바꿀 조건 한 가지
    4. 실제 GPU·TPU 장치, 결과 회수, 세션 종료 기록

    높은 softmax 점수가 물리 동작의 안전성을 보장하지는 않습니다. 여러 물체가 섞인 장면이나 처음 보는 물체에 적용하려면 별도 검증이 필요합니다.''')]
    return make_notebook(cells)


def make_notebooks():
    return {"00_setup_and_data.ipynb": setup_notebook(),
            "01_pretrained_inference.ipynb": inference_notebook(),
            "04_custom_images.ipynb": custom_notebook()}
