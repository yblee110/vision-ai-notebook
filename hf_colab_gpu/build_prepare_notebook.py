"""Maintainer tool: copy visible data cells and add the explicit hf CLI steps."""
from pathlib import Path
import hashlib
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebook_sources.common import md, code
from notebook_sources.data_and_inference import setup_notebook

book = setup_notebook()
book.cells[-1].source = book.cells[-1].source.replace('01_pretrained_inference.ipynb', '01_gpu_inference.ipynb')
book.cells[0] = md('''# 00 · Hugging Face CLI로 모델과 이미지 준비하기
**실행 위치: Codespaces CPU.** 이 노트북은 Colab GPU를 생성하지 않습니다.

오늘의 흐름은 **모델 다운로드 → Colab GPU 연결 → 추론 → 추가 학습 → 결과 회수**입니다.
모델 내부의 Attention 계산은 Transformers가 담당합니다. JAX·TPU는 선택 심화 과정에서 다룹니다.

먼저 프로젝트 최상위에서 `bash scripts/setup.sh`를 실행하고 커널을 **Vision AI (Codespaces CPU)**로 선택하세요.
각 셀에서 공개 라이브러리를 직접 가져오며 별도의 수업 패키지를 설치하지 않습니다.''')

# The generated notebook contains all data code; these authoring imports are never
# executed by students. Insert the download after path/library setup.
checksums = {
    name: hashlib.sha256((ROOT / 'assets/pretrained' / name).read_bytes()).hexdigest()
    for name in ('config.json', 'preprocessor_config.json')
}
checksums['pytorch_model.bin'] = 'e1a51b0c81ff812e079d2189352747947879fee56429e4480064a6695cb34b2c'
download_cells = [
    md('''## Hugging Face CLI와 Colab CLI가 하는 일
    `hf download`는 모델 파일을 Codespaces에 내려받습니다. `colab`은 다른 컴퓨터인 Colab GPU를 만들고 파일 전송·실행·종료를 맡습니다.
    모델을 내려받는 것만으로 GPU가 연결되지는 않습니다.

    이번 공개 모델은 Hugging Face 로그인이 필요하지 않습니다. Google 로그인은 다음 Colab 연결 단계에서 별도로 진행합니다.
    여기서는 고정한 revision의 세 파일만 받습니다. 이 revision의 원본 가중치 형식은 `pytorch_model.bin`입니다.'''),
    code('''
    import subprocess

    HF = Path(sys.executable).parent / "hf"
    if not HF.is_file():
        raise RuntimeError("현재 커널에 hf CLI가 없습니다. scripts/setup.sh 실행 후 커널을 다시 선택하세요.")
    print("Hugging Face Hub:", version("huggingface_hub"))
    print("CLI:", HF)
    MODEL_ID = "facebook/deit-tiny-patch16-224"
    MODEL_REVISION = "b3428f18dcc7b543470d07f14b4a4157815d1880"
    MODEL_DIR = ROOT / "hf_colab_gpu/models/deit-tiny"
    MODEL_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
    '''),
    md('''### 실제 `hf download` 실행
    아래 셀의 인자 목록은 터미널의 다음 명령과 같습니다. `subprocess.run`은 Python 표준 라이브러리이며, 목록에 적힌 공개 CLI를 그대로 실행합니다.

    ```bash
    hf download facebook/deit-tiny-patch16-224 config.json preprocessor_config.json pytorch_model.bin \\
      --revision b3428f18dcc7b543470d07f14b4a4157815d1880 \\
      --local-dir hf_colab_gpu/models/deit-tiny
    ```

    모델 파일은 약 23 MB입니다. 수업에서 확인한 파일 지문과 대조한 뒤 사용합니다.'''),
    code('''
    download_env = os.environ.copy()
    download_env["HF_HOME"] = str(ROOT / ".cache/huggingface")
    download_env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    download_env["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    command = [str(HF), "download", MODEL_ID, *MODEL_FILES,
               "--revision", MODEL_REVISION, "--local-dir", str(MODEL_DIR)]
    print(" ".join(command))
    subprocess.run(command, check=True, env=download_env, timeout=600)
    '''),
    code(f'''
    EXPECTED_MODEL_SHA256 = {checksums!r}
    downloaded_hashes = {{}}
    for filename in MODEL_FILES:
        path = MODEL_DIR / filename
        downloaded_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if downloaded_hashes[filename] != EXPECTED_MODEL_SHA256[filename]:
            raise ValueError(f"모델 파일이 검증한 원본과 다릅니다: {{filename}}")
        print(filename, path.stat().st_size, "bytes · SHA256 확인")
    download_manifest = {{"model_id": MODEL_ID, "revision": MODEL_REVISION,
                         "files": downloaded_hashes, "download_method": "hf download"}}
    (MODEL_DIR / "download_manifest.json").write_text(
        json.dumps(download_manifest, indent=2), encoding="utf-8")
    '''),
    md('''## 같은 물체 이미지 데이터 준비
    아래는 NumPy·Pillow·PyArrow로 이미지 데이터를 읽고 나누는 과정입니다. JAX나 PyTorch는 사용하지 않습니다.
    병·그릇·캔·컵·접시를 학습 500장, 검증 100장, 테스트 200장으로 나눕니다.
    이미 같은 조건의 데이터를 만들었다면 파일의 지문과 분할 조건을 확인한 뒤 재사용합니다.'''),
]
book.cells[4:4] = download_cells
book.cells.append(md('''## 다음 단계 · Colab GPU에 연결하기
이제 Codespaces에 모델 3개 파일·다운로드 기록·준비 데이터 4개 파일이 있습니다.
[실행 안내](../README.md)를 따라 GPU 세션을 만들고, 이 파일들을 하나씩 올립니다.
그다음 `01_gpu_inference.ipynb`, `02_gpu_finetuning.ipynb`를 공식 Colab CLI로 실행합니다.

첫 추론에서는 `AutoImageProcessor`, `AutoModelForImageClassification`이 모델과 전처리를 불러옵니다.
GPU 노트북을 Codespaces CPU에서 실행해도 원격 GPU로 자동 연결되지는 않습니다.'''))
book.metadata['vision_ai']['course_scope'] = 'Hugging Face CLI preparation; no JAX or PyTorch execution'
target = ROOT / 'hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb'
target.parent.mkdir(parents=True, exist_ok=True)
nbformat.validate(book)
nbformat.write(book, target)
print(target, len(book.cells), 'cells')
