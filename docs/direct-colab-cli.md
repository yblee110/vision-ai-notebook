# JAX·TPU 선택 심화 · Colab CLI 직접 실행

이 문서는 JAX·TPU 선택 심화 과정입니다. [기본 HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)을 먼저 진행하세요. 심화 환경은 `bash scripts/setup.sh --with-jax`로 준비합니다.

Codespaces는 코드 편집과 데이터 준비를 맡고, GPU·TPU 학습은 별도의 Colab 컴퓨터에서 실행합니다. 아래 명령은 **Codespaces 터미널에서**, `.vision-lab-root`가 있는 프로젝트 폴더를 기준으로 한 블록씩 실행합니다.

02번·03번 노트북의 코드 셀에는 모델과 학습 코드가 그대로 들어 있습니다. Codespaces에서 셀을 실행하면 Codespaces CPU에서 실행됩니다. GPU·TPU 노트북은 이 문서의 `colab exec -f ...ipynb`로 Colab에서 실행하세요. 먼저 노트북을 열고 각 셀의 import와 연산을 읽은 뒤 실행하면 됩니다.

## 1. 환경, 데이터와 Google 로그인

00번 노트북에서 데이터 다운로드·이미지 전처리·분할 저장을 끝내고 01번에서 CPU 추론을 확인합니다. 그다음 환경과 데이터를 검사합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py --data
colab --help
colab --auth oauth2 sessions
```

CLI 0.6.0에는 `colab login` 명령이 없습니다. `colab --auth oauth2 sessions`가 서버의 세션 목록을 조회하며 필요한 경우 Google 로그인을 안내합니다. 브라우저에서 인증을 마친 뒤 터미널에 인증 코드를 입력합니다. 뒤의 `colab exec <<'PY'`처럼 표준 입력으로 코드를 전달하는 명령보다 **이 로그인 단계를 먼저 완료**해야 합니다.

기본 방식은 OAuth2입니다. 이미 `gcloud`로 Colab 범위의 Application Default Credentials를 설정한 경우에만 `colab --auth adc ...`를 사용할 수 있습니다. ADC를 쓰면 뒤의 모든 명령에도 같은 전역 옵션을 적용합니다. 이 수업의 Codespaces 설정은 gcloud 설치나 ADC 인증을 자동으로 수행하지 않습니다.

세션 목록에서 다른 실습에 사용하는 세션이 있는지 확인합니다. 아래에서는 `vision-gpu`, `vision-tpu`라는 이름을 씁니다. 이미 같은 이름의 세션이 있다면 결과를 보존하고 종료하거나 이번 실습용 이름을 정해 이후 명령에도 일관되게 사용하세요. 학습은 아래에서 새로 만드는 세션으로 시작합니다.

## 2. GPU 세션과 라이브러리 준비

```bash
colab new -s vision-gpu --gpu T4
colab status -s vision-gpu
colab install -s vision-gpu -r requirements/gpu.txt
colab restart-kernel -s vision-gpu
```

`new`에서 Colab 자원을 할당하고 CU를 사용하기 시작합니다. T4 할당 가능 여부는 계정·잔액·가용량에 따라 달라집니다. 설치 목록은 NumPy, Pillow, Safetensors, JAX CUDA 12, Optax, Matplotlib이며 `requirements/gpu.txt`에서 확인할 수 있습니다. 설치 전에 메모리에 들어간 라이브러리와 새 버전이 섞이지 않도록 커널을 다시 시작합니다.

`colab install -r`은 **Codespaces에 있는** requirements 파일을 원격으로 보내 설치합니다. 수업 프로젝트를 하나의 패키지로 설치하지 않습니다.

## 3. GPU에 필요한 파일을 하나씩 올리기

`upload`는 단일 파일을 전송하며 상위 폴더를 자동으로 만들지 않습니다. 다음 코드는 Colab에 폴더만 준비합니다. 모델이나 학습 코드는 이 코드에 넣지 않습니다.

```bash
colab exec -s vision-gpu --timeout 120 <<'PY'
from pathlib import Path

root = Path('/content/vision-ai')
(root / 'assets' / 'pretrained').mkdir(parents=True, exist_ok=True)
(root / 'data' / 'prepared').mkdir(parents=True, exist_ok=True)
(root / 'results' / 'gpu').mkdir(parents=True, exist_ok=True)
PY
```

Colab Python의 실제 경로는 `/content/vision-ai`입니다. 아래 파일 전송 경로 `content/vision-ai/...`는 Jupyter 파일 API의 경로이며 같은 위치를 가리킵니다.

```bash
colab upload -s vision-gpu .vision-lab-root content/vision-ai/.vision-lab-root
colab upload -s vision-gpu assets/pretrained/model.safetensors content/vision-ai/assets/pretrained/model.safetensors
colab upload -s vision-gpu assets/pretrained/config.json content/vision-ai/assets/pretrained/config.json
colab upload -s vision-gpu assets/pretrained/preprocessor_config.json content/vision-ai/assets/pretrained/preprocessor_config.json
colab upload -s vision-gpu assets/pretrained/source.json content/vision-ai/assets/pretrained/source.json
colab upload -s vision-gpu data/prepared/manifest.json content/vision-ai/data/prepared/manifest.json
colab upload -s vision-gpu data/prepared/train.npz content/vision-ai/data/prepared/train.npz
colab upload -s vision-gpu data/prepared/validation.npz content/vision-ai/data/prepared/validation.npz
colab upload -s vision-gpu data/prepared/test.npz content/vision-ai/data/prepared/test.npz
```

## 4. 읽어 본 GPU 노트북을 그대로 실행하기

```bash
colab exec -s vision-gpu -f notebooks/02_gpu_finetuning.ipynb --timeout 1800
```

CLI는 이 파일의 코드 셀을 순서대로 Colab에서 실행합니다. 코드 안의 `import jax`, `import optax`, 모델 함수, 학습 반복문이 그대로 실행됩니다. 노트북 파일을 전송했다고 해서 데이터나 모델 파일까지 자동으로 올라가는 것은 아니므로 3단계가 필요합니다.

실행 결과가 담긴 `notebooks/02_gpu_finetuning_output.ipynb`는 **Codespaces에** 저장됩니다. 각 셀의 출력과 오류를 확인하세요. CLI 0.6.0에서는 원격 셀 오류가 출력되어도 뒤의 셀이 진행될 수 있으므로 명령 종료코드만으로 학습 성공을 판단하지 않습니다. `--timeout 1800`은 각 코드 셀의 실행 대기시간이며, 런타임을 자동으로 종료하는 예약 시간이 아닙니다.

## 5. GPU 결과 회수와 종료

아래 명령은 `results/gpu`에 같은 이름의 파일이 있으면 덮어씁니다. 이전 결과를 남기려면 먼저 해당 폴더를 다른 이름으로 옮겨 두세요.

```bash
mkdir -p results/gpu
colab download -s vision-gpu content/vision-ai/results/gpu/report.json results/gpu/report.json
colab download -s vision-gpu content/vision-ai/results/gpu/head_checkpoint.npz results/gpu/head_checkpoint.npz
colab download -s vision-gpu content/vision-ai/results/gpu/finetuned_checkpoint.npz results/gpu/finetuned_checkpoint.npz
colab download -s vision-gpu content/vision-ai/results/gpu/training.csv results/gpu/training.csv
colab download -s vision-gpu content/vision-ai/results/gpu/predictions.csv results/gpu/predictions.csv
```

파일 회수를 확인한 뒤 이 실습에서 만든 GPU 세션을 종료합니다.

```bash
colab stop -s vision-gpu
colab sessions
```

이후 03번 노트북의 비교에서는 내려받은 `results/gpu/report.json`을 사용합니다. Codespaces나 브라우저 창을 닫는 것만으로 Colab 세션이 종료되지는 않습니다. 학습·다운로드가 실패해도 세션 종료 단계는 실행하세요. 남아 있는 파일 중 필요한 것을 회수한 뒤, **이 실습에서 만든 세션**을 `stop`하고 `sessions`로 확인합니다.

## 6. TPU에서도 같은 흐름을 직접 실행하기

GPU 세션을 종료한 뒤 TPU 세션을 만듭니다. TPU용 라이브러리를 설치하고 커널을 다시 시작합니다.

```bash
colab new -s vision-tpu --tpu v5e1
colab status -s vision-tpu
colab install -s vision-tpu -r requirements/tpu.txt
colab restart-kernel -s vision-tpu
```

TPU에서는 `jax[tpu]`를 설치하지만 노트북에서는 GPU와 동일하게 `import jax`를 사용합니다. TPU가 할당되지 않으면 CPU로 바꾸어 완료했다고 처리하지 말고 계정과 가용량을 확인합니다.

```bash
colab exec -s vision-tpu --timeout 120 <<'PY'
from pathlib import Path

root = Path('/content/vision-ai')
(root / 'assets' / 'pretrained').mkdir(parents=True, exist_ok=True)
(root / 'data' / 'prepared').mkdir(parents=True, exist_ok=True)
(root / 'results' / 'gpu').mkdir(parents=True, exist_ok=True)
(root / 'results' / 'tpu').mkdir(parents=True, exist_ok=True)
PY
```

TPU는 GPU와 다른 컴퓨터이므로 입력 파일을 다시 올립니다. GPU 비교 보고서도 명시적으로 전송합니다.

```bash
colab upload -s vision-tpu .vision-lab-root content/vision-ai/.vision-lab-root
colab upload -s vision-tpu assets/pretrained/model.safetensors content/vision-ai/assets/pretrained/model.safetensors
colab upload -s vision-tpu assets/pretrained/config.json content/vision-ai/assets/pretrained/config.json
colab upload -s vision-tpu assets/pretrained/preprocessor_config.json content/vision-ai/assets/pretrained/preprocessor_config.json
colab upload -s vision-tpu assets/pretrained/source.json content/vision-ai/assets/pretrained/source.json
colab upload -s vision-tpu data/prepared/manifest.json content/vision-ai/data/prepared/manifest.json
colab upload -s vision-tpu data/prepared/train.npz content/vision-ai/data/prepared/train.npz
colab upload -s vision-tpu data/prepared/validation.npz content/vision-ai/data/prepared/validation.npz
colab upload -s vision-tpu data/prepared/test.npz content/vision-ai/data/prepared/test.npz
colab upload -s vision-tpu results/gpu/report.json content/vision-ai/results/gpu/report.json
colab exec -s vision-tpu -f notebooks/03_tpu_and_compare.ipynb --timeout 1800
```

`notebooks/03_tpu_and_compare_output.ipynb`에서 TPU 장치 확인, 학습, GPU와의 비교 결과를 확인합니다.

```bash
mkdir -p results/tpu
colab download -s vision-tpu content/vision-ai/results/tpu/report.json results/tpu/report.json
colab download -s vision-tpu content/vision-ai/results/tpu/head_checkpoint.npz results/tpu/head_checkpoint.npz
colab download -s vision-tpu content/vision-ai/results/tpu/finetuned_checkpoint.npz results/tpu/finetuned_checkpoint.npz
colab download -s vision-tpu content/vision-ai/results/tpu/training.csv results/tpu/training.csv
colab download -s vision-tpu content/vision-ai/results/tpu/predictions.csv results/tpu/predictions.csv
colab stop -s vision-tpu
colab sessions
```

`results/tpu`의 동명 파일도 덮어쓰므로 이전 실험을 보존하려면 다운로드 전에 옮겨 둡니다. 다운로드 파일은 학습 결과이고, 출력 노트북은 각 셀의 실행 기록입니다. 둘 다 남겨 두면 결과와 실행 과정을 함께 확인할 수 있습니다. TPU도 실행·다운로드 오류가 나면 이 실습의 세션 종료를 빠뜨리지 않습니다.

## 7. 회수한 결과 확인

다음 코드는 Codespaces에서 실행합니다. 출력 노트북의 오류 기록과 보고서를 함께 확인합니다. 보고서가 존재한다는 사실만으로 이번 실행이 성공했다고 판단하지 않습니다.

```python
import json
from pathlib import Path

root = Path.cwd()  # 터미널에서 프로젝트 폴더를 기준으로 실행합니다.
for device, notebook in (
    ("gpu", "02_gpu_finetuning_output.ipynb"),
    ("tpu", "03_tpu_and_compare_output.ipynb"),
):
    executed = json.loads((root / "notebooks" / notebook).read_text())
    errors = [
        output
        for cell in executed["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, f"{device}: 출력 노트북에 오류가 있습니다."
    report = json.loads((root / "results" / device / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["device"]["platform"] == device
    for filename in (
        "head_checkpoint.npz", "finetuned_checkpoint.npz",
        "training.csv", "predictions.csv",
    ):
        assert (root / "results" / device / filename).is_file(), filename
    print(device, report["status"], report["device"])
```

04번 노트북에서는 내려받은 체크포인트를 직접 읽고 새 이미지의 추론 결과를 확인합니다. GPU·TPU 세션은 결과 회수 후 종료한 상태로 진행합니다.

## 명령이 어떤 일을 하는지 한눈에 보기

| 명령 | 일어나는 일 |
|---|---|
| `colab --auth oauth2 sessions` | 필요한 로그인 후 서버 세션 목록 조회 |
| `colab new` | 새 Colab 컴퓨터 할당 |
| `colab install -r ...` | 명시한 공개 라이브러리 설치 |
| `colab restart-kernel` | Python 커널 재시작 |
| `colab upload` | 입력 파일 한 개 전송 |
| `colab exec -f ...ipynb` | 노트북의 보이는 코드 셀을 원격 실행 |
| `colab download` | 결과 파일 한 개 회수 |
| `colab stop` | 해당 세션 종료 |
| `colab sessions` | 종료 결과와 남은 세션 확인 |

이 문서의 명령과 `.ipynb` 실행 방식은 설치된 Colab CLI 0.6.0 코드를 확인해 작성했습니다. 참고: [공식 CLI 저장소](https://github.com/googlecolab/google-colab-cli), [0.6.0 노트북 실행 코드](https://github.com/googlecolab/google-colab-cli/blob/v0.6.0/src/colab_cli/commands/execution.py), [0.6.0 파일 전송 코드](https://github.com/googlecolab/google-colab-cli/blob/v0.6.0/src/colab_cli/contents.py).
