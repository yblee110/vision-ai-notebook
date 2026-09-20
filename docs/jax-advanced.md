# 선택 심화 · JAX와 TPU

기본 과정은 [Hugging Face CLI와 Colab GPU](../hf_colab_gpu/README.md)입니다. 모델을 불러오고 추론·파인튜닝하는 흐름을 먼저 익힌 뒤, 모델 연산과 다른 가속기를 더 살펴보고 싶을 때 이 과정을 진행합니다.

## 기본 설명에서는 5분만 다루기

JAX는 배열 연산·자동 미분·컴파일 도구입니다. TPU는 이런 연산을 실행할 수 있는 가속기입니다. PyTorch를 쓰는 기본 GPU 노트북을 그대로 TPU에 보내는 실습은 아닙니다. 심화 과정의 별도 JAX 노트북을 사용합니다.

기본 수업에서 Attention·LayerNorm을 직접 구현하거나 `jax.jit`의 내부 원리를 설명할 필요는 없습니다. ‘같은 과제를 다른 계산 도구와 장치로 실행할 수 있다’는 정도로 소개하고 실제 코드는 심화에서 봅니다.

## 같은 가상환경에 심화 라이브러리 추가

프로젝트 최상위에서 실행합니다.

```bash
bash scripts/setup.sh --with-jax
source .venv/bin/activate
python scripts/doctor.py --with-jax --data
```

기존 기본 환경에 JAX·Optax·Safetensors를 추가합니다. 별도의 가상환경이나 수업 전용 설치 패키지는 만들지 않습니다.

## 심화 노트북과 자료

| 자료 | 내용 |
|---|---|
| [01_pretrained_inference.ipynb](../notebooks/01_pretrained_inference.ipynb) | JAX 연산으로 구현한 사전학습 추론 |
| [02_gpu_finetuning.ipynb](../notebooks/02_gpu_finetuning.ipynb) | JAX·Optax GPU 학습 |
| [03_tpu_and_compare.ipynb](../notebooks/03_tpu_and_compare.ipynb) | 같은 JAX 학습을 TPU에서 실행 |
| [04_custom_images.ipynb](../notebooks/04_custom_images.ipynb) | 별도 사진과 JAX 체크포인트 적용 |
| [JAX용 Colab CLI 명령](direct-colab-cli.md) | 심화 모델·입력·결과 경로와 장치별 설치 |
| [JAX 학생 교안](handson.md) | 모델 계산과 학습 범위에 대한 자세한 설명 |
| [JAX 기존 실측 결과](direct-test-results.md) | 이전 JAX 구현의 실제 GPU·TPU 검사 |

기본 과정의 `save_pretrained` 폴더와 심화 과정의 JAX NPZ 체크포인트는 형식이 다릅니다. 각 과정의 추론 노트북에서 해당 형식으로 복원하세요. 장치 간 성능을 비교하려면 모델·전처리·최적화·데이터·시간 측정 조건도 같게 맞춰야 합니다. 이번 두 과정의 정확도만 보고 PyTorch와 JAX의 우열을 결론 내리지 않습니다.
