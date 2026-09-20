# Vision AI · Hugging Face CLI와 Colab GPU

기본 실습은 **모델 다운로드 → GPU 추론 → 우리 물체로 파인튜닝 → 결과 확인**입니다. 모델 구조는 Transformers로 불러오고, PyTorch로 학습합니다. JAX·TPU는 선택 심화 과정으로 남겼습니다.

**[새 기본 실습 시작하기 →](hf_colab_gpu/README.md)**

## 처음 시작하는 학생

이 폴더의 내용물을 새 GitHub 저장소 최상위에 올리고 Codespace를 만듭니다. `.devcontainer`, `.vision-lab-root` 같은 숨김 파일도 포함합니다. 기본 환경은 HF CLI·Colab CLI와 데이터 준비 라이브러리만 설치합니다. Google 로그인과 GPU 할당은 학생이 실습 중 진행합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

이미 쓰던 환경이라면 `bash scripts/setup.sh`로 필요한 라이브러리를 추가합니다. 가상환경은 하나를 사용하며, 프로젝트 자체를 설치 패키지로 묶지 않습니다. 코드 셀에서 공개 라이브러리를 직접 불러옵니다.

## 기본 노트북 3개

| 노트북 | 실행 위치 | 하는 일 |
|---|---|---|
| [00 · HF CLI와 데이터 준비](hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb) | Codespaces CPU | 실제 `hf download`, 모델 파일 확인, 이미지 분할 |
| [01 · GPU 추론](hf_colab_gpu/notebooks/01_gpu_inference.ipynb) | Colab GPU | `from_pretrained`, 이미지 전처리, Top-5 예측 |
| [02 · GPU 파인튜닝](hf_colab_gpu/notebooks/02_gpu_finetuning.ipynb) | Colab GPU | 분류기 학습, 마지막 블록 파인튜닝, 비교·저장·재로딩 |

01·02는 Codespaces에서 코드를 읽고 공식 `colab exec -f ...ipynb`로 실행합니다. GPU 연결·설치·입력 전송을 먼저 해야 하므로 [단계별 실행 안내](hf_colab_gpu/README.md)를 따라가세요. Codespaces CPU의 Run All은 Colab GPU에 자동으로 연결되지 않습니다.

## 강의 자료

- [기본 과정 학생 교안](hf_colab_gpu/handson.md)
- [이번 HF·PyTorch GPU 실측 결과](hf_colab_gpu/test-results.md)
- [학생 교안과 검증 결과를 합친 최종 교안](hf_colab_gpu/final-guide.md)
- [새 GitHub Codespaces에서 실제 검증한 결과](hf_colab_gpu/codespaces-verification.md)
- [JAX·TPU 선택 심화 안내](docs/jax-advanced.md)
- [모델·데이터 출처](THIRD_PARTY.md)
- [양자화 선택 읽을거리](docs/quantization-reference.md)

기본 데이터는 병·그릇·캔·컵·접시 5개 클래스이며 학습 500장·검증 100장·테스트 200장입니다. 32×32 공개 이미지로 실습 흐름을 배우는 것이므로 실제 작업대 카메라 성능과 구분합니다. 원본 ImageNet 1,000개 라벨 추론과 새 5개 라벨 분류의 정확도를 직접 비교하지 않습니다.

기존 JAX 노트북과 검증 기록은 심화 자료로 보존했습니다. 새 PyTorch 과정의 결과와 이전 JAX 수치를 섞지 않습니다. 다운로드한 모델·개인 사진·학습 결과·인증 정보·가상환경은 학생 배포본에서 제외합니다.
