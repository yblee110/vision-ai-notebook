# Vision AI · Hugging Face CLI와 Colab GPU

이 저장소의 실습은 **`hf_colab_gpu/notebooks`의 00·01·02번 세 개**입니다. **모델 다운로드 → GPU 추론 → 우리 물체로 파인튜닝 → 결과 확인** 순서로 진행합니다. 모델 구조는 Transformers로 불러오고, PyTorch로 학습합니다.

**[실습 시작하기 →](hf_colab_gpu/README.md)**

## 처음 시작하는 학생

1. 이 저장소의 **[Fork](https://github.com/yblee110/vision-ai-notebook/fork)**를 눌러 본인 GitHub 계정으로 복사합니다.
2. **본인이 Fork한 저장소**에서 **Code → Codespaces → Create codespace on main**을 선택합니다. 머신을 고르는 경우 **2코어·8GB**를 사용합니다.
3. 자동 환경 설치가 끝날 때까지 기다린 뒤 아래 명령으로 준비 상태를 확인합니다.
4. 기본 노트북 00번을 열고 **커널 선택 → Jupyter 커널 → Vision AI (Codespaces CPU)**를 선택해 실행합니다.

`.devcontainer`와 설치 설정은 Fork에 함께 포함됩니다. 기본 환경은 HF CLI·Colab CLI와 데이터 준비 라이브러리만 설치합니다. Google 로그인과 Colab GPU 할당은 학생이 본인 계정으로 실습 중 진행합니다. Codespaces와 Colab의 사용량은 별도로 관리합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

이미 쓰던 환경이라면 `bash scripts/setup.sh`로 필요한 라이브러리를 추가합니다. 가상환경은 하나를 사용하며, 프로젝트 자체를 설치 패키지로 묶지 않습니다. 코드 셀에서 공개 라이브러리를 직접 불러옵니다.

## 실습 노트북 3개

| 노트북 | 실행 위치 | 하는 일 |
|---|---|---|
| [00 · HF CLI와 데이터 준비](hf_colab_gpu/notebooks/00_hf_download_and_data.ipynb) | Codespaces CPU | 실제 `hf download`, 모델 파일 확인, 이미지 분할 |
| [01 · GPU 추론](hf_colab_gpu/notebooks/01_gpu_inference.ipynb) | Colab GPU | `from_pretrained`, 이미지 전처리, Top-5 예측 |
| [02 · GPU 파인튜닝](hf_colab_gpu/notebooks/02_gpu_finetuning.ipynb) | Colab GPU | 분류기 학습, 마지막 블록 파인튜닝, 비교·저장·재로딩 |

01·02에는 **각각 두 개의 빈 실습 코드 셀**이 있습니다. [문제 조건과 프롬프트](hf_colab_gpu/ai-coding-workshop.md)를 읽고 AI와 코드를 작성한 뒤, GPU를 만들기 전에 `python scripts/check_student_cells.py`로 빈 셀과 문법을 확인합니다. 같은 노트북의 확인 셀과 실행 결과로 답을 점검하며 별도 정답 노트북은 없습니다.

01·02는 Codespaces에서 편집하고 공식 `colab exec -f ...ipynb`로 실행합니다. GPU 연결·설치·입력 전송을 먼저 해야 하므로 [단계별 실행 안내](hf_colab_gpu/README.md)를 따라가세요. Codespaces CPU의 Run All은 Colab GPU에 자동으로 연결되지 않습니다.

## 강의 자료

- [AI 코딩 네 가지 문제와 진행 방법](hf_colab_gpu/ai-coding-workshop.md)
- [학생 교안](hf_colab_gpu/handson.md)
- [이번 HF·PyTorch GPU 실측 결과](hf_colab_gpu/test-results.md)
- [학생 교안과 검증 결과를 합친 최종 교안](hf_colab_gpu/final-guide.md)
- [새 GitHub Codespaces에서 실제 검증한 결과](hf_colab_gpu/codespaces-verification.md)
- [모델·데이터 출처](THIRD_PARTY.md)
- [양자화 선택 읽을거리](hf_colab_gpu/quantization-reference.md)

기본 데이터는 병·그릇·캔·컵·접시 5개 클래스이며 학습 500장·검증 100장·테스트 200장입니다. 32×32 공개 이미지로 실습 흐름을 배우는 것이므로 실제 작업대 카메라 성능과 구분합니다. 원본 ImageNet 1,000개 라벨 추론과 새 5개 라벨 분류의 정확도를 직접 비교하지 않습니다.

`hf_colab_gpu/validation`은 2026-09-19 실행에서 남긴 보고서·CSV·그림을 담은 참고 폴더입니다. 추가로 실행할 노트북은 없습니다. 당시의 완성된 코드로 측정한 수치이므로 학생이 채운 코드의 결과와 구분합니다.

`.devcontainer`, `requirements`, `scripts`, `tests`는 세 노트북의 환경 준비와 검사를 위한 파일입니다. 다운로드한 모델·개인 사진·학습 결과·인증 정보·가상환경은 저장소에 올리지 않습니다.
