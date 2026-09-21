"""Turn the inference notebook into two guided, answer-free AI coding exercises."""
from copy import deepcopy
from textwrap import dedent

import nbformat as nbf


def _source(cell):
    value = cell["source"]
    return value if isinstance(value, str) else "".join(value)


def _md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def _code(text, *, exercise_id=None):
    source = dedent(text).strip()
    if len(source.splitlines()) > 40:
        raise ValueError(f"Provided cell exceeds 40 lines: {source[:70]}")
    metadata = {}
    if exercise_id is not None:
        metadata = {"tags": ["exercise-check"], "exercise_id": exercise_id}
    return nbf.v4.new_code_cell(source, metadata=metadata)


def _task(exercise_id, expected_function):
    return nbf.v4.new_code_cell("", metadata={
        "tags": ["student-task"], "exercise_id": exercise_id,
        "expected_function": expected_function,
    })


def adapt_inference(cells):
    """Adapt the original 13 cells; fail loudly if their source layout changes."""
    assert len(cells) == 13, "Original inference notebook layout changed"
    anchors = {
        0: "# Hugging Face 모델을 Colab GPU에서 추론하기",
        8: "## 4. 이미지 한 장을 GPU에서 추론하기",
        9: 'target_class = classes.index("cup")',
        10: "fig, axes = plt.subplots",
        11: "## 5. 실행 기록 확인",
        12: "inference_report = {",
    }
    for index, prefix in anchors.items():
        assert _source(cells[index]).startswith(prefix), (index, prefix)
    original = [nbf.from_dict(deepcopy(cell)) for cell in cells]
    for cell in original:
        cell["source"] = _source(cell)
    original[0]["source"] += dedent("""

        **실습 방법:** 아래 빈 코드 셀 두 곳에 AI와 함께 만든 함수를 넣습니다.
        각 문제의 입력과 조건을 읽고, 제시한 프롬프트를 자신의 말로 바꿔 요청해 보세요.
        코드를 넣은 뒤 바로 아래 확인 셀을 실행합니다. 오류가 나면 오류 메시지와 문제 조건을
        AI에게 함께 보여 주고 수정합니다. 조건을 바꿔 다시 시도한 뒤 같은 노트북에서 결과와 해설을 확인합니다.
        처음에는 위에서부터 실행하고, 입력을 바꿀 때는 해당 준비 셀부터 결과 저장 셀까지 다시 실행하세요.
        빈 셀을 채우지 않은 상태에서 전체 실행하면 확인 셀에서 멈추는 것이 정상입니다.
        정답 코드가 있는 별도 노트북은 없습니다. 제시된 수치 예시는 연습용이며 실제 GPU 결과와 다를 수 있습니다.
    """)
    reset = _code('''
        INFERENCE_CHECKS = {"01-infer": False, "01-topk": False}
        for name in ("infer_probabilities", "topk_predictions", "probabilities", "predictions"):
            globals().pop(name, None)
    ''')
    guard = '''if not all(globals().get("INFERENCE_CHECKS", {}).get(key, False)
           for key in ("01-infer", "01-topk")):
    raise RuntimeError("두 문제의 확인 셀을 모두 통과한 뒤 결과를 확인하세요.")
'''
    original[10]["source"] = guard + original[10]["source"]
    original[12]["source"] = guard + original[12]["source"]
    result = [original[0], reset, *original[1:8], _md("""
        ## 4. 실습 1 — 이미지에서 확률 구하기

        모델에 사진 한 장을 넣고, 1,000개 분류 각각의 확률을 받는 함수 하나를 만듭니다.
        모델과 전처리 도구는 이미 준비되어 있습니다. 설치하거나 새 모델을 내려받을 필요가 없습니다.

        **제공 입력**

        - `model`: GPU에 올라간 사전학습 모델
        - `processor`: 이미지 크기와 색상 값을 모델에 맞추는 도구
        - `image`: RGB 형식의 PIL 이미지 한 장
        - `device`: 이번 세션의 GPU 장치

        **완성 조건**

        1. 함수 이름과 인자는 `infer_probabilities(model, processor, image, device)`로 정합니다.
        2. `processor`로 이미지를 PyTorch 텐서로 바꾸고, 모든 입력 텐서를 `device`로 옮깁니다.
        3. `model.eval()`과 `torch.inference_mode()`를 사용합니다. 모델 가중치를 바꾸거나 학습하지 않습니다.
        4. 모델의 `logits`에 마지막 축 기준 `softmax`를 적용합니다.
           이미지 한 장의 결과만 꺼내 **1차원 PyTorch 텐서**로 반환합니다. 출력만 하는 함수는 통과하지 못합니다.

        **예상 결과의 형태:** `torch.Size([1000])`, GPU 텐서, 확률 합계는 약 `1.0`,
        각 값은 `0` 이상 `1` 이하입니다. 이는 출력 조건이며 특정 정답 확률을 뜻하지 않습니다.

        **AI에게 요청할 프롬프트**

        > 저는 Python 초보자입니다. 위 입력과 완성 조건에 맞게
        > `infer_probabilities(model, processor, image, device)` 함수만 작성해 주세요.
        > 이미 준비된 객체를 사용하고 설치·다운로드·학습 코드는 넣지 마세요.
        > 전처리, GPU 이동, 평가 모드, 추론 모드, softmax가 각각 필요한 이유를 짧게 설명해 주세요.
        > 사진이 바뀌어도 작동해야 하며 숫자를 고정해서 반환하면 안 됩니다.
    """), _code('''
        # 먼저 그대로 실행하고, 성공한 뒤 다른 종류나 사진 번호를 선택해 보세요.
        SAMPLE_CLASS = "cup"
        SAMPLE_OFFSET = 0
        INFERENCE_CHECKS.update({"01-infer": False, "01-topk": False})
        for name in ("probabilities", "predictions"):
            globals().pop(name, None)
        assert SAMPLE_CLASS in classes, f"선택 가능한 종류: {classes}"
        target_class = classes.index(SAMPLE_CLASS)
        sample_indices = np.flatnonzero(splits["test"]["labels"] == target_class)
        assert isinstance(SAMPLE_OFFSET, int) and 0 <= SAMPLE_OFFSET < len(sample_indices)
        image_index = int(sample_indices[SAMPLE_OFFSET])
        image = Image.fromarray(splits["test"]["images"][image_index]).convert("RGB")

        def model_fingerprint(model):
            digest = hashlib.sha256()
            for name, value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        print("사진의 실제 종류:", SAMPLE_CLASS, "· 같은 종류의 사진 번호:", SAMPLE_OFFSET)
        print("입력 형식:", image.mode, "· 원본 크기:", image.size)
    '''), _md("""
        **아래 빈 코드 셀을 채우세요.** 함수만 넣고, 다음 확인 셀에서 실행 결과를 확인합니다.
    """), _task("01-infer", "infer_probabilities"), _code('''
        INFERENCE_CHECKS.update({"01-infer": False, "01-topk": False})
        globals().pop("probabilities", None)
        globals().pop("predictions", None)
        if not callable(globals().get("infer_probabilities")):
            raise RuntimeError("실습 1의 빈 셀에 infer_probabilities 함수를 만들고 실행하세요.")
        weights_before = model_fingerprint(model)
        model.train()  # 학생 함수가 평가 모드로 전환하는지도 확인합니다.
        forward_calls = []
        def record_forward(module, args, output):
            forward_calls.append((output.logits.detach().clone(), torch.is_grad_enabled(),
                                  torch.is_inference_mode_enabled(), module.training))
        hook = model.register_forward_hook(record_forward)
        try:
            candidate = infer_probabilities(model, processor, image, device)
        finally:
            hook.remove()
        assert forward_calls, "확률을 직접 만들지 말고 제공된 모델을 호출하세요."
        assert all(not grad and inference and not training for _, grad, inference, training in forward_calls), "평가 모드와 추론 모드 안에서 모델을 실행하세요."
        assert isinstance(candidate, torch.Tensor), "반환값은 PyTorch 텐서여야 합니다."
        assert candidate.ndim == 1 and len(candidate) == model.config.num_labels, "1차원 확률을 반환하세요."
        assert candidate.device == next(model.parameters()).device, "확률도 모델과 같은 GPU에 있어야 합니다."
        assert not candidate.requires_grad, "추론 모드를 사용하세요."
        assert not model.training, "모델을 평가 모드로 전환하세요."
        assert torch.isfinite(candidate).all().item(), "확률에 NaN이나 무한대가 있습니다."
        assert ((candidate >= 0) & (candidate <= 1)).all().item(), "확률 범위는 0~1입니다."
        assert abs(float(candidate.sum()) - 1.0) < 1e-5, "softmax와 확률 합계를 확인하세요."
        torch.testing.assert_close(candidate, forward_calls[-1][0].softmax(dim=-1)[0])
        assert weights_before == model_fingerprint(model), "가중치가 바뀌었습니다. 모델 불러오기 셀부터 다시 실행하세요."
        probabilities = candidate
        INFERENCE_CHECKS["01-infer"] = True
        print("실습 1 확인 통과")
        print("출력 크기:", probabilities.shape, "· 확률 합계:", f"{float(probabilities.sum()):.6f}")
        print("GPU:", probabilities.device, "· 기울기 계산:", probabilities.requires_grad)
    ''', exercise_id="01-infer"), _md("""
        **다르게 시도하기:** 준비 셀의 `SAMPLE_CLASS`를 `bottle`로 바꾸거나 `SAMPLE_OFFSET`을 `1`로
        바꿔 보세요. 준비 셀부터 다시 실행했을 때 확률 벡터의 길이와 합계는 어떻게 될지 먼저 예상해 보세요.
        이후 실습 2와 그래프까지 실행하면 사진에 따라 어떤 예측이 달라졌는지 볼 수 있습니다.

        <details>
        <summary>시도한 뒤 결과 해설 보기</summary>

        사진이 바뀌어도 모델이 구분하는 항목은 1,000개이므로 벡터 길이는 같습니다.
        softmax를 거친 확률 합계도 약 1입니다. 개별 항목의 확률은 달라질 수 있습니다.
        추론은 가중치를 수정하는 과정이 아닙니다. 평가 모드와 추론 모드 역시 서로 다른 역할을 하므로
        둘 다 사용합니다. 합계가 1이라는 사실만으로 예측이 정확하다고 할 수는 없습니다.

        </details>
    """), _md("""
        ## 5. 실습 2 — 확률을 사람이 읽는 상위 예측으로 바꾸기

        숫자 1,000개에서 확률이 높은 항목을 골라 이름과 함께 정리합니다.

        **제공 입력:** 앞 문제의 `probabilities`, 번호를 이름으로 바꾸는 `model.config.id2label`,
        보여 줄 항목 수 `k`입니다. 여기서는 **원래 모델의 ImageNet 1,000개 이름**을 사용합니다.
        실습 데이터의 `classes`는 5개 정답 종류이므로 예측 번호를 이 목록에 연결하면 안 됩니다.

        **완성 조건**

        1. `topk_predictions(probabilities, id2label, k)` 함수를 만듭니다.
        2. 큰 확률부터 `k`개를 골라 `label`과 `probability`를 가진 딕셔너리 목록으로 반환합니다.
        3. `probability`는 0~1 범위의 Python `float`입니다. 퍼센트 문자열 변환은 화면에 출력할 때만 합니다.
        4. `k`가 `1`, `3`, `5`일 때 모두 작동하고 입력 확률 텐서와 이름 목록을 변경하지 않아야 합니다.

        **연습용 예상값:** `[0.10, 0.60, 0.03, 0.20, 0.07]`과 순서대로 `라벨 A`~`라벨 E`를 넣고
        `k=3`이면 `라벨 B 60% → 라벨 D 20% → 라벨 A 10%` 순서입니다.
        실제 사진의 이름·확률은 실행 결과로 확인하며 이 예시와 같을 필요가 없습니다.

        **AI에게 요청할 프롬프트**

        > 위 조건에 맞는 `topk_predictions(probabilities, id2label, k)` 함수를 작성해 주세요.
        > `torch.topk`를 사용해 높은 확률부터 정렬하고, 모델의 `id2label`로 이름을 찾아 주세요.
        > CPU와 GPU에서 모두 작동하고 원래 입력은 바꾸지 않아야 합니다.
        > 정답 데이터의 5개 클래스 이름이나 예시 확률을 코드에 고정하지 마세요.
        > 반환값의 자료형과, k를 바꾸면 무엇이 달라지는지도 설명해 주세요.

        **아래 빈 코드 셀을 채운 뒤 확인 셀을 실행하세요.**
    """), _task("01-topk", "topk_predictions"), _code('''
        INFERENCE_CHECKS["01-topk"] = False
        globals().pop("predictions", None)
        if not INFERENCE_CHECKS.get("01-infer", False):
            raise RuntimeError("먼저 실습 1의 확인 셀을 통과하세요.")
        if not callable(globals().get("topk_predictions")):
            raise RuntimeError("실습 2의 빈 셀에 topk_predictions 함수를 만들고 실행하세요.")
        example = torch.tensor([0.10, 0.60, 0.03, 0.20, 0.07])
        example_labels = dict(enumerate(["라벨 A", "라벨 B", "라벨 C", "라벨 D", "라벨 E"]))
        for source, names in ((example, example_labels), (probabilities, model.config.id2label.copy())):
            unchanged = source.clone()
            unchanged_names = names.copy()
            for k in (1, 3, 5):
                answer = topk_predictions(source, names, k)
                assert isinstance(answer, list) and len(answer) == k, "길이가 k인 목록을 반환하세요."
                expected_values, expected_indices = source.topk(k)
                for item, value, index in zip(answer, expected_values, expected_indices):
                    assert isinstance(item, dict) and set(item) == {"label", "probability"}
                    assert item["label"] == names[int(index)], "번호와 이름의 연결 또는 순서를 확인하세요."
                    assert type(item["probability"]) is float, "확률을 Python float로 반환하세요."
                    assert abs(item["probability"] - float(value)) < 1e-6, "확률 값을 변경하지 마세요."
                assert torch.equal(source, unchanged), "입력 확률을 변경하지 마세요."
                assert names == unchanged_names, "번호와 이름의 연결을 변경하지 마세요."
        predictions = topk_predictions(probabilities, model.config.id2label.copy(), 5)
        INFERENCE_CHECKS["01-topk"] = True
        print("실습 2 확인 통과 · 예시 입력과 실제 GPU 결과에서 k=1, 3, 5 검증")
        print("실습 데이터 정답:", classes[target_class])
        for item in predictions:
            print(f"{item['probability']:6.2%}  {item['label']}")
    ''', exercise_id="01-topk"), _md("""
        **다르게 시도하기:** 먼저 `k=1`과 `k=3` 중 어떤 출력이 판단하기 쉬울지 예상해 보세요.
        아래 셀은 두 결과를 나란히 출력합니다. 이 셀의 `k` 목록을 바꿔 더 살펴봐도 됩니다.
        아래 그래프와 저장 파일은 비교 기준을 맞추기 위해 계속 상위 5개를 사용합니다.
    """), _code('''
        if not all(INFERENCE_CHECKS.values()):
            raise RuntimeError("두 문제의 확인 셀을 먼저 통과하세요.")
        for k in (1, 3):
            print(f"\\n상위 {k}개를 보여 줄 때")
            for item in topk_predictions(probabilities, model.config.id2label.copy(), k):
                print(f"{item['probability']:6.2%}  {item['label']}")
    '''), _md("""
        <details>
        <summary>시도한 뒤 결과 해설 보기</summary>

        k를 늘리면 더 많은 후보가 보입니다. 사진이나 모델은 바뀌지 않았으므로 같은 항목의 확률과 순서는 같습니다.
        1위만 보면 간단하지만, 여러 후보를 보면 모델이 비슷하게 보는 이름도 확인할 수 있습니다.
        상위 5개 확률의 합은 전체 1,000개 확률의 합과 다릅니다. 보통 1보다 작습니다.
        원래 모델의 이름과 실습 정답 이름이 다를 수 있으므로 이 한 장의 결과를 5종 분류 정확도로 해석하지 않습니다.

        </details>

        ## 6. 완성한 두 함수로 실제 결과 확인

        선택한 사진과 상위 5개 예측을 함께 봅니다. 예상과 다르면 입력 사진, 예측 이름,
        두 번째·세 번째 후보를 차례로 살펴보세요. 작은 사진을 확대해 넣었다는 점도 생각해 봅니다.
    """), original[10], original[11], original[12]]
    result[-2]["source"] = result[-2]["source"].replace("## 5. 실행 기록 확인", "## 7. 실행 기록 확인", 1)
    for cell in result:
        if cell["cell_type"] == "code":
            assert len(cell["source"].splitlines()) <= 40
            cell["outputs"] = []
            cell["execution_count"] = None
    return result
