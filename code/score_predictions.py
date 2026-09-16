# -*- coding: utf-8 -*-
"""本项目四任务评估：关键词、六经归属、十选一答案、解释参考一致度。"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_DIR = PROJECT_ROOT / "models" / "bert-base-chinese"

PRIVATE_TEST_LABELS = DATA_DIR / "test_labels_private.json"
DEFAULT_REFERENCE = PRIVATE_TEST_LABELS
DEFAULT_KEYWORD_FILE = PRIVATE_TEST_LABELS
DEFAULT_PREDICTIONS = RESULTS_DIR / "validation_results.json"
DEFAULT_MODELS = (
    "gpt-5.6-sol",
    "glm-5.2",
    "qwen3-8b",
    "qwen3-32b",
    "deepseek-v4-pro",
)
DEFAULT_OUTPUT = RESULTS_DIR / "test_evaluation_report.json"
DEFAULT_LOCAL_BERT_DIR = MODEL_DIR

WEIGHTS = {
    "question_keywords": 0.20,
    "six_meridian": 0.30,
    "answer": 0.40,
    "explanation": 0.10,
}
QUESTION_ROUGE_WEIGHT = 0.35
QUESTION_BERTSCORE_WEIGHT = 0.65
EXPLANATION_ROUGE_WEIGHT = 0.30
EXPLANATION_BERTSCORE_WEIGHT = 0.70

MERIDIAN_LABELS = tuple("ABCDEF")
QUESTION_LABELS = tuple("ABCDEFGHIJ")
EXPERT_KEYWORDS_FIELD = "question_keywords"


def load_json_list(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON格式错误：{path}；第{exc.lineno}行第{exc.colno}列：{exc.msg}"
        ) from exc
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError(f"{path} 必须是由 JSON 对象组成的数组")
    return data


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def text_tokens(value: Any) -> list[str]:
    """中文按字、连续拉丁字母和数字按词切分，忽略空白及标点。"""
    text = clean_text(value).lower()
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+|\d+", text)


def lcs_length(first: list[str], second: list[str]) -> int:
    """使用一维动态规划计算最长公共子序列。"""
    if len(first) > len(second):
        first, second = second, first
    previous = [0] * (len(first) + 1)
    for token_second in second:
        current = [0]
        for index, token_first in enumerate(first, start=1):
            if token_first == token_second:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_scores(candidate: Any, reference: Any) -> dict[str, float]:
    candidate_tokens = text_tokens(candidate)
    reference_tokens = text_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(candidate_tokens, reference_tokens)
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (
        precision + recall
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def normalize_keywords(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("keywords", [])
    if isinstance(value, str):
        parts = re.split(r"[;；、,，\n]+", value)
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = []

    normalized = {clean_text(item) for item in parts if clean_text(item)}
    return sorted(normalized)


def keywords_to_text(value: Any) -> str:
    return "；".join(normalize_keywords(value))


def get_gold_keywords(reference_item: dict) -> Optional[list[str]]:
    if EXPERT_KEYWORDS_FIELD not in reference_item:
        return None
    return normalize_keywords(reference_item[EXPERT_KEYWORDS_FIELD])


def merge_expert_keywords(
    reference_rows: list[dict],
    keyword_rows: list[dict],
    *,
    keyword_source: str,
) -> list[dict]:
    """按 question_id 将 merged 文件中的专家关键词合并到59题测试集。"""
    keyword_index = index_by_question_id(
        keyword_rows,
        source_name=keyword_source,
    )
    merged: list[dict] = []
    missing: list[str] = []
    empty: list[str] = []
    for reference in reference_rows:
        question_id = clean_text(reference.get("question_id", ""))
        keyword_record = keyword_index.get(question_id)
        if keyword_record is None:
            missing.append(question_id)
            continue
        if EXPERT_KEYWORDS_FIELD not in keyword_record:
            missing.append(question_id)
            continue
        keywords = normalize_keywords(
            keyword_record.get(EXPERT_KEYWORDS_FIELD)
        )
        if not keywords:
            empty.append(question_id)
            continue
        item = dict(reference)
        item[EXPERT_KEYWORDS_FIELD] = keywords
        merged.append(item)

    if missing or empty:
        messages = []
        if missing:
            messages.append(
                f"{len(missing)}题缺少字段，例如：{', '.join(missing[:5])}"
            )
        if empty:
            messages.append(
                f"{len(empty)}题关键词为空，例如：{', '.join(empty[:5])}"
            )
        raise ValueError(
            f'{keyword_source} 的 "{EXPERT_KEYWORDS_FIELD}" 标注不完整：'
            + "；".join(messages)
        )
    return merged


def extract_standalone_letters(value: Any) -> list[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    return re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", text)


def parse_meridian_answer(value: Any) -> tuple[set[str], set[str]]:
    letters = set(extract_standalone_letters(value))
    valid = letters.intersection(MERIDIAN_LABELS)
    invalid = letters.difference(MERIDIAN_LABELS)
    return valid, invalid


def parse_single_answer(value: Any) -> Optional[str]:
    letters = set(extract_standalone_letters(value))
    if len(letters) != 1:
        return None
    answer = next(iter(letters))
    return answer if answer in QUESTION_LABELS else None


def set_f1(
    prediction: set[str],
    gold: set[str],
    *,
    invalid_prediction: bool = False,
) -> float:
    if invalid_prediction:
        return 0.0
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    return 2 * len(prediction & gold) / (len(prediction) + len(gold))


def safe_mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


class BertScoreCalculator:
    """缓存同一个中文BERTScore模型，并批量计算F1。"""

    def __init__(
        self,
        *,
        model_type: str = "bert-base-chinese",
        idf: bool = False,
        rescale_with_baseline: bool = False,
        batch_size: int = 16,
        device: Optional[str] = None,
        num_layers: Optional[int] = None,
    ) -> None:
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 BERTScore 依赖，请安装：pip install bert-score"
            ) from exc

        kwargs: dict[str, Any] = {
            "lang": "zh",
            "model_type": model_type,
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
            "batch_size": batch_size,
        }
        if device:
            kwargs["device"] = device
        if num_layers is not None:
            kwargs["num_layers"] = num_layers
        self.scorer = BERTScorer(**kwargs)
        self.configuration = {
            "model_type": model_type,
            "lang": "zh",
            "idf": idf,
            "rescale_with_baseline": rescale_with_baseline,
            "batch_size": batch_size,
            "device": device or "auto",
            "num_layers": num_layers,
        }

    def score(self, candidates: list[str], references: list[str]) -> list[float]:
        if len(candidates) != len(references):
            raise ValueError("BERTScore候选和参考文本数量不一致")

        scores = [0.0] * len(candidates)
        valid_indices = [
            index
            for index, (candidate, reference) in enumerate(
                zip(candidates, references)
            )
            if clean_text(candidate) and clean_text(reference)
        ]
        if not valid_indices:
            return scores

        valid_candidates = [candidates[index] for index in valid_indices]
        valid_references = [references[index] for index in valid_indices]
        _, _, f1_tensor = self.scorer.score(
            valid_candidates,
            valid_references,
        )
        for index, value in zip(valid_indices, f1_tensor.tolist()):
            # 所有子指标统一限制到[0,1]。
            scores[index] = max(0.0, min(1.0, float(value)))
        return scores


def index_by_question_id(
    rows: list[dict],
    *,
    source_name: str,
) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        question_id = clean_text(row.get("question_id", ""))
        if not question_id:
            raise ValueError(f"{source_name} 中存在缺少 question_id 的记录")
        if question_id in indexed:
            raise ValueError(f"{source_name} 中 question_id 重复：{question_id}")
        indexed[question_id] = row
    return indexed


def calculate_macro_f1(
    item_details: list[dict],
) -> tuple[float, dict[str, dict[str, float]]]:
    per_label: dict[str, dict[str, float]] = {}
    label_scores: list[float] = []
    for label in MERIDIAN_LABELS:
        tp = fp = fn = support = 0
        for item in item_details:
            gold = set(item["_gold_meridian_set"])
            prediction = set(item["_pred_meridian_set"])
            if item["meridian_invalid_letters"]:
                prediction = set()
            gold_positive = label in gold
            pred_positive = label in prediction
            support += int(gold_positive)
            tp += int(gold_positive and pred_positive)
            fp += int(not gold_positive and pred_positive)
            fn += int(gold_positive and not pred_positive)
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else 2 * tp / denominator
        label_scores.append(f1)
        per_label[label] = {
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return safe_mean(label_scores), per_label


def evaluate_model(
    *,
    reference_rows: list[dict],
    prediction_rows: list[dict],
    prediction_path: Path,
    bertscore: BertScoreCalculator,
) -> dict:
    predictions = index_by_question_id(
        prediction_rows,
        source_name=str(prediction_path),
    )

    gold_keyword_texts: list[str] = []
    predicted_keyword_texts: list[str] = []
    gold_explanations: list[str] = []
    predicted_explanations: list[str] = []
    item_details: list[dict] = []
    missing_ids: list[str] = []

    for reference in reference_rows:
        question_id = clean_text(reference.get("question_id", ""))
        prediction = predictions.get(question_id)
        if prediction is None:
            prediction = {}
            missing_ids.append(question_id)

        gold_keywords = get_gold_keywords(reference)
        if gold_keywords is None:
            raise ValueError(
                f'{question_id} 缺少专家关键词字段 "{EXPERT_KEYWORDS_FIELD}"'
            )
        predicted_keywords = normalize_keywords(
            prediction.get("keyword_result", {})
        )
        gold_keyword_text = keywords_to_text(gold_keywords)
        predicted_keyword_text = keywords_to_text(predicted_keywords)

        gold_meridian, gold_meridian_invalid = parse_meridian_answer(
            reference.get("six_meridian_answer", "")
        )
        if gold_meridian_invalid or not gold_meridian:
            raise ValueError(
                f"{question_id} 的标准六经答案非法："
                f"{reference.get('six_meridian_answer')!r}"
            )
        predicted_meridian, predicted_meridian_invalid = parse_meridian_answer(
            prediction.get("six_meridian_result", {}).get("answer", "")
            if isinstance(prediction.get("six_meridian_result"), dict)
            else ""
        )
        meridian_f1 = set_f1(
            predicted_meridian,
            gold_meridian,
            invalid_prediction=bool(predicted_meridian_invalid),
        )
        meridian_exact = (
            not predicted_meridian_invalid
            and predicted_meridian == gold_meridian
        )

        gold_answer = parse_single_answer(reference.get("question_answer", ""))
        if gold_answer is None:
            raise ValueError(
                f"{question_id} 的标准十选一答案非法："
                f"{reference.get('question_answer')!r}"
            )
        predicted_answer = parse_single_answer(
            prediction.get("question_result", {}).get("answer", "")
            if isinstance(prediction.get("question_result"), dict)
            else ""
        )
        answer_correct = predicted_answer == gold_answer

        predicted_explanation = clean_text(
            prediction.get("explanation_result", {}).get("explanation", "")
            if isinstance(prediction.get("explanation_result"), dict)
            else ""
        )
        gold_explanation = clean_text(reference.get("explanation", ""))

        keyword_rouge = rouge_l_scores(
            predicted_keyword_text,
            gold_keyword_text,
        )["recall"]
        explanation_rouge = rouge_l_scores(
            predicted_explanation,
            gold_explanation,
        )["f1"]

        gold_keyword_texts.append(gold_keyword_text)
        predicted_keyword_texts.append(predicted_keyword_text)
        gold_explanations.append(gold_explanation)
        predicted_explanations.append(predicted_explanation)
        item_details.append(
            {
                "question_id": question_id,
                "prediction_present": bool(prediction),
                "gold_keywords": gold_keywords,
                "predicted_keywords": predicted_keywords,
                "question_rouge_l_recall": keyword_rouge,
                "question_bertscore_f1": None,
                "question_score": None,
                "gold_meridian_answer": sorted(gold_meridian),
                "predicted_meridian_answer": sorted(predicted_meridian),
                "meridian_invalid_letters": sorted(
                    predicted_meridian_invalid
                ),
                "meridian_set_f1": meridian_f1,
                "meridian_exact_match": meridian_exact,
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "answer_exact_match": answer_correct,
                "explanation_rouge_l_f1": explanation_rouge,
                "explanation_bertscore_f1": None,
                "explanation_score": None,
                # 仅供内部计算Macro-F1，写报告前会删除。
                "_gold_meridian_set": sorted(gold_meridian),
                "_pred_meridian_set": sorted(predicted_meridian),
            }
        )

    keyword_bert_scores = bertscore.score(
        predicted_keyword_texts,
        gold_keyword_texts,
    )
    explanation_bert_scores = bertscore.score(
        predicted_explanations,
        gold_explanations,
    )

    for item, keyword_bert, explanation_bert in zip(
        item_details,
        keyword_bert_scores,
        explanation_bert_scores,
    ):
        item["question_bertscore_f1"] = keyword_bert
        item["question_score"] = (
            QUESTION_ROUGE_WEIGHT * item["question_rouge_l_recall"]
            + QUESTION_BERTSCORE_WEIGHT * keyword_bert
        )
        item["explanation_bertscore_f1"] = explanation_bert
        item["explanation_score"] = (
            EXPLANATION_ROUGE_WEIGHT * item["explanation_rouge_l_f1"]
            + EXPLANATION_BERTSCORE_WEIGHT * explanation_bert
        )

    question_score = safe_mean(item["question_score"] for item in item_details)
    meridian_score = safe_mean(
        item["meridian_set_f1"] for item in item_details
    )
    answer_score = safe_mean(
        float(item["answer_exact_match"]) for item in item_details
    )
    explanation_score = safe_mean(
        item["explanation_score"] for item in item_details
    )
    total_score = (
        WEIGHTS["question_keywords"] * question_score
        + WEIGHTS["six_meridian"] * meridian_score
        + WEIGHTS["answer"] * answer_score
        + WEIGHTS["explanation"] * explanation_score
    )

    macro_f1, per_label_f1 = calculate_macro_f1(item_details)
    single_items = [
        item for item in item_details if len(item["_gold_meridian_set"]) == 1
    ]
    multi_items = [
        item for item in item_details if len(item["_gold_meridian_set"]) > 1
    ]
    for item in item_details:
        item.pop("_gold_meridian_set", None)
        item.pop("_pred_meridian_set", None)

    model_name = next(
        (
            clean_text(row.get("model", ""))
            for row in prediction_rows
            if clean_text(row.get("model", ""))
        ),
        prediction_path.stem,
    )
    return {
        "model": model_name,
        "prediction_file": str(prediction_path),
        "sample_count": len(reference_rows),
        "prediction_count": len(prediction_rows),
        "matched_count": len(reference_rows) - len(missing_ids),
        "missing_question_ids": missing_ids,
        "scores": {
            "question_keyword_completeness": question_score,
            "six_meridian_set_f1": meridian_score,
            "answer_exact_match_accuracy": answer_score,
            "explanation_reference_consistency": explanation_score,
            "total": total_score,
            "total_100": 100 * total_score,
        },
        "auxiliary_metrics": {
            "six_meridian_exact_match_ratio": safe_mean(
                float(item["meridian_exact_match"]) for item in item_details
            ),
            "six_meridian_macro_f1": macro_f1,
            "six_meridian_per_label": per_label_f1,
            "single_meridian_set_f1": safe_mean(
                item["meridian_set_f1"] for item in single_items
            ),
            "single_meridian_count": len(single_items),
            "multi_meridian_set_f1": safe_mean(
                item["meridian_set_f1"] for item in multi_items
            ),
            "multi_meridian_count": len(multi_items),
            "invalid_meridian_output_count": sum(
                bool(item["meridian_invalid_letters"])
                for item in item_details
            ),
            "invalid_single_answer_count": sum(
                item["predicted_answer"] is None for item in item_details
            ),
        },
        "per_item": item_details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估本项目多个模型的四阶段测试集结果"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=f"包含59题标准答案的测试集，默认：{DEFAULT_REFERENCE}",
    )
    parser.add_argument(
        "--keyword-file",
        type=Path,
        default=DEFAULT_KEYWORD_FILE,
        help=(
            f'包含 question_id 和 "{EXPERT_KEYWORDS_FIELD}" 的专家关键词文件，'
            f"默认：{DEFAULT_KEYWORD_FILE}"
        ),
    )
    parser.add_argument(
        "--predictions",
        nargs="+",
        type=Path,
        default=[DEFAULT_PREDICTIONS],
        help=(
            "06统一管道生成的结果JSON；支持传入多个文件，"
            f"默认：{DEFAULT_PREDICTIONS}"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="需要评估的模型名称；默认评估06统一管道中的五个模型",
    )
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bertscore-model",
        default=str(DEFAULT_LOCAL_BERT_DIR),
        help=f"本地BERTScore模型目录，默认：{DEFAULT_LOCAL_BERT_DIR}",
    )
    parser.add_argument(
        "--bertscore-num-layers",
        type=int,
        default=8,
        help="BERTScore使用的隐藏层，默认：8",
    )
    parser.add_argument("--bertscore-idf", action="store_true")
    parser.add_argument("--bertscore-rescale", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="例如 cpu、cuda、cuda:0")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size 必须大于0")
    missing_files = [path for path in args.predictions if not path.exists()]
    if missing_files:
        raise SystemExit(f"模型结果文件不存在：{missing_files}")

    base_reference_rows = load_json_list(args.reference)
    index_by_question_id(base_reference_rows, source_name=str(args.reference))
    try:
        reference_rows = merge_expert_keywords(
            base_reference_rows,
            load_json_list(args.keyword_file),
            keyword_source=str(args.keyword_file),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    bertscore_model_path = Path(args.bertscore_model)
    if not bertscore_model_path.is_dir():
        raise SystemExit(f"本地BERTScore模型目录不存在：{bertscore_model_path}")
    required_model_files = ("config.json", "tokenizer_config.json", "vocab.txt")
    missing_model_files = [
        name
        for name in required_model_files
        if not (bertscore_model_path / name).is_file()
    ]
    has_model_weights = any(
        (bertscore_model_path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if missing_model_files or not has_model_weights:
        raise SystemExit(
            f"本地BERTScore模型不完整：{bertscore_model_path}；"
            f"缺失文件={missing_model_files}；存在权重={has_model_weights}"
        )

    bertscore = BertScoreCalculator(
        model_type=str(bertscore_model_path.resolve()),
        idf=args.bertscore_idf,
        rescale_with_baseline=args.bertscore_rescale,
        batch_size=args.batch_size,
        device=args.device,
        num_layers=args.bertscore_num_layers,
    )
    reference_ids = {
        clean_text(row.get("question_id", "")) for row in reference_rows
    }
    rows_by_model: dict[str, dict[str, dict]] = {
        model: {} for model in args.models
    }
    skipped_non_reference: dict[str, int] = {
        model: 0 for model in args.models
    }
    for prediction_path in args.predictions:
        for row in load_json_list(prediction_path):
            model_name = clean_text(row.get("model", ""))
            # 06统一管道的失败/中断记录可能只完成部分Prompt；完整评估中按缺失计0。
            status = clean_text(row.get("status", ""))
            if model_name in rows_by_model and status in ("", "success"):
                question_id = clean_text(row.get("question_id", ""))
                if question_id in reference_ids:
                    # 同模型同题存在历史重复记录时，以文件中最后一条成功记录为准。
                    rows_by_model[model_name][question_id] = row
                elif question_id:
                    skipped_non_reference[model_name] += 1

    model_reports = []
    for model_name in args.models:
        prediction_rows = list(rows_by_model[model_name].values())
        print(
            f"[评估] {model_name} | {len(prediction_rows)} 条 | "
            f"跳过非测试集 {skipped_non_reference[model_name]} 条 | "
            f"来源：{', '.join(str(path) for path in args.predictions)}"
        )
        report = evaluate_model(
                reference_rows=reference_rows,
                prediction_rows=prediction_rows,
                prediction_path=args.predictions[0],
                bertscore=bertscore,
            )
        # 空结果时 evaluate_model 无法从记录推断名称，使用指定模型名。
        report["model"] = model_name
        report["prediction_files"] = [str(path) for path in args.predictions]
        report["skipped_non_reference_count"] = skipped_non_reference[model_name]
        model_reports.append(report)

    ranking = sorted(
        (
            {
                "rank": 0,
                "model": report["model"],
                "total": report["scores"]["total"],
                "total_100": report["scores"]["total_100"],
            }
            for report in model_reports
        ),
        key=lambda row: row["total"],
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    report = {
        "evaluation_name": "中医六经四任务测试集评估",
        "reference_file": str(args.reference),
        "expert_keyword_file": str(args.keyword_file),
        "expert_keyword_field": EXPERT_KEYWORDS_FIELD,
        "formula": {
            "question_keywords": (
                "0.35 * ROUGE-L Recall + 0.65 * BERTScore F1"
            ),
            "six_meridian": "Example-based Set F1 / Dice F1",
            "answer": "Strict single-letter Exact Match Accuracy",
            "explanation": (
                "0.30 * ROUGE-L F1 + 0.70 * BERTScore F1"
            ),
            "total": (
                "0.20 * question_keywords + 0.30 * six_meridian + "
                "0.40 * answer + 0.10 * explanation"
            ),
        },
        "weights": WEIGHTS,
        "rouge_tokenization": (
            "Chinese character + contiguous Latin word/number; punctuation ignored"
        ),
        "bertscore_configuration": bertscore.configuration,
        "ranking": ranking,
        "models": model_reports,
    }
    write_json(args.output, report)

    print(f"\n[完成] 报告已保存：{args.output}")
    for row in ranking:
        print(
            f"{row['rank']}. {row['model']}: "
            f"{row['total_100']:.2f}"
        )


if __name__ == "__main__":
    main()


