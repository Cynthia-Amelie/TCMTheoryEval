# -*- coding: utf-8 -*-
"""统一运行五个模型的四阶段评测流程，并保存逐题模型输出。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from dotenv import load_dotenv

try:
    import dashscope
    from dashscope import Generation
except ImportError:
    dashscope = None
    Generation = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Always load configuration from the repository root, regardless of the
# directory from which this script is launched.
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_INPUT = DATA_DIR / "test.json"
DEFAULT_OUTPUT_DIR = RESULTS_DIR
DEFAULT_RESULTS = "validation_results.json"

DASHSCOPE_API_KEY = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
GPT_API_KEY = (
    os.getenv("OPENAI_HK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
).strip()
GPT_API_URL = os.getenv(
    "OPENAI_HK_API_URL", "https://api.openai-hk.com/v1/chat/completions"
).strip()
STANDARD_DASHSCOPE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/api/v1"
).strip()
MAAS_DASHSCOPE_URL = os.getenv(
    "MAAS_DASHSCOPE_BASE_URL",
    "https://llm-u1x9o06s0wexttqg.cn-beijing.maas.aliyuncs.com/api/v1",
).strip()

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 3000
SEED = None
INTERNET_ACCESS = False
RETRIEVAL_ENABLED = False
TOOLS_ENABLED = False
GPT_REASONING_EFFORT = "none"
DASHSCOPE_ENABLE_THINKING = False
RUN_SIGNATURE = "non-thinking_temp0_top-p1_max-tokens3000_no-seed_no-web-tools_v1"
MAX_API_RETRIES = 5
MAX_FORMAT_RETRIES = 3
RETRY_WAIT_SECONDS = 10
CALL_INTERVAL_SECONDS = 0.5

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gpt-5.6-sol": {
        "provider": "OpenAI-HK compatible endpoint",
        "api_style": "requests chat/completions (testGPT.py)",
        "base_url": GPT_API_URL,
        "thinking_mode": "关闭（reasoning_effort=none）",
        "thinking_disabled_explicitly": True,
        "seed": None,
        "seed_status": "为保持跨模型一致，统一不发送 seed",
    },
    "glm-5.2": {
        "provider": "DashScope MaaS",
        "api_style": "DashScope Generation messages (testGLM.py)",
        "base_url": MAAS_DASHSCOPE_URL,
        "thinking_mode": "关闭（enable_thinking=False）",
        "thinking_disabled_explicitly": True,
        "seed": None,
        "seed_status": "为保持跨模型一致，统一不发送 seed",
    },
    "qwen3-8b": {
        "provider": "DashScope",
        "api_style": "DashScope Generation prompt (testQWEN3-8B.py)",
        "base_url": STANDARD_DASHSCOPE_URL,
        "thinking_mode": "关闭（enable_thinking=False）",
        "thinking_disabled_explicitly": True,
        "seed": None,
        "seed_status": "为保持跨模型一致，统一不发送 seed",
    },
    "qwen3-32b": {
        "provider": "DashScope",
        "api_style": "DashScope Generation messages (testQWEN3-32B.py)",
        "base_url": STANDARD_DASHSCOPE_URL,
        "thinking_mode": "关闭（enable_thinking=False）",
        "thinking_disabled_explicitly": True,
        "seed": None,
        "seed_status": "为保持跨模型一致，统一不发送 seed",
    },
    "deepseek-v4-pro": {
        "provider": "DashScope",
        "api_style": "DashScope Generation messages",
        "base_url": STANDARD_DASHSCOPE_URL,
        "thinking_mode": "关闭（enable_thinking=False）",
        "thinking_disabled_explicitly": True,
        "seed": None,
        "seed_status": "为保持跨模型一致，统一不发送 seed",
    },
}

SYSTEM_PROMPT = (
    "你是一名严谨的中医六经辨证专家。只依据题目和当前提示中的信息作答。"
    "必须严格输出合法 JSON，不要输出 Markdown 代码块或 JSON 之外的文字。"
)


def audit_benchmark_configuration(models: List[str]) -> None:
    """正式运行前强制核验跨模型实验参数，发现偏差立即停止。"""
    problems: List[str] = []
    if TEMPERATURE != 0.0:
        problems.append(f"temperature={TEMPERATURE}，要求为0")
    if TOP_P != 1.0:
        problems.append(f"top_p={TOP_P}，要求为1")
    if MAX_TOKENS != 3000:
        problems.append(f"max_tokens={MAX_TOKENS}，要求为3000")
    if SEED is not None:
        problems.append("seed必须统一不发送")
    if INTERNET_ACCESS or RETRIEVAL_ENABLED or TOOLS_ENABLED:
        problems.append("联网、检索和工具必须全部关闭")
    if GPT_REASONING_EFFORT != "none":
        problems.append("GPT reasoning_effort必须为none")
    if DASHSCOPE_ENABLE_THINKING is not False:
        problems.append("DashScope enable_thinking必须为False")
    for model in models:
        profile = MODEL_CONFIGS[model]
        if not profile.get("thinking_disabled_explicitly", False):
            problems.append(f"{model} 未显式关闭思考模式")
        if profile.get("seed") is not None:
            problems.append(f"{model} 配置了seed")
    if problems:
        raise SystemExit("实验配置审计失败：\n- " + "\n- ".join(problems))
    print(
        "[配置审计通过] temperature=0 | top_p=1 | max_tokens=3000 | "
        "seed=未发送 | 联网=关闭 | 检索=关闭 | 工具=关闭 | 思考=全部关闭"
    )


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ModelClient(ABC):
    def __init__(self, model: str) -> None:
        self.model = model
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class FatalModelError(RuntimeError):
    """鉴权或权限错误；继续请求同一模型不会自行恢复。"""


class GPTClient(ModelClient):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        if not GPT_API_KEY:
            raise RuntimeError("未配置 OPENAI_HK_API_KEY 或 OPENAI_API_KEY")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
            "reasoning_effort": GPT_REASONING_EFFORT,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GPT_API_KEY}",
        }
        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                response = requests.post(
                    GPT_API_URL,
                    headers=headers,
                    data=json.dumps(payload).encode("utf-8"),
                    timeout=180,
                )
                body = response.json()
                if response.status_code != 200:
                    error_body = body.get("error", body)
                    message = f"HTTP {response.status_code}: {error_body}"
                    error_code = str(_value(error_body, "code", "")).lower()
                    error_message = str(_value(error_body, "message", "")).lower()
                    if (
                        response.status_code in (401, 403)
                        or error_code == "model_not_found"
                        or "no available channel" in error_message
                        or "无可用渠道" in error_message
                    ):
                        raise FatalModelError(message)
                    raise RuntimeError(message)
                text = body["choices"][0]["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("API 返回空文本")
                return text.strip()
            except FatalModelError as exc:
                print(f"[{self.model} 不可重试错误] {exc}")
                raise
            except Exception as exc:
                print(f"[{self.model} API 重试 {attempt}/{MAX_API_RETRIES}] {exc}")
                if attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS)
        raise RuntimeError(f"{self.model} API 连续失败 {MAX_API_RETRIES} 次")


class DashScopeClient(ModelClient):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        if dashscope is None or Generation is None:
            raise RuntimeError("缺少 dashscope，请先安装 dashscope")
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        config = MODEL_CONFIGS[self.model]
        dashscope.base_http_api_url = config["base_url"]
        args: Dict[str, Any] = {
            "api_key": DASHSCOPE_API_KEY,
            "model": self.model,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        }
        if self.model == "qwen3-8b":
            args["prompt"] = f"系统要求：\n{system_prompt}\n\n用户任务：\n{user_prompt}"
        else:
            args["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            args["result_format"] = "message"
        # Qwen3、GLM 与 DeepSeek-V4-Pro 均显式关闭思考模式。
        args["enable_thinking"] = DASHSCOPE_ENABLE_THINKING

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                response = Generation.call(**args)
                if response.status_code != 200:
                    raise RuntimeError(
                        f"{getattr(response, 'code', '')}: {getattr(response, 'message', '')}"
                    )
                output = response.output
                if self.model == "qwen3-8b":
                    text = _value(output, "text")
                    if not text:
                        try:
                            text = output.choices[0].message.content
                        except (AttributeError, IndexError, KeyError, TypeError):
                            text = None
                else:
                    choice = output.choices[0]
                    message = choice.message
                    text = message.content
                    if isinstance(text, list):
                        text = "".join(
                            str(_value(block, "text", "")) for block in text
                        )
                if not isinstance(text, str) or not text.strip():
                    choice = output.choices[0] if _value(output, "choices") else None
                    message = _value(choice, "message")
                    reasoning = _value(message, "reasoning_content", "")
                    raise ValueError(
                        "API 返回空文本；"
                        f"finish_reason={_value(choice, 'finish_reason')!r}，"
                        f"content_type={type(_value(message, 'content')).__name__}，"
                        f"reasoning_chars={len(str(reasoning or ''))}"
                    )
                return text.strip()
            except Exception as exc:
                print(f"[{self.model} API 重试 {attempt}/{MAX_API_RETRIES}] {exc}")
                if attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS)
        raise RuntimeError(f"{self.model} API 连续失败 {MAX_API_RETRIES} 次")


def keyword_prompt(item: dict) -> str:
    return (
        "任务一：从问题中提取 2—6 个有助于判断答案的关键词或关键短语。"
        "保持原文措辞，不回答选择题。\n"
        '输出格式：{"keywords":["关键词1","关键词2"]}\n\n'
        f"问题：{item.get('question', '')}"
    )


def meridian_prompt(item: dict, keywords: dict) -> str:
    return (
        "任务二：完成六经归属多选题，可选 1—6 项，只返回选项字母；多项用分号连接。\n"
        '输出格式：{"answer":"A;B","reason":"一句简短理由"}\n\n'
        f"问题：{item.get('question', '')}\n"
        f"六经归属选项：\n{item.get('six_meridian_options', '')}\n"
        f"任务一回答：{json.dumps(keywords, ensure_ascii=False)}"
    )


def question_prompt(item: dict, keywords: dict, meridian: dict) -> str:
    return (
        "任务三：完成问题选择题，只能选择一个 A—J 选项。\n"
        '输出格式：{"answer":"A","reason":"一句简短理由"}\n\n'
        f"问题：{item.get('question', '')}\n"
        f"问题选项：\n{item.get('question_options', '')}\n"
        f"任务一回答：{json.dumps(keywords, ensure_ascii=False)}\n"
        f"任务二回答：{json.dumps(meridian, ensure_ascii=False)}"
    )


def explanation_prompt(item: dict, keywords: dict, meridian: dict, answer: dict) -> str:
    return (
        "任务四：结合前三次回答，用 1—3 句话简短解释最终选择。\n"
        '输出格式：{"explanation":"简短解释"}\n\n'
        f"问题：{item.get('question', '')}\n"
        f"六经归属选项：\n{item.get('six_meridian_options', '')}\n"
        f"问题选项：\n{item.get('question_options', '')}\n"
        f"任务一回答：{json.dumps(keywords, ensure_ascii=False)}\n"
        f"任务二回答：{json.dumps(meridian, ensure_ascii=False)}\n"
        f"任务三回答：{json.dumps(answer, ensure_ascii=False)}"
    )


def parse_json_response(text: str) -> Optional[dict]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S | re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    candidates = [cleaned]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


class NestedSolver:
    def __init__(self, client: ModelClient) -> None:
        self.client = client

    def ask(self, prompt: str, key: str) -> dict:
        current = prompt
        for attempt in range(1, MAX_FORMAT_RETRIES + 1):
            generated_text = self.client.generate(SYSTEM_PROMPT, current)
            parsed = parse_json_response(generated_text)
            if parsed is not None and key in parsed:
                return parsed
            current = prompt + f'\n\n上次格式错误。只返回包含字段 "{key}" 的合法 JSON。'
            print(f"[{self.client.model} 格式重试 {attempt}/{MAX_FORMAT_RETRIES}]")
        raise RuntimeError(f"模型未返回包含 {key} 的合法 JSON")

    def solve(
        self,
        item: dict,
        existing: Optional[dict] = None,
        checkpoint: Optional[Callable[[dict], None]] = None,
    ) -> dict:
        state = dict(existing or {})
        state.update({
            "question_id": item.get("question_id", ""),
            "question": item.get("question", ""),
            "model": self.client.model,
            "status": "in_progress",
            "run_signature": RUN_SIGNATURE,
        })

        def save_stage() -> None:
            if checkpoint:
                checkpoint(dict(state))

        keywords = state.get("keyword_result")
        if not isinstance(keywords, dict) or "keywords" not in keywords:
            keywords = self.ask(keyword_prompt(item), "keywords")
            state["keyword_result"] = keywords
            save_stage()
            time.sleep(CALL_INTERVAL_SECONDS)

        meridian = state.get("six_meridian_result")
        if not isinstance(meridian, dict) or "answer" not in meridian:
            meridian = self.ask(meridian_prompt(item, keywords), "answer")
            state["six_meridian_result"] = meridian
            save_stage()
            time.sleep(CALL_INTERVAL_SECONDS)

        answer = state.get("question_result")
        if not isinstance(answer, dict) or "answer" not in answer:
            answer = self.ask(question_prompt(item, keywords, meridian), "answer")
            state["question_result"] = answer
            save_stage()
            time.sleep(CALL_INTERVAL_SECONDS)

        explanation = state.get("explanation_result")
        if not isinstance(explanation, dict) or "explanation" not in explanation:
            explanation = self.ask(
                explanation_prompt(item, keywords, meridian, answer), "explanation"
            )
            state["explanation_result"] = explanation
            save_stage()
        state.update({
            "question_id": item.get("question_id", ""),
            "question": item.get("question", ""),
            "model": self.client.model,
            "status": "success",
            "keyword_result": keywords,
            "six_meridian_result": meridian,
            "question_result": answer,
            "explanation_result": explanation,
        })
        return state


def load_json_list(path: Path) -> List[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} 必须是 JSON 对象数组")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def upsert_result(rows: List[dict], new_row: dict) -> None:
    """按模型和问题编号原地更新记录，避免同一道题出现多条失败记录。"""
    key = (str(new_row.get("model", "")), str(new_row.get("question_id", "")))
    for index, row in enumerate(rows):
        if (str(row.get("model", "")), str(row.get("question_id", ""))) == key:
            rows[index] = new_row
            return
    rows.append(new_row)


def make_client(model: str) -> ModelClient:
    return GPTClient(model) if model == "gpt-5.6-sol" else DashScopeClient(model)


def main() -> None:
    parser = argparse.ArgumentParser(description="统一运行五个模型的四阶段评测")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CONFIGS), default=list(MODEL_CONFIGS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="只检查统一实验配置，不调用任何模型API",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit 必须大于 0")

    audit_benchmark_configuration(args.models)
    if args.audit_only:
        return

    results_path = args.output_dir / DEFAULT_RESULTS
    items = load_json_list(args.input)
    rows = load_json_list(results_path) if results_path.exists() else []

    for model in args.models:
        client = make_client(model)
        solver = NestedSolver(client)
        successful_ids = {
            str(row.get("question_id", "")) for row in rows
            if row.get("model") == model
            and row.get("status") == "success"
            and row.get("run_signature") == RUN_SIGNATURE
        }
        pending = [item for item in items if str(item.get("question_id", "")) not in successful_ids]
        if args.limit is not None:
            pending = pending[:args.limit]
        print(f"\n[{model}] 总题数 {len(items)}，已成功 {len(successful_ids)}，本次 {len(pending)}")

        for number, item in enumerate(pending, 1):
            question_id = str(item.get("question_id", ""))
            print(f"[{model} {number}/{len(pending)}] {question_id}")
            existing = next((
                row for row in rows
                if row.get("model") == model
                and str(row.get("question_id", "")) == question_id
                and row.get("run_signature") == RUN_SIGNATURE
            ), None)

            def save_checkpoint(checkpoint_row: dict) -> None:
                upsert_result(rows, checkpoint_row)
                write_json(results_path, rows)

            fatal_error = False
            try:
                row = solver.solve(
                    item,
                    existing=existing,
                    checkpoint=save_checkpoint,
                )
            except Exception as exc:
                row = dict(existing or {})
                # 若前面的阶段刚刚完成，它们已在 checkpoint 中，读取最新记录。
                latest = next((
                    saved for saved in rows
                    if saved.get("model") == model
                    and str(saved.get("question_id", "")) == question_id
                ), None)
                if latest:
                    row.update(latest)
                row.update({
                    "question_id": question_id,
                    "question": item.get("question", ""),
                    "model": model,
                    "status": "failed",
                    "run_signature": RUN_SIGNATURE,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                fatal_error = isinstance(exc, FatalModelError)
                print(f"[{model} 题目失败] {question_id}: {exc}")
            upsert_result(rows, row)
            write_json(results_path, rows)
            if fatal_error:
                print(f"[{model}] 遇到鉴权/权限错误，已保存进度并停止该模型。")
                break
            time.sleep(CALL_INTERVAL_SECONDS)

    print(f"\n结果：{results_path}")


if __name__ == "__main__":
    main()
