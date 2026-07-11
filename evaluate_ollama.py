"""Run repeatable text and vision A/B checks against local Ollama models."""

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


BARE_PLACEHOLDER_RE = re.compile(r"^\[(?:图片|视频|语音|文件|表情|动画表情)\]$")


def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def model_capabilities(api_root: str, model: str) -> list[str]:
    try:
        data = post_json(f"{api_root}/api/show", {"model": model}, timeout=30)
        return list(data.get("capabilities", []))
    except Exception:
        return []


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["qwen3.5:9b", "myclone:latest", "myclone-vl:latest"])
    parser.add_argument("--require-models", nargs="+", default=None,
                        help="Models whose automatic gates control the exit code. "
                             "Defaults to the last model in --models; earlier "
                             "models are treated as A/B baselines.")
    parser.add_argument("--cases", default=str(Path(__file__).with_name("eval_cases.json")))
    parser.add_argument("--image", default="",
                        help="Image used by cases with use_image=true")
    parser.add_argument("--api-root", default="http://127.0.0.1:11434")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output", default="eval-report.json")
    args = parser.parse_args()
    required_models = set(args.require_models or [args.models[-1]])
    unknown_required = required_models - set(args.models)
    if unknown_required:
        parser.error(f"--require-models not present in --models: {sorted(unknown_required)}")

    cases = json.loads(Path(args.cases).read_text("utf-8"))
    image_b64 = ""
    if args.image:
        image_b64 = base64.b64encode(Path(args.image).read_bytes()).decode("ascii")

    report = {
        "api_root": args.api_root,
        "temperature": args.temperature,
        "required_models": sorted(required_models),
        "models": {},
    }
    overall_pass = True
    system = (
        "准确理解对话上下文，再用自然、简短但意思完整的聊天语气回复。"
        "信息不足时可以澄清，不要输出媒体占位符。"
    )

    for model in args.models:
        capabilities = model_capabilities(args.api_root, model)
        model_report = {"capabilities": capabilities, "cases": []}
        report["models"][model] = model_report
        print(f"\n=== {model} | capabilities={','.join(capabilities) or 'unknown'} ===")

        for case in cases:
            if case.get("use_image") and not image_b64:
                model_report["cases"].append({
                    "id": case["id"], "goal": case.get("goal", ""), "skipped": True,
                    "reason": "--image not provided",
                })
                continue

            messages = [{"role": "system", "content": system}]
            messages.extend(json.loads(json.dumps(case["messages"], ensure_ascii=False)))
            if case.get("use_image"):
                messages[-1]["images"] = [image_b64]

            started = time.perf_counter()
            result = {
                "id": case["id"],
                "goal": case.get("goal", ""),
                "uses_image": bool(case.get("use_image")),
            }
            try:
                data = post_json(f"{args.api_root}/api/chat", {
                    "model": model,
                    "messages": messages,
                    "think": False,
                    "stream": False,
                    "options": {
                        "temperature": args.temperature,
                        "top_p": 0.9,
                        "repeat_penalty": 1.05,
                        "num_predict": 160,
                    },
                })
                reply = (data.get("message", {}).get("content") or "").strip()
                checks = {
                    "nonempty": bool(reply),
                    "not_bare_media_placeholder": not bool(BARE_PLACEHOLDER_RE.fullmatch(reply)),
                    "vision_capability_present": (not case.get("use_image") or "vision" in capabilities),
                }
                result.update({
                    "reply": reply,
                    "chars": len(reply),
                    "seconds": round(time.perf_counter() - started, 2),
                    "checks": checks,
                    "gate_pass": all(checks.values()),
                })
                print(f"[{case['id']}] {reply}")
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                result.update({
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.perf_counter() - started, 2),
                    "gate_pass": False,
                })
                print(f"[{case['id']}] ERROR: {exc}")
            if model in required_models:
                overall_pass = overall_pass and result.get("gate_pass", True)
            model_report["cases"].append(result)

    report["automatic_gates_passed"] = overall_pass
    output = Path(args.output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"\nReport: {output.resolve()}")
    print("语义目标仍需人工阅读 A/B 回复；自动门禁只负责空回复、占位符和视觉能力。")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
