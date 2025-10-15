import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: str | Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def parse_context_pairs(pairs: List[str]) -> Dict[str, str]:
    context: Dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(
                f"Context assignment '{pair}' must be in key=value format."
            )
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Context assignment '{pair}' is missing a key.")
        context[key] = value
    return context


def get_venus_agent():
    from models.navigation.ui_venus_navi_agent import VenusNaviAgent

    return VenusNaviAgent


def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


@dataclass
class ModelConfig:
    model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.6
    max_tokens: int = 2048
    max_pixels: int = 12845056
    min_pixels: int = 3136
    max_model_len: int = 10000
    max_num_seqs: int = 5
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1

    def __str__(self) -> str:
        return f"ModelConfig({', '.join(f'{k}={v}' for k, v in asdict(self).items())})"


def main() -> None:
    parser = argparse.ArgumentParser(description="UI-Venus navigation runner")
    parser.add_argument("--model_path", type=str, default="/root/models/uivenus-7B")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_file", type=str, default="examples/trace/trace.json")
    parser.add_argument("--output_file", type=str, default="./saved_trace.json")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--max_pixels", type=int, default=12845056)
    parser.add_argument("--min_pixels", type=int, default=3136)
    parser.add_argument("--max_model_len", type=int, default=128000)
    parser.add_argument("--max_num_seqs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["instructed", "autonomous"],
        default="instructed",
        help="Inference mode. Autonomous mode enables experiment 1 prompt variants.",
    )
    parser.add_argument(
        "--variant_id",
        type=str,
        default=None,
        help="Prompt variant identifier (Experiment 1). Ignored in instructed mode.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional JSON file containing additional autonomous context.",
    )
    parser.add_argument(
        "--context_kv",
        action="append",
        default=[],
        help="Additional autonomous context entries in key=value format (repeatable).",
    )

    args = parser.parse_args()
    logger = setup_logger("UI-Venus")

    model_config = ModelConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_tokens=args.max_tokens,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        temperature=args.temperature,
        n=args.n,
    )
    logger.info("%s", model_config)

    data = read_json(args.input_file)
    if not isinstance(data, list):
        raise ValueError(f"Input file {args.input_file} must contain a list of traces.")

    is_autonomous = args.mode == "autonomous"

    context_overrides: Dict[str, str] = {}
    if args.context:
        context_data = read_json(args.context)
        if not isinstance(context_data, dict):
            parser.error("--context file must contain a JSON object with key-value pairs.")
        context_overrides.update({str(k): str(v) for k, v in context_data.items()})

    try:
        context_overrides.update(parse_context_pairs(args.context_kv))
    except ValueError as exc:
        parser.error(str(exc))

    variant_id: Optional[str] = None
    if is_autonomous:
        variant_id = args.variant_id or "default"
    elif args.variant_id:
        logger.warning("--variant_id is ignored in instructed mode.")

    try:
        VenusNaviAgent = get_venus_agent()
        venus_agent = VenusNaviAgent(
            model_config,
            logger,
            args.history_length,
            autonomous_variant_id=variant_id,
            autonomous_context=context_overrides if is_autonomous else None,
        )
        logger.info("VenusNaviAgent initialized successfully")
    except Exception as exc:
        logger.error("VenusNaviAgent initialization failed: %s", exc)
        raise

    traces_output: List[Dict[str, Any]] = []
    for trace_index, trace in enumerate(data):
        if not isinstance(trace, list):
            logger.warning("Trace %d is not a list; skipping.", trace_index)
            continue

        trace_inputs: List[Dict[str, Any]] = []
        finished_event: Optional[Dict[str, Any]] = None
        for item_index, item in enumerate(trace):
            if not isinstance(item, dict):
                logger.warning(
                    "Trace %d item %d is not a dict; skipping.", trace_index, item_index
                )
                continue

            task = item.get("task")
            image_path = item.get("image_path")

            if not image_path:
                logger.warning(
                    "Trace %d item %d missing 'image_path'; skipping step.",
                    trace_index,
                    item_index,
                )
                continue

            trace_inputs.append(
                {
                    "item_index": item_index,
                    "task": task,
                    "image_path": image_path,
                }
            )

            goal_argument = None if is_autonomous else task
            if is_autonomous:
                logger.info(
                    "Autonomous mode: skipping user task text %r for trace %d item %d "
                    "and relying on variant %s.",
                    task,
                    trace_index,
                    item_index,
                    variant_id,
                )

            venus_agent.step(goal_argument, image_path)

            last_action = getattr(venus_agent, "last_action", None)
            if (
                finished_event is None
                and last_action
                and isinstance(last_action, dict)
                and last_action.get("action", "").lower() == "finished"
            ):
                history_index = len(venus_agent.history) - 1
                step_snapshot = venus_agent.history[-1]
                finished_event = {
                    "step_index": history_index,
                    "trace_item_index": item_index,
                    "action": step_snapshot.action,
                    "conclusion": step_snapshot._conclusion,
                }
                logger.info(
                    "Finished action detected (trace=%d, step=%d, item=%d); "
                    "short-circuiting remaining frames.",
                    trace_index,
                    history_index,
                    item_index,
                )
                break

        history_record = venus_agent.export_history()
        trace_result: Dict[str, Any] = {
            "trace_index": trace_index,
            "mode": args.mode,
            "variant_id": (
                venus_agent.autonomous_variant.variant_id
                if is_autonomous and venus_agent.autonomous_variant
                else None
            ),
            "inputs": trace_inputs,
            "steps": history_record,
        }

        if is_autonomous and trace_inputs:
            trace_result["ground_truth_task"] = trace_inputs[0].get("task")

        if finished_event is not None:
            trace_result["finished_event"] = finished_event
            trace_result["stopped_on_finished"] = True
            trace_result["skipped_items_after_finished"] = len(trace) - (
                finished_event["trace_item_index"] + 1
            )
        else:
            trace_result["stopped_on_finished"] = False
            trace_result["skipped_items_after_finished"] = 0

        traces_output.append(trace_result)
        venus_agent.reset()

    run_summary: Dict[str, Any] = {
        "mode": args.mode,
        "variant_id": (
            venus_agent.autonomous_variant.variant_id
            if is_autonomous and venus_agent.autonomous_variant
            else None
        ),
        "input_file": args.input_file,
        "history_length": args.history_length,
    }
    if is_autonomous:
        run_summary["autonomous_context"] = venus_agent.autonomous_context

    save_json(
        args.output_file,
        {
            "run_summary": run_summary,
            "traces": traces_output,
        },
    )


if __name__ == "__main__":
    main()
