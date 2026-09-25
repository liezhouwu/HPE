"""扫描 try3 已完成实验并生成易读的汇总报告。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).parents[1]
DEFAULT_RESULT_ROOTS = (ROOT / "result", ROOT / "result_metafi_ssl")
DEFAULT_REPORT_DIR = ROOT / "reports"


@dataclass(frozen=True)
class ResultRecord:
    category: str
    method: str
    protocol: str
    split: str
    scope: str
    label_budget: str
    seed: str
    checkpoint: str
    epoch: int | None
    select_mpjpe_mm: float | None
    test_mpjpe_mm: float | None
    test_pelvis_mpjpe_mm: float | None
    test_pa_mpjpe_mm: float | None
    train_time_s: float | None
    result_dir: str


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _identity(result_dir: Path) -> Mapping[str, Any]:
    return _read_json(result_dir / "run_identity.json") or {}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _category(result_dir: Path, identity: Mapping[str, Any]) -> str:
    parts = {part.lower() for part in result_dir.parts}
    method = str(identity.get("method", ""))
    if "result_metafi_ssl" not in parts:
        return "baseline"
    prefix = "small" if any(part.startswith("small") for part in parts) else "full"
    if method == "sup" or method.startswith("sup_"):
        # ViT 线的随机初始化监督对照方法名为 sup_vit，与 MetaFi 侧的 sup 同属监督对照。
        return f"{prefix}-supervised"
    return f"{prefix}-ssl"


def _record_from_absolute(path: Path) -> ResultRecord | None:
    result_dir = path.parent
    data = _read_json(path)
    if data is None or not isinstance(data.get("test_metrics"), Mapping):
        return None
    identity = _identity(result_dir)
    metrics = data["test_metrics"]
    return ResultRecord(
        category=_category(result_dir, identity),
        method=str(identity.get("method", "baseline")),
        protocol=str(data.get("protocol", identity.get("protocol", "?"))),
        split=str(data.get("split", identity.get("split", "?"))),
        scope=str(identity.get("data_scope", "official")),
        label_budget=str(identity.get("label_budget", "100%")),
        seed=str(identity.get("finetune_seed", identity.get("pretrain_seed", "-"))),
        checkpoint=str(data.get("checkpoint_file", "best_absolute.pth")),
        epoch=data.get("selected_epoch") if isinstance(data.get("selected_epoch"), int) else None,
        select_mpjpe_mm=_number(data.get("selected_value")),
        test_mpjpe_mm=_number(metrics.get("absolute_mpjpe_mm")),
        test_pelvis_mpjpe_mm=_number(metrics.get("pelvis_mpjpe_mm")),
        test_pa_mpjpe_mm=_number(metrics.get("pa_mpjpe_mm")),
        train_time_s=_number(data.get("train_time_s")),
        result_dir=_display_path(result_dir),
    )


def _record_from_legacy(path: Path) -> ResultRecord | None:
    result_dir = path.parent
    if (result_dir / "summary_absolute.json").is_file():
        return None
    data = _read_json(path)
    if data is None:
        return None
    metrics = data.get("test_metrics", data)
    if not isinstance(metrics, Mapping) or "test_mpjpe_mm" not in metrics and "absolute_mpjpe_mm" not in metrics:
        return None
    identity = _identity(result_dir)
    return ResultRecord(
        category=_category(result_dir, identity),
        method=str(identity.get("method", "baseline")),
        protocol=str(data.get("protocol", identity.get("protocol", "?"))),
        split=str(data.get("split", identity.get("split", "?"))),
        scope=str(identity.get("data_scope", "official")),
        label_budget=str(identity.get("label_budget", "100%")),
        seed=str(identity.get("finetune_seed", data.get("seed", "-"))),
        checkpoint="best_model.pth",
        epoch=data.get("best_epoch") if isinstance(data.get("best_epoch"), int) else None,
        select_mpjpe_mm=_number(data.get("best_selection_value")),
        test_mpjpe_mm=_number(metrics.get("test_mpjpe_mm", metrics.get("absolute_mpjpe_mm"))),
        test_pelvis_mpjpe_mm=_number(metrics.get("test_mpjpe_pelvis_mm", metrics.get("pelvis_mpjpe_mm"))),
        test_pa_mpjpe_mm=_number(metrics.get("test_pampjpe_mm", metrics.get("pa_mpjpe_mm"))),
        train_time_s=_number(data.get("train_time_s")),
        result_dir=_display_path(result_dir),
    )


def collect_records(result_roots: Iterable[Path] = DEFAULT_RESULT_ROOTS) -> list[ResultRecord]:
    """读取每个已完成 run 的主 absolute checkpoint，跳过中断或重复工件。"""
    records: list[ResultRecord] = []
    seen: set[Path] = set()
    for root in result_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("summary_absolute.json"):
            record = _record_from_absolute(path)
            if record is not None:
                records.append(record)
                seen.add(path.parent.resolve())
        for path in root.rglob("summary.json"):
            if path.parent.resolve() in seen:
                continue
            record = _record_from_legacy(path)
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda item: (item.protocol, item.split, item.category, item.method, item.seed, item.result_dir))


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(records: list[ResultRecord], report_dir: Path = DEFAULT_REPORT_DIR) -> tuple[Path, Path, Path]:
    """写入 Markdown、CSV 和 JSON 三种汇总格式。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "RESULTS_SUMMARY.md"
    csv_path = report_dir / "RESULTS_SUMMARY.csv"
    json_path = report_dir / "RESULTS_SUMMARY.json"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# try3 实验结果汇总",
        "",
        f"生成时间：{generated_at}",
        "",
        f"已收集完成实验：{len(records)} 个。每行只使用该 run 的主 absolute checkpoint，避免混用不同 checkpoint 的单项最优指标。",
        "",
        "| 类别 | 方法 | Protocol | Split | 标签 | Seed | Epoch | Test official MPJPE | Test pelvisMPJPE | Test PA-MPJPE | 训练时间 | 结果目录 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in records:
        minutes = None if item.train_time_s is None else item.train_time_s / 60.0
        lines.append(
            f"| {item.category} | {item.method} | {item.protocol} | {item.split} | {item.label_budget} | {item.seed} | {item.epoch if item.epoch is not None else '-'} | "
            f"{_fmt(item.test_mpjpe_mm)} mm | {_fmt(item.test_pelvis_mpjpe_mm)} mm | {_fmt(item.test_pa_mpjpe_mm)} mm | {_fmt(minutes)} min | {item.result_dir} |"
        )
    lines.extend([
        "",
        "## 阅读提示",
        "",
        "- baseline：官方完整监督复现结果。",
        "- small-supervised：4shot 小样本随机初始化监督对照。",
        "- small-ssl：小样本无标签预训练后再用同一批 4shot 标签微调。",
        "- full-ssl：全量 strict 无标签预训练后微调。",
        "- 小样本对比应优先比较相同 protocol、split、seed、label budget 下的 small-supervised 与 small-ssl。",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    fields = list(ResultRecord.__annotations__.keys())
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            writer.writerow(asdict(item))
    json_path.write_text(json.dumps([asdict(item) for item in records], ensure_ascii=False, indent=2), encoding="utf-8")

    # latest 文件方便直接查看；每次同时复制一个带时间戳快照，避免历史报告被覆盖。
    snapshot_dir = report_dir / "history" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    for source in (markdown_path, csv_path, json_path):
        shutil.copy2(source, snapshot_dir / source.name)
    return markdown_path, csv_path, json_path


def generate_report() -> list[ResultRecord]:
    records = collect_records()
    markdown_path, csv_path, json_path = write_report(records)
    print(f"[汇总] {len(records)} 个完成实验 → {markdown_path.relative_to(ROOT)}", flush=True)
    print(f"       {csv_path.relative_to(ROOT)} | {json_path.relative_to(ROOT)}", flush=True)
    print("       历史快照已写入 reports/history/", flush=True)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 try3 实验结果汇总报告")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    records = collect_records()
    markdown_path, csv_path, json_path = write_report(records, args.report_dir)
    print(f"[汇总] {len(records)} 个完成实验", flush=True)
    print(markdown_path)
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
