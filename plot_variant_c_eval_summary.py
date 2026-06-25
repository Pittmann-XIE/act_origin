#!/usr/bin/env python3
"""Plot Variant C RQ success, future loss, and compression ratio together."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERNS = {
    "success_rate": re.compile(r"^Success rate:\s*([0-9.]+)", re.MULTILINE),
    "compression_ratio": re.compile(r"^compression_ratio:\s*([0-9.]+)", re.MULTILINE),
    "eval_future_loss": re.compile(r"^eval_future_loss:\s*([0-9.]+)", re.MULTILINE),
}


def parse_result(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    stage_match = re.search(r"rq([0-9]+)stages", path.name)
    if stage_match is None:
        raise ValueError(f"Cannot infer RQ stage from {path.name}")

    row = {"stage": float(stage_match.group(1))}
    for key, pattern in PATTERNS.items():
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Missing {key} in {path}")
        row[key] = float(match.group(1))
    return row


def build_plot(rows: list[dict[str, float]], output: Path) -> None:
    rows = sorted(rows, key=lambda row: row["stage"])
    stages = np.array([row["stage"] for row in rows])
    labels = [f"RQ-{int(stage)}" for stage in stages]
    success = np.array([row["success_rate"] * 100.0 for row in rows])
    future_loss = np.array([row["eval_future_loss"] for row in rows])
    compression = np.array([row["compression_ratio"] for row in rows])

    fig, ax_success = plt.subplots(figsize=(9.5, 5.8))
    ax_loss = ax_success.twinx()
    ax_compression = ax_success.twinx()
    ax_compression.spines["right"].set_position(("axes", 1.14))

    bars = ax_compression.bar(
        stages,
        compression,
        width=0.58,
        color="#d9e2ec",
        edgecolor="#8fa3b8",
        alpha=0.45,
        label="Compression ratio",
        zorder=1,
    )
    ax_compression.set_ylabel("Compression ratio (x)", color="#53657a")
    ax_compression.tick_params(axis="y", colors="#53657a")
    ax_compression.set_ylim(0, compression.max() * 1.18)

    success_line = ax_success.plot(
        stages,
        success,
        color="#1f77b4",
        marker="o",
        linewidth=2.6,
        markersize=7,
        label="Success rate",
        zorder=3,
    )
    loss_line = ax_loss.plot(
        stages,
        future_loss,
        color="#d62728",
        marker="s",
        linewidth=2.6,
        markersize=6.5,
        label="eval_future_loss",
        zorder=4,
    )

    ax_success.set_title("Variant C RQ Evaluation Summary", fontsize=15, fontweight="bold", pad=12)
    ax_success.set_xlabel("Active RQ stages")
    ax_success.set_ylabel("Success rate (%)", color="#1f77b4")
    ax_loss.set_ylabel("eval_future_loss", color="#d62728")
    ax_success.tick_params(axis="y", colors="#1f77b4")
    ax_loss.tick_params(axis="y", colors="#d62728")
    ax_success.set_xticks(stages)
    ax_success.set_xticklabels(labels)
    ax_success.set_ylim(0, 100)
    ax_loss.set_ylim(0.010, future_loss.max() * 1.03)
    ax_success.grid(axis="y", color="#d8dee6", linewidth=0.8)
    ax_success.set_axisbelow(True)

    for rect, value in zip(bars, compression):
        ax_compression.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height() + compression.max() * 0.025,
            f"{value:.0f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#53657a",
        )

    for x, y in zip(stages, success):
        ax_success.text(x, y + 1.6, f"{y:.0f}%", ha="center", color="#1f77b4", fontsize=9)
    for x, y in zip(stages, future_loss):
        ax_loss.text(x, y + 0.00018, f"{y:.5f}", ha="center", color="#d62728", fontsize=9)

    handles = [bars, success_line[0], loss_line[0]]
    labels = [handle.get_label() for handle in handles]
    ax_success.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.89), frameon=True)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir",
        type=Path,
        nargs="?",
        default=Path("checkpoints/checkpoints_variant_c_sim_transfer_cube_scripted_top_rq_N30_M4_K512_D512_hierarchy_three_stage"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.checkpoint_dir.glob("result_policy_best_rq*stages.txt"))
    if not paths:
        raise SystemExit(f"No result files found in {args.checkpoint_dir}")

    output = args.output or args.checkpoint_dir / "variant_c_rq_eval_summary.png"
    build_plot([parse_result(path) for path in paths], output)
    print(output)
    print(output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
