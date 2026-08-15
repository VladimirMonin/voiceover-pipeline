#!/usr/bin/env python3
"""Run one local ASR benchmark adapter against an explicit corpus manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from voiceover_pipeline.asr_benchmark import (
    create_benchmark_adapter,
    run_benchmark,
    write_benchmark_reports,
)


def _prompt_text(path: Path | None) -> str | None:
    if path is None:
        return None
    if path.name.startswith(".env"):
        raise ValueError("Prompt files named .env are not accepted")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Prompt file must not be blank")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Manifest JSON path")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Optional local root for corpus-relative audio paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Report directory")
    parser.add_argument(
        "--provider",
        choices=("whisper", "qwen-local", "nemotron-local"),
        required=True,
    )
    parser.add_argument("--model", default=None, help="Optional local model identifier")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute", default="auto")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional non-.env text file for a prompt on/off comparison",
    )
    args = parser.parse_args()

    adapter = create_benchmark_adapter(
        args.provider,
        model_id=args.model,
        device=args.device,
        compute=args.compute,
    )
    report = run_benchmark(
        args.corpus,
        adapter,
        corpus_root=args.corpus_root,
        prompt_text=_prompt_text(args.prompt_file),
    )
    json_path, markdown_path = write_benchmark_reports(report, args.output_dir)
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
