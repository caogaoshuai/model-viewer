from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .diff import compare_models
from .formatting import json_dumps
from .parsing import load_model
from .rendering import memory_summary, parse_views, render_diff, render_memory, render_show


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mad",
        description="Model architecture viewer and diff CLI.",
    )
    parser.add_argument("--model-source", choices=["auto", "local", "hf", "ms"], default="auto")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--hub-token", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show", help="Render one model.")
    show.add_argument("model")
    show.add_argument("--view", default="overview", help="all or comma list: overview,heatmap,detail,mapping,memory,tree,patterns")
    show.add_argument("--format", choices=["term", "markdown", "mermaid", "drawio", "html", "json"], default="term")
    show.add_argument("--layer", type=int, default=0)
    show.add_argument("-o", "--output")

    diff = subparsers.add_parser("diff", help="Compare two models.")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.add_argument("--view", default="all", help="all or comma list: overview,heatmap,detail,mapping,memory,tree,patterns")
    diff.add_argument("--format", choices=["term", "markdown", "mermaid", "drawio", "html", "json"], default="term")
    diff.add_argument("--layer", type=int, default=0)
    diff.add_argument("--fuzzy-match", action="store_true", help="Detect fused qkv/gate_up and tied lm_head mappings.")
    diff.add_argument("--fail-on-change", action="store_true", help="Exit 2 when any non-exact diff row exists.")
    diff.add_argument("-o", "--output")

    snapshot = subparsers.add_parser("snapshot", help="Export normalized model metadata.")
    snapshot.add_argument("model")
    snapshot.add_argument("-o", "--output", required=True)

    memory = subparsers.add_parser("memory", help="Render memory footprint.")
    memory.add_argument("model")
    memory.add_argument("--mode", choices=["train", "deploy"], default="deploy")
    memory.add_argument("--seq-len", type=int, default=None)
    memory.add_argument("--batch-size", type=int, default=1)
    memory.add_argument("--format", choices=["term", "markdown", "html", "json"], default="term")
    memory.add_argument("-o", "--output")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            snapshot = _load(args.model, args)
            content = render_show(snapshot, parse_views(args.view), args.format, layer=args.layer)
            _write_or_print(content, args.output)
            _print_warnings(snapshot)
            return 0

        if args.command == "diff":
            left = _load(args.left, args)
            right = _load(args.right, args)
            model_diff = compare_models(left, right, fuzzy_match=args.fuzzy_match)
            content = render_diff(model_diff, parse_views(args.view), args.format, layer=args.layer)
            _write_or_print(content, args.output)
            _print_warnings(left)
            _print_warnings(right)
            if args.fail_on_change and model_diff.has_change:
                return 2
            return 0

        if args.command == "snapshot":
            snapshot = _load(args.model, args)
            _write_or_print(json_dumps(snapshot.to_dict()), args.output)
            _print_warnings(snapshot)
            return 0

        if args.command == "memory":
            snapshot = _load(args.model, args)
            if args.format == "json":
                buckets = memory_summary(
                    snapshot,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    include_kv=args.mode == "deploy",
                )
                content = json_dumps({"model": snapshot.name, "mode": args.mode, "buckets": buckets})
            else:
                content = render_memory(
                    snapshot,
                    seq_len=args.seq_len,
                    batch_size=args.batch_size,
                    include_kv=args.mode == "deploy",
                )
            if args.format == "html":
                from .formatting import html_page

                content = html_page(f"Memory Footprint: {snapshot.name}", content)
            _write_or_print(content, args.output)
            _print_warnings(snapshot)
            return 0

    except Exception as error:
        print(f"mad: error: {error}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


def _load(model: str, args: argparse.Namespace):
    return load_model(
        model,
        model_source=getattr(args, "model_source", "auto"),
        revision=getattr(args, "model_revision", None),
        hub_token=getattr(args, "hub_token", None),
    )


def _write_or_print(content: str, output: Optional[str]) -> None:
    if output:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + ("\n" if not content.endswith("\n") else ""), encoding="utf-8")
        return
    print(content)


def _print_warnings(snapshot) -> None:
    for warning in snapshot.warnings:
        print(f"mad: warning: {snapshot.name}: {warning}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
