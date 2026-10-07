"""CLI for opt-in model and moderation provider canaries."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run model/moderation provider canaries.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-only",
        action="store_true",
        help="probe only the configured model chain",
    )
    group.add_argument(
        "--moderation-only",
        action="store_true",
        help="probe only the configured moderation endpoint",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit one JSON object instead of human-readable lines",
    )
    return parser


async def _run(args: argparse.Namespace) -> dict:
    from .core.provider_canary import run_provider_canary

    return await run_provider_canary(
        include_model=not args.moderation_only,
        include_moderation=not args.model_only,
    )


def main() -> int:
    args = _parser().parse_args()
    result = asyncio.run(_run(args))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for check in result.get("checks") or []:
            detail = check.get("error") or check.get("reason") or ""
            suffix = f" ({detail})" if detail else ""
            print(
                f"{check.get('check')}: "
                f"{check.get('status')}{suffix}"
            )
    return 0 if result.get("status") == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
