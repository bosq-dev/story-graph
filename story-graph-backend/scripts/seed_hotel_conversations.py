#!/usr/bin/env python3
"""Seed a fresh Story Graph instance with deterministic hotel conversations.

Creates 4 conversations using profile `hotel_customer_service` with overlap between
rooms, issues and requested actions so the resulting graph is useful for admin analysis.

This script uses the backend API so all normal extraction/pipeline logic is exercised.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class ConversationSeed:
    user_id: str
    user_name: str
    messages: list[str]


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to reach backend at {url}: {exc}") from exc


def _get_health(url: str, timeout: int) -> None:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Health endpoint returned status {resp.status}")
    except Exception as exc:
        raise RuntimeError(f"Backend health check failed at {url}: {exc}") from exc


def run_seed(base_url: str, timeout: int, pause_seconds: float) -> None:
    health_url = f"{base_url.rstrip('/')}/health"
    message_url = f"{base_url.rstrip('/')}/chat/message"

    _get_health(health_url, timeout=timeout)

    # Admin question ideas for this dataset:
    # - Quais problemas mais recorrentes por quarto?
    # - Quais usuarios pediram troca de quarto ou troca de travesseiro?
    # - O quarto 2 tem historico de mal cheiro com quantos hospedes diferentes?
    # - Quais pedidos de reembolso estao associados a problemas de limpeza?
    # - Quais atividades foram solicitadas para o quarto 7 apos relatos de barulho?
    seeds = [
        ConversationSeed(
            user_id="guest-ana",
            user_name="Ana",
            messages=[
                "Oi, estou no quarto 2 e o quarto esta com cheiro ruim desde cedo.",
                "Preciso trocar de quarto por causa desse mal cheiro no quarto 2.",
                "Se nao tiver troca hoje, quero reembolso parcial da diaria.",
            ],
        ),
        ConversationSeed(
            user_id="guest-bruno",
            user_name="Bruno",
            messages=[
                "Boa noite, fiquei no quarto 2 e senti mal cheiro muito forte.",
                # "Tambem preciso de troca de travesseiros, os que vieram estao umidos.",
            ],
        ),
        # ConversationSeed(
        #     user_id="guest-carla",
        #     user_name="Carla",
        #     messages=[
        #         "Estou no quarto 7 e tem barulho no corredor durante a madrugada.",
        #         "Quero troca de quarto, nao consigo dormir com esse barulho.",
        #         "Tambem peço checkout tardio sem custo por causa do incidente.",
        #     ],
        # ),
        ConversationSeed(
            user_id="guest-diego",
            user_name="Diego",
            messages=[
                # "No quarto 7 encontrei sujeira no banheiro e toalhas manchadas.",
                "Solicito limpeza imediata do quarto 7 e troca de enxoval.",
                # "Se isso nao resolver hoje, vou pedir cancelamento da reserva.",
            ],
        ),
    ]

    total_messages = 0
    session_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        session_id: str | None = None
        triplets_total = 0

        for idx, message in enumerate(seed.messages, start=1):
            payload = {
                "message": message,
                "session_id": session_id,
                "user_id": seed.user_id,
                "user_name": seed.user_name,
                "prompt_profile": "hotel_customer_service",
            }
            data = _post_json(message_url, payload, timeout=timeout)
            session_id = str(data.get("session_id") or "").strip() or session_id
            triplets_total += int(data.get("stored_triplets_count", 0))
            total_messages += 1

            print(
                f"[{seed.user_name}] msg {idx}/{len(seed.messages)} -> "
                f"session={session_id} stored_triplets={data.get('stored_triplets_count', 0)}"
            )

            if pause_seconds > 0:
                time.sleep(pause_seconds)

        session_summaries.append(
            {
                "user_name": seed.user_name,
                "user_id": seed.user_id,
                "session_id": session_id,
                "messages": len(seed.messages),
                "stored_triplets_total": triplets_total,
            }
        )

    print("\nSeed finished.")
    print(f"Conversations created: {len(session_summaries)}")
    print(f"Messages sent: {total_messages}")
    print("Session summary:")
    for item in session_summaries:
        print(
            f"- user={item['user_name']} ({item['user_id']}) "
            f"session={item['session_id']} "
            f"messages={item['messages']} "
            f"stored_triplets_total={item['stored_triplets_total']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed 3 hotel conversations in Story Graph backend."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between messages to ease log inspection (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        run_seed(base_url=args.base_url, timeout=args.timeout, pause_seconds=args.pause_seconds)
        return 0
    except Exception as exc:
        print(f"Seed failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
