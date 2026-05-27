"""
Isolated WebSocket diagnostic — uses aiohttp directly (same lib as prod).

Two phases, 30s each:
  Phase A (0–30s)  : connect + subscribe BTC trades, NO ping sent
  Phase B (30–60s) : send {"method": "ping"} every 15s

Prints a per-second message count and a final summary so we can distinguish:
  - Connexion refusée / TLS error       → problème IP/ban externe
  - Connecté mais 0 msg sur 60s         → HL ne push pas (confirme heartbeat bug)
  - Messages dès phase A                → problème ailleurs dans ws_client
  - Messages seulement après ping (B)   → heartbeat applicatif requis, confirmé
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import aiohttp

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

WS_URL = "wss://api.hyperliquid.xyz/ws"

SUB_BTC_TRADES = {
    "method": "subscribe",
    "subscription": {"type": "trades", "coin": "BTC"},
}

PHASE_A_DURATION = 30   # seconds — no pings
PHASE_B_DURATION = 30   # seconds — ping every 15s
PING_INTERVAL    = 15   # seconds within phase B
PING_FRAME       = {"method": "ping"}


async def run_test() -> None:
    counters = {"phase_a": 0, "phase_b": 0, "pong": 0, "other": 0}
    raw_msgs: list[tuple[float, dict]] = []   # (elapsed, msg) for first 20 msgs

    log.info("=== PHASE A : connecting (no ping) ===")
    t0 = time.monotonic()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WS_URL) as ws:
                log.info(f"connected to {WS_URL}")

                await ws.send_json(SUB_BTC_TRADES)
                log.info(f"sent subscription: {SUB_BTC_TRADES}")

                phase = "A"
                phase_b_start: float | None = None
                last_ping_at: float = t0

                async def _recv() -> None:
                    nonlocal phase, phase_b_start, last_ping_at

                    async for msg in ws:
                        elapsed = time.monotonic() - t0
                        now_phase = "A" if elapsed < PHASE_A_DURATION else "B"

                        if now_phase != phase:
                            phase = "B"
                            phase_b_start = elapsed
                            log.info(f"=== PHASE B : ping every {PING_INTERVAL}s ===")

                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                data = json.loads(msg.data)
                            except json.JSONDecodeError:
                                log.warning(f"invalid JSON: {msg.data[:80]}")
                                continue

                            channel = data.get("channel", "")

                            if channel == "pong":
                                counters["pong"] += 1
                                log.info(f"[{elapsed:.1f}s] PONG received")
                            elif channel in ("trades", "l2Book", "bbo",
                                             "activeAssetCtx", "subscriptionResponse"):
                                counters[f"phase_{phase.lower()}"] += 1
                                if len(raw_msgs) < 20:
                                    raw_msgs.append((elapsed, data))
                                if counters[f"phase_{phase.lower()}"] <= 3:
                                    log.info(
                                        f"[{elapsed:.1f}s] msg #{counters[f'phase_{phase.lower()}']} "
                                        f"channel={channel!r} "
                                        f"payload_len={len(msg.data)}"
                                    )
                            else:
                                counters["other"] += 1
                                log.debug(f"[{elapsed:.1f}s] unknown channel={channel!r}: {msg.data[:120]}")

                        elif msg.type == aiohttp.WSMsgType.PING:
                            log.debug(f"[{elapsed:.1f}s] WS PING frame (protocol level)")
                        elif msg.type == aiohttp.WSMsgType.PONG:
                            log.debug(f"[{elapsed:.1f}s] WS PONG frame (protocol level)")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            log.warning(f"[{elapsed:.1f}s] ERROR frame: {ws.exception()}")
                            return
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSING,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            log.warning(f"[{elapsed:.1f}s] server closed: {msg.type.name} {msg.data}")
                            return

                async def _control() -> None:
                    """Switch to phase B and send pings; stop after total duration."""
                    nonlocal phase

                    # Wait for phase A to end
                    await asyncio.sleep(PHASE_A_DURATION)
                    phase = "B"
                    log.info(f"=== PHASE B : sending ping every {PING_INTERVAL}s ===")

                    elapsed = time.monotonic() - t0
                    deadline = t0 + PHASE_A_DURATION + PHASE_B_DURATION

                    while time.monotonic() < deadline:
                        await ws.send_json(PING_FRAME)
                        log.info(f"[{time.monotonic()-t0:.1f}s] sent {PING_FRAME}")
                        await asyncio.sleep(PING_INTERVAL)

                    log.info("=== total duration reached, closing ===")
                    await ws.close()

                recv_task    = asyncio.create_task(_recv())
                control_task = asyncio.create_task(_control())

                done, pending = await asyncio.wait(
                    [recv_task, control_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass

    except aiohttp.ClientConnectorError as e:
        log.error(f"CONNECTION REFUSED / network error: {e}")
        log.error("→ Scénario : problème IP/ban ou réseau externe")
        return
    except Exception as e:
        log.error(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        return

    # ── Summary ──────────────────────────────────────────────────────────────
    total = counters["phase_a"] + counters["phase_b"]
    print("\n" + "=" * 60)
    print("RÉSULTATS DU TEST ISOLÉ")
    print("=" * 60)
    print(f"  Phase A (0–{PHASE_A_DURATION}s, sans ping) : {counters['phase_a']} messages")
    print(f"  Phase B ({PHASE_A_DURATION}–{PHASE_A_DURATION+PHASE_B_DURATION}s, avec ping) : {counters['phase_b']} messages")
    print(f"  Pongs reçus                        : {counters['pong']}")
    print(f"  Autres channels                    : {counters['other']}")
    print(f"  TOTAL messages de données          : {total}")
    print()

    if total == 0:
        print("DIAGNOSTIC : 0 message reçu en 60s")
        if counters["pong"] > 0:
            print("  → HL répond aux pings mais ne push pas de données")
            print("  → Possible : IP soft-bannie (data blacklistée mais WS acceptée)")
        else:
            print("  → Ni données ni pong reçus")
            print("  → Possible : IP bannie ou WS fermé silencieusement par HL")
    elif counters["phase_a"] == 0 and counters["phase_b"] > 0:
        print("DIAGNOSTIC : données seulement APRÈS envoi du ping")
        print("  → HEARTBEAT APPLICATIF REQUIS CONFIRMÉ")
        print("  → Fix : ajouter ping JSON toutes les 50s dans ws_client.py")
    elif counters["phase_a"] > 0:
        print("DIAGNOSTIC : données reçues dès la Phase A (sans ping)")
        print("  → HL push normalement, le bug est INTERNE à ws_client/recorder")
    print("=" * 60)

    if raw_msgs:
        print(f"\nPremiers messages reçus ({min(len(raw_msgs), 5)}) :")
        for elapsed, d in raw_msgs[:5]:
            ch = d.get("channel", "?")
            print(f"  [{elapsed:.1f}s] channel={ch!r}  keys={list(d.keys())}")


if __name__ == "__main__":
    asyncio.run(run_test())
