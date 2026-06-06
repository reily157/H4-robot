"""Tests for analyzer/ingest.py."""

from __future__ import annotations

import gzip
import json
import logging
import os
from datetime import timezone

import pandas as pd
import pytest

from analyzer.ingest import load_to_duckdb

# ---------------------------------------------------------------------------
# Fixtures — synthetic JSONL files
# ---------------------------------------------------------------------------

_BBO_ROW = {
    "ts_local": "2026-06-02T06:00:01.123456+00:00",
    "ts_remote": "2026-06-02T06:00:01.000000+00:00",
    "coin": "#1360",
    "bid_px": 0.55,
    "bid_sz": 100.0,
    "ask_px": 0.57,
    "ask_sz": 80.0,
}

_BOOK_ROW = {
    "ts_local": "2026-06-02T06:00:01.123456+00:00",
    "ts_remote": "2026-06-02T06:00:01.000000+00:00",
    "coin": "#1360",
    "side": "A",
    "level_idx": 0,
    "px": 0.57,
    "sz": 80.0,
    "n_orders": None,
}

_TRADES_ROW = {
    "ts_local": "2026-06-02T06:05:00.000000+00:00",
    "ts_remote": "2026-06-02T06:05:00.000000+00:00",
    "coin": "#1360",
    "px": 0.56,
    "sz": 10.0,
    "side": "B",
    "tid": "abc123",
}

_PERP_CTX_ROW = {
    "ts_local": "2026-06-02T06:00:00.000000+00:00",
    "coin": "BTC",
    "mark_px": 73000.0,
    "mid_px": 73001.0,
    "oracle_px": 73000.5,
    "funding": 0.0001,
    "open_interest": 12345.0,
}

_CYCLE_ROW = {
    "cycle_id": "BTC_202606020600",
    "started_at": "2026-06-01T07:01:14.402019+00:00",
    "bucket_question_id": 26,
    "bucket_expiry": "2026-06-02T06:00:00+00:00",
    "bucket_thresholds": "71869.0,74802.0",
    "bucket_underlying": "BTC",
    "binary_outcome_id": 136,
    "binary_target_price": 73336.0,
    "binary_expiry": "2026-06-02T06:00:00+00:00",
    "raw_meta": {"outcomes": [], "questions": []},
}

_OUTCOME_ROW = {
    "cycle_id": "BTC_202606020600",
    "outcome_id": 136,
    "role": "binary",
    "yes_coin": "#1360",
    "no_coin": "#1361",
    "description": None,
}


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_jsonl_gz(path: str, rows: list[dict]) -> None:
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_empty_dir(tmp_path):
    """Empty data directory → all tables created with 0 rows, no error."""
    con = load_to_duckdb(str(tmp_path))
    for table in ("bbo", "book_levels", "trades", "perp_ctx", "cycles", "outcomes_map"):
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert count == 0, f"Table {table} should be empty"


def test_load_jsonl_bbo(tmp_path):
    """Plain .jsonl bbo file → 3 rows with correct columns."""
    _write_jsonl(str(tmp_path / "bbo_20260602.jsonl"), [_BBO_ROW] * 3)
    con = load_to_duckdb(str(tmp_path))
    count = con.execute("SELECT count(*) FROM bbo").fetchone()[0]
    assert count == 3
    cols = {r[0] for r in con.execute("DESCRIBE bbo").fetchall()}
    for col in ("ts_local", "ts_remote", "coin", "bid_px", "bid_sz", "ask_px", "ask_sz"):
        assert col in cols, f"Column '{col}' missing from bbo"


def test_load_gz_bbo(tmp_path):
    """Compressed .jsonl.gz → same result as plain .jsonl."""
    _write_jsonl_gz(str(tmp_path / "bbo_20260602.jsonl.gz"), [_BBO_ROW] * 5)
    con = load_to_duckdb(str(tmp_path))
    count = con.execute("SELECT count(*) FROM bbo").fetchone()[0]
    assert count == 5


def test_load_mixed_gz_and_plain(tmp_path):
    """One .gz + one .jsonl for the same table → rows from both are loaded."""
    _write_jsonl_gz(str(tmp_path / "bbo_20260601.jsonl.gz"), [_BBO_ROW] * 4)
    _write_jsonl(str(tmp_path / "bbo_20260602.jsonl"), [_BBO_ROW] * 2)
    con = load_to_duckdb(str(tmp_path))
    count = con.execute("SELECT count(*) FROM bbo").fetchone()[0]
    assert count == 6


def test_load_cycles_dedup(tmp_path):
    """Two entries with same cycle_id → deduplicated to 1 row (latest started_at kept)."""
    row_early = {**_CYCLE_ROW, "started_at": "2026-06-01T07:00:00+00:00"}
    row_late = {**_CYCLE_ROW, "started_at": "2026-06-01T07:01:14+00:00"}
    _write_jsonl(str(tmp_path / "cycles.jsonl"), [row_early, row_late])
    con = load_to_duckdb(str(tmp_path))
    count = con.execute("SELECT count(*) FROM cycles").fetchone()[0]
    assert count == 1
    ts = con.execute("SELECT started_at FROM cycles").fetchone()[0]
    # Latest started_at should be kept
    assert "07:01" in str(ts)


def test_load_cycles_threshold_parsing(tmp_path):
    """bucket_thresholds string → threshold_low and threshold_high floats."""
    _write_jsonl(str(tmp_path / "cycles.jsonl"), [_CYCLE_ROW])
    con = load_to_duckdb(str(tmp_path))
    row = con.execute("SELECT threshold_low, threshold_high FROM cycles").fetchone()
    assert row[0] == pytest.approx(71869.0)
    assert row[1] == pytest.approx(74802.0)


def test_load_outcomes_map(tmp_path):
    """outcomes_map.jsonl → expected columns present."""
    _write_jsonl(str(tmp_path / "outcomes_map.jsonl"), [_OUTCOME_ROW])
    con = load_to_duckdb(str(tmp_path))
    count = con.execute("SELECT count(*) FROM outcomes_map").fetchone()[0]
    assert count == 1
    cols = {r[0] for r in con.execute("DESCRIBE outcomes_map").fetchall()}
    for col in ("cycle_id", "outcome_id", "role", "yes_coin", "no_coin"):
        assert col in cols, f"Column '{col}' missing from outcomes_map"


def test_timestamp_is_timezone_aware(tmp_path):
    """ts_local in bbo is loaded as timezone-aware (UTC) timestamp."""
    _write_jsonl(str(tmp_path / "bbo_20260602.jsonl"), [_BBO_ROW])
    con = load_to_duckdb(str(tmp_path))
    df = con.execute("SELECT ts_local FROM bbo LIMIT 1").fetchdf()
    ts = df["ts_local"].iloc[0]
    assert ts.tzinfo is not None or str(df["ts_local"].dtype).endswith("UTC"), (
        f"ts_local must be timezone-aware, got dtype={df['ts_local'].dtype}"
    )


def test_perp_ctx_columns(tmp_path):
    """perp_ctx table has all expected columns."""
    _write_jsonl(str(tmp_path / "perp_ctx_20260602.jsonl"), [_PERP_CTX_ROW])
    con = load_to_duckdb(str(tmp_path))
    cols = {r[0] for r in con.execute("DESCRIBE perp_ctx").fetchall()}
    for col in ("ts_local", "coin", "mark_px", "mid_px", "oracle_px", "funding", "open_interest"):
        assert col in cols


def test_book_levels_columns(tmp_path):
    """book_levels table has all expected columns including n_orders."""
    _write_jsonl(str(tmp_path / "book_levels_20260602.jsonl"), [_BOOK_ROW])
    con = load_to_duckdb(str(tmp_path))
    cols = {r[0] for r in con.execute("DESCRIBE book_levels").fetchall()}
    for col in ("ts_local", "ts_remote", "coin", "side", "level_idx", "px", "sz", "n_orders"):
        assert col in cols


def test_trades_columns(tmp_path):
    """trades table has all expected columns including tid."""
    _write_jsonl(str(tmp_path / "trades_20260602.jsonl"), [_TRADES_ROW])
    con = load_to_duckdb(str(tmp_path))
    cols = {r[0] for r in con.execute("DESCRIBE trades").fetchall()}
    for col in ("ts_local", "ts_remote", "coin", "px", "sz", "side", "tid"):
        assert col in cols


def test_returns_duckdb_connection(tmp_path):
    """load_to_duckdb returns a live DuckDB connection."""
    import duckdb
    con = load_to_duckdb(str(tmp_path))
    assert isinstance(con, duckdb.DuckDBPyConnection)
    # Connection must be usable
    result = con.execute("SELECT 42").fetchone()[0]
    assert result == 42


@pytest.mark.slow
def test_real_data_counts(caplog):
    """
    Smoke test: load the real data/ directory.
    Logs counts per table — no assertions on exact numbers (dataset evolves).

    book_levels is excluded: 2.8 GB compressed on a 1.9 GB RAM / no-swap VPS.
    It is not used by any of H1-H7; other tables are sufficient for analysis.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        pytest.skip("data/ directory not found")

    with caplog.at_level(logging.INFO, logger="analyzer.ingest"):
        con = load_to_duckdb(data_dir, skip_tables=frozenset({"book_levels"}))

    for table in ("bbo", "trades", "perp_ctx", "cycles", "outcomes_map"):
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count:,} rows")
        assert count >= 0  # trivially true — just verifying no error

    # book_levels was skipped — table exists but is empty
    bl_count = con.execute("SELECT count(*) FROM book_levels").fetchone()[0]
    assert bl_count == 0
