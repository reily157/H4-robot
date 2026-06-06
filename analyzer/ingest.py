"""
JSONL → DuckDB in-memory loader for the H4-robot analyzer.

All JSONL files are read without modification (read-only).
DuckDB reads .gz files natively — no external decompression needed.
"""

from __future__ import annotations

import glob
import logging
import os

import duckdb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-table configuration: SELECT clause, raw column types, empty DDL
# ---------------------------------------------------------------------------
# Timestamps (ts_local, ts_remote) are read as VARCHAR to avoid DuckDB
# auto-inference failures when some rows lack microseconds precision.
# The ::TIMESTAMPTZ cast in the SELECT clause handles all ISO 8601 variants.

_BBO_COLS = """
    ts_local::TIMESTAMPTZ   AS ts_local,
    ts_remote               AS ts_remote,
    coin                    AS coin,
    bid_px                  AS bid_px,
    bid_sz                  AS bid_sz,
    ask_px                  AS ask_px,
    ask_sz                  AS ask_sz
"""

_BOOK_LEVELS_COLS = """
    ts_local::TIMESTAMPTZ        AS ts_local,
    ts_remote                    AS ts_remote,
    coin                         AS coin,
    side                         AS side,
    level_idx                    AS level_idx,
    px                           AS px,
    sz                           AS sz,
    TRY_CAST(n_orders AS BIGINT) AS n_orders
"""

_TRADES_COLS = """
    ts_local::TIMESTAMPTZ   AS ts_local,
    ts_remote               AS ts_remote,
    coin                    AS coin,
    px                      AS px,
    sz                      AS sz,
    side                    AS side,
    tid                     AS tid
"""

_PERP_CTX_COLS = """
    ts_local::TIMESTAMPTZ   AS ts_local,
    coin                    AS coin,
    mark_px                 AS mark_px,
    mid_px                  AS mid_px,
    oracle_px               AS oracle_px,
    funding                 AS funding,
    open_interest           AS open_interest
"""

# Explicit raw column types used with read_json (overrides auto-detection).
# All timestamp fields are VARCHAR here; casts happen in the SELECT clause above.
_RAW_COLUMNS: dict[str, dict[str, str]] = {
    "bbo": {
        "ts_local": "VARCHAR", "ts_remote": "VARCHAR", "coin": "VARCHAR",
        "bid_px": "DOUBLE", "bid_sz": "DOUBLE", "ask_px": "DOUBLE", "ask_sz": "DOUBLE",
    },
    "book_levels": {
        "ts_local": "VARCHAR", "ts_remote": "VARCHAR", "coin": "VARCHAR",
        "side": "VARCHAR", "level_idx": "BIGINT",
        "px": "DOUBLE", "sz": "DOUBLE", "n_orders": "JSON",
    },
    "trades": {
        "ts_local": "VARCHAR", "ts_remote": "VARCHAR", "coin": "VARCHAR",
        "px": "DOUBLE", "sz": "DOUBLE", "side": "VARCHAR", "tid": "VARCHAR",
    },
    "perp_ctx": {
        "ts_local": "VARCHAR", "coin": "VARCHAR",
        "mark_px": "DOUBLE", "mid_px": "DOUBLE", "oracle_px": "DOUBLE",
        "funding": "DOUBLE", "open_interest": "DOUBLE",
    },
}

# Explicit DDL for empty tables (used when no files match the prefix or table is skipped).
_EMPTY_DDL: dict[str, str] = {
    "bbo": (
        "CREATE TABLE bbo ("
        "ts_local TIMESTAMPTZ, ts_remote VARCHAR, coin VARCHAR,"
        "bid_px DOUBLE, bid_sz DOUBLE, ask_px DOUBLE, ask_sz DOUBLE)"
    ),
    "book_levels": (
        "CREATE TABLE book_levels ("
        "ts_local TIMESTAMPTZ, ts_remote VARCHAR, coin VARCHAR,"
        "side VARCHAR, level_idx BIGINT, px DOUBLE, sz DOUBLE, n_orders BIGINT)"
    ),
    "trades": (
        "CREATE TABLE trades ("
        "ts_local TIMESTAMPTZ, ts_remote VARCHAR, coin VARCHAR,"
        "px DOUBLE, sz DOUBLE, side VARCHAR, tid VARCHAR)"
    ),
    "perp_ctx": (
        "CREATE TABLE perp_ctx ("
        "ts_local TIMESTAMPTZ, coin VARCHAR,"
        "mark_px DOUBLE, mid_px DOUBLE, oracle_px DOUBLE,"
        "funding DOUBLE, open_interest DOUBLE)"
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_files(data_dir: str, prefix: str) -> list[str]:
    """
    Return sorted list of all .jsonl.gz and .jsonl files matching prefix_*.
    Plain .jsonl files come from the current (still-writing) day.
    """
    gz = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.jsonl.gz")))
    plain = sorted(glob.glob(os.path.join(data_dir, f"{prefix}_*.jsonl")))
    return gz + plain


def _file_list_sql(files: list[str]) -> str:
    """Return a SQL list literal for read_json_auto(['f1', 'f2', ...])."""
    quoted = ", ".join(f"'{f}'" for f in files)
    return f"[{quoted}]"


def _load_event_table(
    con: duckdb.DuckDBPyConnection,
    data_dir: str,
    table_name: str,
    prefix: str,
    select_cols: str,
) -> int:
    """
    Create table_name from all matching JSONL/gz files.
    If no files found, creates an empty table by reading 0 rows.
    Returns row count.
    """
    files = _find_files(data_dir, prefix)

    if not files:
        con.execute(_EMPTY_DDL[table_name])
        logger.info("%-15s: 0 rows (no files found)", table_name)
        return 0

    file_list = _file_list_sql(files)
    raw_cols = _RAW_COLUMNS[table_name]
    columns_sql = "{" + ", ".join(f"{k}: '{v}'" for k, v in raw_cols.items()) + "}"
    con.execute(
        f"CREATE TABLE {table_name} AS "
        f"SELECT {select_cols} "
        f"FROM read_json({file_list}, columns={columns_sql})"
    )
    count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    logger.info("%-15s: %d rows from %d file(s)", table_name, count, len(files))
    return count


# ---------------------------------------------------------------------------
# Cycles — needs deduplication and threshold parsing
# ---------------------------------------------------------------------------


def _load_cycles(con: duckdb.DuckDBPyConnection, data_dir: str) -> int:
    """
    Load cycles.jsonl, deduplicate by cycle_id (keep latest started_at),
    parse bucket_thresholds string into threshold_low / threshold_high floats,
    and serialize raw_meta as JSON string.
    """
    path = os.path.join(data_dir, "cycles.jsonl")
    if not os.path.exists(path):
        con.execute(
            "CREATE TABLE cycles ("
            "  cycle_id VARCHAR, started_at TIMESTAMPTZ, "
            "  bucket_question_id BIGINT, bucket_expiry TIMESTAMPTZ, "
            "  bucket_thresholds VARCHAR, threshold_low DOUBLE, threshold_high DOUBLE, "
            "  bucket_underlying VARCHAR, binary_outcome_id BIGINT, "
            "  binary_target_price DOUBLE, binary_expiry TIMESTAMPTZ, "
            "  raw_meta VARCHAR"
            ")"
        )
        logger.info("%-15s: 0 rows (cycles.jsonl not found)", "cycles")
        return 0

    # Timestamp fields are read as VARCHAR to avoid DuckDB auto-inference failures
    # on microsecond-precision ISO 8601 strings (format varies by sample size).
    # We cast explicitly to TIMESTAMPTZ after reading.
    con.execute(
        f"""
        CREATE TABLE cycles AS
        WITH ranked AS (
            SELECT
                cycle_id,
                started_at::TIMESTAMPTZ                                      AS started_at,
                bucket_question_id,
                bucket_expiry::TIMESTAMPTZ                                   AS bucket_expiry,
                bucket_thresholds,
                TRY_CAST(split_part(bucket_thresholds, ',', 1) AS DOUBLE)   AS threshold_low,
                TRY_CAST(split_part(bucket_thresholds, ',', 2) AS DOUBLE)   AS threshold_high,
                bucket_underlying,
                binary_outcome_id,
                binary_target_price,
                binary_expiry::TIMESTAMPTZ                                   AS binary_expiry,
                raw_meta::VARCHAR                                            AS raw_meta,
                ROW_NUMBER() OVER (
                    PARTITION BY cycle_id ORDER BY started_at DESC
                ) AS rn
            FROM read_json('{path}', columns={{
                cycle_id:            'VARCHAR',
                started_at:          'VARCHAR',
                bucket_question_id:  'BIGINT',
                bucket_expiry:       'VARCHAR',
                bucket_thresholds:   'VARCHAR',
                bucket_underlying:   'VARCHAR',
                binary_outcome_id:   'BIGINT',
                binary_target_price: 'DOUBLE',
                binary_expiry:       'VARCHAR',
                raw_meta:            'JSON'
            }})
        )
        SELECT
            cycle_id, started_at, bucket_question_id, bucket_expiry,
            bucket_thresholds, threshold_low, threshold_high,
            bucket_underlying, binary_outcome_id, binary_target_price,
            binary_expiry, raw_meta
        FROM ranked
        WHERE rn = 1
        """
    )
    count = con.execute("SELECT count(*) FROM cycles").fetchone()[0]
    logger.info("%-15s: %d rows (deduplicated)", "cycles", count)
    return count


# ---------------------------------------------------------------------------
# Outcomes map
# ---------------------------------------------------------------------------


def _load_outcomes_map(con: duckdb.DuckDBPyConnection, data_dir: str) -> int:
    path = os.path.join(data_dir, "outcomes_map.jsonl")
    if not os.path.exists(path):
        con.execute(
            "CREATE TABLE outcomes_map ("
            "  cycle_id VARCHAR, outcome_id BIGINT, role VARCHAR, "
            "  yes_coin VARCHAR, no_coin VARCHAR, description VARCHAR"
            ")"
        )
        logger.info("%-15s: 0 rows (outcomes_map.jsonl not found)", "outcomes_map")
        return 0

    con.execute(
        f"""
        CREATE TABLE outcomes_map AS
        SELECT
            cycle_id,
            outcome_id,
            role,
            yes_coin,
            no_coin,
            CAST(description AS VARCHAR) AS description
        FROM read_json_auto('{path}')
        """
    )
    count = con.execute("SELECT count(*) FROM outcomes_map").fetchone()[0]
    logger.info("%-15s: %d rows", "outcomes_map", count)
    return count


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------


def _create_indexes(con: duckdb.DuckDBPyConnection) -> None:
    for table in ("bbo", "book_levels", "trades", "perp_ctx"):
        try:
            con.execute(
                f"CREATE INDEX idx_{table}_coin_ts ON {table} (coin, ts_local)"
            )
        except duckdb.CatalogException:
            pass  # table was empty, index creation skipped gracefully


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_to_duckdb(
    data_dir: str,
    db_path: str = ":memory:",
    skip_tables: frozenset[str] = frozenset(),
) -> duckdb.DuckDBPyConnection:
    """
    Read all JSONL files in data_dir and create DuckDB tables.

    Tables created:
        - book_levels (ts_local, ts_remote, coin, side, level_idx, px, sz, n_orders)
        - trades      (ts_local, ts_remote, coin, px, sz, side, tid)
        - bbo         (ts_local, ts_remote, coin, bid_px, bid_sz, ask_px, ask_sz)
        - perp_ctx    (ts_local, coin, mark_px, mid_px, oracle_px, funding, open_interest)
        - cycles      (cycle_id, started_at, bucket_*, threshold_low, threshold_high,
                        binary_*, raw_meta)
        - outcomes_map (cycle_id, outcome_id, role, yes_coin, no_coin, description)

    Timestamps (ts_local, started_at, *_expiry) are stored as TIMESTAMPTZ (UTC).
    bucket_thresholds is kept as-is (VARCHAR) AND parsed into threshold_low/threshold_high.
    raw_meta is stored as JSON string (VARCHAR).

    skip_tables: set of table names to skip loading (created empty with their schema).
        Use to exclude large tables that exceed available RAM.
        Note: book_levels is 2.8 GB compressed on this VPS (1.9 GB RAM, no swap) and
        is not required by any of H1-H7 — skip it when RAM is constrained.

    DuckDB reads .gz files natively — no external decompression.
    Returns a DuckDB connection ready for queries.
    """
    con = duckdb.connect(db_path)

    event_tables = [
        ("bbo",         "bbo",         _BBO_COLS),
        ("book_levels", "book_levels", _BOOK_LEVELS_COLS),
        ("trades",      "trades",      _TRADES_COLS),
        ("perp_ctx",    "perp_ctx",    _PERP_CTX_COLS),
    ]

    for table_name, prefix, select_cols in event_tables:
        if table_name in skip_tables:
            con.execute(_EMPTY_DDL[table_name])
            logger.info("%-15s: skipped (in skip_tables)", table_name)
        else:
            _load_event_table(con, data_dir, table_name, prefix, select_cols)

    _load_cycles(con, data_dir)
    _load_outcomes_map(con, data_dir)
    _create_indexes(con)

    return con
