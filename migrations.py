"""
Versioned DuckDB schema migrations.

The schema is defined as an ordered list of (version, description, sql) tuples
in MIGRATIONS. `store.py` calls `apply_pending(conn)` at startup, which:
    1. Creates the _schema_version table if absent (via the first migration)
    2. Reads the current version (max of applied versions, or 0)
    3. Applies all migrations with version > current, in order
    4. Records each applied migration with timestamp

To add a new migration:
    - Append a (N+1, description, sql) tuple to MIGRATIONS
    - Never modify past entries — they have been applied in production
    - Keep schema.sql in sync for human-readable documentation
    - The authoritative source is this Python file
"""

from __future__ import annotations

import logging

import duckdb


log = logging.getLogger(__name__)


# ─── Migrations registry ───────────────────────────────────────────────────────
# Format: (version: int, description: str, sql: str)
# Versions MUST be strictly increasing unique integers starting at 1.

MIGRATIONS: list[tuple[int, str, str]] = [
    (
        1,
        "initial schema: cycles, outcomes_map, book_levels, trades, bbo, perp_ctx, raw_ctx, health_log, _schema_version",
        """
        -- Schema version tracking
        CREATE TABLE IF NOT EXISTS _schema_version (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL,
            description VARCHAR NOT NULL
        );

        -- Cycle metadata (one row per observation day)
        CREATE TABLE IF NOT EXISTS cycles (
            cycle_id VARCHAR PRIMARY KEY,
            started_at TIMESTAMP NOT NULL,
            bucket_question_id INTEGER,
            bucket_expiry TIMESTAMP,
            bucket_thresholds VARCHAR,
            bucket_underlying VARCHAR,
            binary_outcome_id INTEGER,
            binary_target_price DOUBLE,
            binary_expiry TIMESTAMP,
            raw_meta VARCHAR
        );

        -- Mapping outcome -> coin for each cycle
        CREATE TABLE IF NOT EXISTS outcomes_map (
            cycle_id VARCHAR NOT NULL,
            outcome_id INTEGER NOT NULL,
            role VARCHAR NOT NULL,
            yes_coin VARCHAR NOT NULL,
            no_coin VARCHAR NOT NULL,
            description VARCHAR,
            PRIMARY KEY (cycle_id, outcome_id)
        );

        -- L2 book updates (one row per level per snapshot)
        CREATE TABLE IF NOT EXISTS book_levels (
            ts_local TIMESTAMP NOT NULL,
            ts_remote TIMESTAMP,
            coin VARCHAR NOT NULL,
            side VARCHAR NOT NULL,
            level_idx INTEGER NOT NULL,
            px DOUBLE NOT NULL,
            sz DOUBLE NOT NULL,
            n_orders INTEGER
        );
        CREATE INDEX IF NOT EXISTS book_levels_idx ON book_levels (coin, ts_local);

        -- Trades (one row per fill)
        CREATE TABLE IF NOT EXISTS trades (
            ts_local TIMESTAMP NOT NULL,
            ts_remote TIMESTAMP,
            coin VARCHAR NOT NULL,
            px DOUBLE NOT NULL,
            sz DOUBLE NOT NULL,
            side VARCHAR,
            tid VARCHAR
        );
        CREATE INDEX IF NOT EXISTS trades_idx ON trades (coin, ts_local);

        -- BBO snapshots (compact best bid/ask)
        CREATE TABLE IF NOT EXISTS bbo (
            ts_local TIMESTAMP NOT NULL,
            ts_remote TIMESTAMP,
            coin VARCHAR NOT NULL,
            bid_px DOUBLE,
            bid_sz DOUBLE,
            ask_px DOUBLE,
            ask_sz DOUBLE
        );
        CREATE INDEX IF NOT EXISTS bbo_idx ON bbo (coin, ts_local);

        -- BTC perp context (mark, oracle, funding)
        CREATE TABLE IF NOT EXISTS perp_ctx (
            ts_local TIMESTAMP NOT NULL,
            coin VARCHAR NOT NULL,
            mark_px DOUBLE,
            mid_px DOUBLE,
            oracle_px DOUBLE,
            funding DOUBLE,
            open_interest DOUBLE
        );
        CREATE INDEX IF NOT EXISTS perp_ctx_idx ON perp_ctx (coin, ts_local);

        -- Raw active context for HIP-4 outcomes (best-effort, flexible schema)
        CREATE TABLE IF NOT EXISTS raw_ctx (
            ts_local TIMESTAMP NOT NULL,
            coin VARCHAR NOT NULL,
            sub_type VARCHAR NOT NULL,
            payload_json VARCHAR NOT NULL
        );
        CREATE INDEX IF NOT EXISTS raw_ctx_idx ON raw_ctx (coin, ts_local);

        -- Recorder health metrics
        CREATE TABLE IF NOT EXISTS health_log (
            ts TIMESTAMP NOT NULL,
            ws_connected BOOLEAN,
            n_subs_active INTEGER,
            msgs_per_sec DOUBLE,
            buffer_size INTEGER,
            last_db_flush TIMESTAMP
        );
        """,
    ),
]


# ─── Migration runner ──────────────────────────────────────────────────────────

def _table_exists(conn: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """Check whether a table exists in the current DuckDB connection."""
    result = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table_name],
    ).fetchone()
    return result is not None and result[0] > 0


def get_current_version(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the highest applied migration version, or 0 if none applied."""
    if not _table_exists(conn, "_schema_version"):
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_version").fetchone()
    return int(row[0]) if row else 0


def _validate_registry() -> None:
    """Sanity check: versions must be strictly increasing unique integers."""
    versions = [v for v, _, _ in MIGRATIONS]
    if versions != sorted(versions):
        raise RuntimeError(f"MIGRATIONS must be sorted by version, got {versions}")
    if len(versions) != len(set(versions)):
        raise RuntimeError(f"MIGRATIONS contains duplicate versions: {versions}")
    if versions and versions[0] < 1:
        raise RuntimeError(f"MIGRATIONS must start at version >= 1, got {versions[0]}")


def apply_pending(conn: duckdb.DuckDBPyConnection) -> list[int]:
    """
    Apply all migrations with version > current_version, in order.

    Returns the list of versions actually applied (empty if already up to date).
    Each migration runs in a transaction — failure rolls back cleanly.
    """
    _validate_registry()

    current = get_current_version(conn)
    applied: list[int] = []

    for version, description, sql in MIGRATIONS:
        if version <= current:
            continue

        log.info(f"applying migration {version}: {description}")
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(sql)
            conn.execute(
                "INSERT INTO _schema_version (version, applied_at, description) VALUES (?, NOW(), ?)",
                [version, description],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        applied.append(version)
        log.info(f"migration {version} applied successfully")

    if not applied:
        log.debug(f"schema already at version {current}, nothing to do")
    else:
        log.info(f"applied {len(applied)} migration(s): {applied}")

    return applied
