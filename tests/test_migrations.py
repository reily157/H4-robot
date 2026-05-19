"""
Tests for migrations.py — versioned DuckDB schema migrations.

All tests use :memory: DuckDB to be fast and side-effect free.
"""

import pytest
import duckdb

from migrations import (
    MIGRATIONS,
    apply_pending,
    get_current_version,
    _table_exists,
    _validate_registry,
)


@pytest.fixture
def conn():
    """Fresh in-memory DuckDB connection for each test."""
    c = duckdb.connect(":memory:")
    yield c
    c.close()


# ─── Registry validation ───────────────────────────────────────────────────────

class TestRegistryValidation:
    def test_versions_sorted_and_unique(self):
        # The MIGRATIONS list itself must be valid
        _validate_registry()

    def test_versions_strictly_increasing(self):
        versions = [v for v, _, _ in MIGRATIONS]
        assert versions == sorted(versions)
        assert len(versions) == len(set(versions))

    def test_versions_start_at_1(self):
        assert MIGRATIONS[0][0] == 1


# ─── Fresh DB application ──────────────────────────────────────────────────────

class TestFreshDB:

    def test_no_schema_version_table_initially(self, conn):
        assert not _table_exists(conn, "_schema_version")

    def test_current_version_zero_on_fresh_db(self, conn):
        assert get_current_version(conn) == 0

    def test_apply_pending_creates_all_tables(self, conn):
        applied = apply_pending(conn)
        assert applied == [1]

        for table in [
            "_schema_version", "cycles", "outcomes_map",
            "book_levels", "trades", "bbo", "perp_ctx",
            "raw_ctx", "health_log",
        ]:
            assert _table_exists(conn, table), f"missing table: {table}"

    def test_version_recorded_after_apply(self, conn):
        apply_pending(conn)
        assert get_current_version(conn) == 1

    def test_applied_at_timestamp_present(self, conn):
        apply_pending(conn)
        row = conn.execute(
            "SELECT version, applied_at, description FROM _schema_version WHERE version = 1"
        ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] is not None
        assert "initial schema" in row[2].lower()


# ─── Idempotency ──────────────────────────────────────────────────────────────

class TestIdempotency:

    def test_apply_pending_twice_does_nothing(self, conn):
        applied1 = apply_pending(conn)
        applied2 = apply_pending(conn)
        assert applied1 == [1]
        assert applied2 == []

    def test_version_stays_at_1_after_double_apply(self, conn):
        apply_pending(conn)
        apply_pending(conn)
        assert get_current_version(conn) == 1

    def test_no_duplicate_rows_in_schema_version(self, conn):
        apply_pending(conn)
        apply_pending(conn)
        rows = conn.execute("SELECT COUNT(*) FROM _schema_version").fetchone()
        assert rows[0] == 1


# ─── Table existence helper ───────────────────────────────────────────────────

class TestTableExists:

    def test_returns_false_for_missing(self, conn):
        assert not _table_exists(conn, "nonexistent_table")

    def test_returns_true_after_creation(self, conn):
        conn.execute("CREATE TABLE foo (x INTEGER)")
        assert _table_exists(conn, "foo")


# ─── Smoke test: insertion into each table works after migration ───────────────

class TestSchemaInsertable:
    """After migration, each table should accept a representative row."""

    def setup_method(self, method):
        self.conn = duckdb.connect(":memory:")
        apply_pending(self.conn)

    def teardown_method(self, method):
        self.conn.close()

    def test_cycles_insertable(self):
        self.conn.execute(
            "INSERT INTO cycles (cycle_id, started_at) VALUES (?, ?)",
            ["20260520", "2026-05-20 06:00:00"],
        )
        row = self.conn.execute("SELECT cycle_id FROM cycles").fetchone()
        assert row[0] == "20260520"

    def test_outcomes_map_insertable(self):
        self.conn.execute(
            "INSERT INTO outcomes_map (cycle_id, outcome_id, role, yes_coin, no_coin) "
            "VALUES (?, ?, ?, ?, ?)",
            ["20260520", 67, "bucket_idx_0", "#670", "#671"],
        )

    def test_book_levels_insertable(self):
        self.conn.execute(
            "INSERT INTO book_levels (ts_local, coin, side, level_idx, px, sz) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ["2026-05-20 06:00:00", "#670", "bid", 0, 0.155, 100.0],
        )

    def test_trades_insertable(self):
        self.conn.execute(
            "INSERT INTO trades (ts_local, coin, px, sz) VALUES (?, ?, ?, ?)",
            ["2026-05-20 06:00:00", "#670", 0.155, 10.0],
        )

    def test_bbo_insertable(self):
        self.conn.execute(
            "INSERT INTO bbo (ts_local, coin) VALUES (?, ?)",
            ["2026-05-20 06:00:00", "#670"],
        )

    def test_perp_ctx_insertable(self):
        self.conn.execute(
            "INSERT INTO perp_ctx (ts_local, coin, mark_px) VALUES (?, ?, ?)",
            ["2026-05-20 06:00:00", "BTC", 67000.0],
        )

    def test_raw_ctx_insertable(self):
        self.conn.execute(
            "INSERT INTO raw_ctx (ts_local, coin, sub_type, payload_json) "
            "VALUES (?, ?, ?, ?)",
            ["2026-05-20 06:00:00", "#670", "activeAssetCtx", "{}"],
        )

    def test_health_log_insertable(self):
        self.conn.execute(
            "INSERT INTO health_log (ts) VALUES (?)",
            ["2026-05-20 06:00:00"],
        )
