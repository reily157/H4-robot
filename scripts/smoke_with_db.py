"""Smoke test with persistent DB to verify data is actually written."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recorder import Recorder

DB_PATH = "data/smoke_check.duckdb"

async def main():
    Path("data").mkdir(exist_ok=True)
    # Remove old empty DB if it exists
    db_file = Path(DB_PATH)
    if db_file.exists():
        db_file.unlink()
    
    rec = Recorder(db_path=DB_PATH, health_port=0)
    await rec.init()
    print(f"Running 60s recorder with persistent DB at {DB_PATH}...")
    await rec.run(duration_s=60)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())