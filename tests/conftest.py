"""
Shared pytest configuration.

Sets asyncio_mode = "auto" so we don't need to decorate every async test.
Adds the project root to sys.path so tests can `from store import Store`
without an explicit package layout.
"""

import sys
from pathlib import Path

# Add project root to sys.path so `from migrations import ...` works
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
