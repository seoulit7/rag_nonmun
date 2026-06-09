#!/usr/bin/env python
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8")

IDX = sorted({3})


def main() -> int:
    mod = __import__("test_fallback_t0_indices_condition_c", fromlist=["*"])
    mod.FALLBACK_QUERY_INDICES_T0_COND_C = IDX
    return mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
