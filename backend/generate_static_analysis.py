"""
Pre-compute the policy-analysis bundle and emit it as a static JSON file.

This is what makes the Analysis page work on GitHub Pages, where there is no
backend. The frontend hits /api/analysis/overview when a backend is reachable
and falls back to fetching `analysis.json` from `frontend/public/` otherwise.

Usage:
    cd backend
    python generate_static_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: `python generate_static_analysis.py`
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nlp.analysis import policy_analyzer  # noqa: E402
from seed_data import SCHEMES  # noqa: E402


def main() -> None:
    out = HERE.parent / "frontend" / "public" / "analysis.json"
    print(f"Fitting analyzer on {len(SCHEMES)} schemes...")
    policy_analyzer.fit(SCHEMES)
    summary = policy_analyzer.summary()
    print(
        f"  -> {summary['n_topics_lda']} LDA topics, "
        f"{summary['n_topics_semantic']} semantic topics "
        f"(backend: {summary['semantic_backend']})"
    )
    policy_analyzer.export_to(out)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
