"""Version consistency: runtime __version__ must match pyproject.toml.

Drift found after the 0.5.2 release (pip metadata said 0.5.2 while
``hufu.__version__`` still read 0.4.0) — pinned so it cannot recur.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import hufu


class VersionConsistencyTests(unittest.TestCase):
    def test_runtime_version_matches_pyproject(self) -> None:
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        match = re.search(
            r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), flags=re.M
        )
        self.assertIsNotNone(match)
        self.assertEqual(hufu.__version__, match.group(1))

    def test_citation_version_matches(self) -> None:
        cff = Path(__file__).parent.parent / "CITATION.cff"
        match = re.search(r"^version:\s*(\S+)", cff.read_text(encoding="utf-8"), flags=re.M)
        self.assertIsNotNone(match)
        self.assertEqual(hufu.__version__, match.group(1))


if __name__ == "__main__":
    unittest.main()
