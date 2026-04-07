from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-slow-model",
        action="store_true",
        default=False,
        help="collect and run resource-intensive model tests",
    )


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    if config.getoption("--run-slow-model"):
        return False

    return collection_path.name in {"test_models.py", "test_efficientnet.py"}
