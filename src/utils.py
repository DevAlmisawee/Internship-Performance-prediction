"""
utils.py
--------
Small shared utility helpers used across the project (logging setup,
path resolution, JSON I/O helpers). Keeping these in one place avoids
duplicating boilerplate across train.py / evaluate.py / predict.py / app.py.
"""

import os
import json
import logging


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a standardized logger for use across project modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def save_json(obj, path: str):
    """Save a Python object as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str):
    """Load a JSON file into a Python object."""
    with open(path) as f:
        return json.load(f)
