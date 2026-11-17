"""Shared helpers: configuration, logging, seeding and file paths."""
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

LOG = logging.getLogger("gnn_oulad")


def load_config(path="config.yaml", **overrides):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(Path(output_dir) / "run_log.txt", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    return LOG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        LOG.info("%s finished in %.1f s", self.label, time.time() - self.start)
