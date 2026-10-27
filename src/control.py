"""Budget and process control shared by the Kaggle experiment."""

import json
import math
import os
import time
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def choose_updates(seconds_per_update, remaining_seconds, maximum=2000, completed=50):
    """Same target for both experiments; retain 20% for validation and saves."""
    if seconds_per_update <= 0 or not math.isfinite(seconds_per_update):
        raise ValueError("Invalid pilot timing")
    if remaining_seconds <= 0:
        return completed
    affordable = completed + int(0.8 * remaining_seconds / seconds_per_update)
    target = min(maximum, affordable)
    return max(completed, (target // 250) * 250)


def init_budget(path, cap_seconds):
    """Restarting within the same output directory cannot reset the deadline."""
    path = Path(path)
    if path.exists():
        state = json.loads(path.read_text())
        state["deadline"] = min(state["deadline"], state["started_at"] + cap_seconds)
    else:
        started = time.time()
        state = {"started_at": started, "deadline": started + cap_seconds}
    atomic_json(path, state)
    return state


def budget_left(state, reserve=0):
    return max(0.0, state["deadline"] - time.time() - reserve)
