import json
import os
from datetime import datetime

def log_result(results_dir: str, name: str, payload: dict):
    """Append a timestamped result dict to <results_dir>/<name>.json (creates if absent)."""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    payload = {"timestamp": datetime.utcnow().isoformat(), **payload}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
