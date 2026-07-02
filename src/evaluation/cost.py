import json
from typing import Dict


class CostLoader:
    """Reads per-image slicing costs from fold_i_stats.json."""

    def __init__(self, stats_path: str):
        with open(stats_path) as f:
            self._data = json.load(f)

    def test_costs(self) -> Dict[str, float]:
        """Returns {image_name: slicing_time_ms} for test split."""
        return {
            m["image_name"]: m["slicing_time_ms"]
            for m in self._data.get("test_metrics", [])
        }

    def mean_test_slicing_time(self) -> float:
        costs = list(self.test_costs().values())
        return round(sum(costs) / len(costs), 4) if costs else 0.0
