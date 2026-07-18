from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aco_pheromone_common import run_ow_cahd  # noqa: E402


if __name__ == "__main__":
    run_ow_cahd("cvrp", "cvrp_aco_ow_cahd.yaml")

