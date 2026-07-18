from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aco_pheromone_common import run_eohs  # noqa: E402


if __name__ == "__main__":
    run_eohs("cvrp", "cvrp_aco_eohs.yaml")

