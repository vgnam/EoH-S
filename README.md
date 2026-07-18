# EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official implementation of [**"EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design"** ](https://arxiv.org/abs/2508.03082)accepted at **AAAI** 2026 as an **Oral** presentation.

[Paper](https://openreview.net/forum?id=JiOY4d5ktq) | [Project Page](#) | [Demo](#)

---

## 📋 Table of Contents

- [Overview](#overview)
- [OW-CAHD Extension](#ow-cahd-extension)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🔍 Overview

**EoH-S (Evolution of Heuristic Set)** introduces **Automated Heuristic Set Design (AHSD)**, a novel formulation that addresses the generalization limitations of traditional LLM-driven Automated Heuristic Design (AHD). While existing methods design a single heuristic for all problem instances, EoH-S automatically generates a small-sized complementary heuristic set where each problem instance is optimized by at least one heuristic in the set.

### Key Features

- 🎯 **Automated Heuristic Set Design (AHSD)**: Novel formulation that generates complementary heuristic sets instead of single heuristics
- 🔄 **Complementary Population Management**: Maintains diversity through specialized population management strategies
- 🧠 **Diversity-Aware Memetic Search**: Combines evolutionary search with local refinement for high-quality heuristic discovery
- 🌐 **Robust Cross-Distribution Performance**: Designed heuristics show strong generalization across instances of varying sizes and distributions

### Method Overview

![Figure 1: Overview of the EoH-S framework and Comparison to Existing LLM-driven AHD methods](https://github.com/FeiLiu36/EoH-S/raw/main/figures/framework.png)

---

## OW-CAHD Extension

**OW-CAHD (Open-World Continual Automated Heuristic Design)** extends EoH-S
from a fixed training distribution to a stream of potentially changing problem
regimes. EoH-S remains the inner heuristic-set optimizer, while OW-CAHD adds a
wake-sleep controller around it:

1. **Wake:** receive a batch of observed instances and summarize it with a
   task-specific descriptor.
2. **Regime modeling:** compare the batch with known regimes and, when
   configured, ask the LLM to synthesize executable instance generators for a
   new mixture regime.
3. **Belief update and sleep replay:** update the probability of each known
   regime, then sample a belief-weighted replay set from their generators.
4. **Continual EoH-S evolution:** rescore and warm-start from the previous full
   EoH-S population, evolve on the replay set, and preserve the population
   across rounds.
5. **Portfolio deployment:** greedily select a small complementary portfolio
   from the candidate heuristics and evaluate it on held-out ID/OOD datasets.

The repository includes reproducible open-world pipelines for constructive TSP
and CVRP. Both use separate training families and hidden ID/OOD datasets at
sizes 20, 50, and 100. The total EoH-S sampling budget is shared across
OW-CAHD rounds, making comparisons with the fixed-distribution EoH-S baseline
explicit.

### Run OW-CAHD

Set `OPENAI_API_KEY`; optionally override `OPENAI_BASE_URL` and `OPENAI_MODEL`.
Run commands from the repository root:

```bash
# TSP
python examples/training/tsp_set/run_ow_cahd.py

# CVRP
python examples/training/cvrp_set/run_ow_cahd.py
```

The main configurations are:

- `cfg/ow_cahd.yaml`: TSP OW-CAHD and hidden evaluation.
- `cfg/cvrp_ow_cahd.yaml`: CVRP OW-CAHD and hidden evaluation.
- `cfg/eohs.yaml` and `cfg/cvrp_eohs.yaml`: matched EoH-S baselines.

Each OW-CAHD run records round history, regime generators, candidate pools,
deployed portfolios, token usage, and post-training hidden utility. Dataset
generation and task-specific protocol details live in
`examples/training/tsp_set/` and `examples/training/cvrp_set/`.

---

## 📁 Repository Structure

- **`code/`**: Source code for EoH-S and baseline methods, implemented using the LLM4AD platform
- **`datasets/`**: Training and testing datasets, along with instance generation scripts
- **`examples/`**: Running scripts and configurations for different optimization tasks
- **`heuristics/`**: Final heuristics designed by EoH-S and baseline methods
- **`results/`**: Detailed experimental results, including performance metrics and logs for all methods

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher and < 3.13
- pip package manager

### Setup Instructions

1. **Clone the repository**
   
   ```bash
   git clone https://github.com/FeiLiu36/EoH-S.git
   cd eohs
   ```
   
2. **Install LLM4AD platform**
   ```bash
   cd code
   pip install .
   cd ..
   ```

3. **Set up API Endpoint, key, and LLM** in the running script (e.g., examples/obp_set/run_eohs.py)
   
   ```bash
       llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                      key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                      model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                      timeout=60)
   ```
   

---

## 🚀 Quick Start

### Running EoH-S on Online Bin Packing

```bash
cd examples
python obp_set/run_eohs.py
```

The results will be saved in the `logs/` folder.

### Running Other Tasks

```bash
# For TSP
python tsp_set/run_eohs.py

# For other tasks
python <task_name>/run_eohs.py
```

### Customizing Experiments

```python
from llm4ad.task.optimization.online_bin_packing_set import OBPSEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eohs import EoHS,EoHSProfiler

def main():

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='deepseek-v3',  # your llm, e.g., 'gpt-3.5-turbo'
                   timeout=60)

    task = OBPSEvaluation(
        timeout_seconds=120,
        dataset='./dataset_100_2k_128_5_80_training.pkl',
        return_list=True)

    method = EoHS(llm=llm,
                 profiler=EoHSProfiler(log_dir='logs/eohs', log_style='simple'),
                 evaluation=task,
                 max_sample_nums=2000,
                 max_generations=1000,
                 pop_size=10,
                 num_samplers=4,
                 num_evaluators=4,
                 debug_mode=False)

    method.run()

if __name__ == '__main__':
    main()

```

You can modify the configuration files in `examples/` to customize:

- Population size: pop_size

- Total number of sampled heuristics: max_sample_nums

- Number of parallel evaluations: num_samplers and num_evaluators

- Others: e.g., LLM type and timeout

  

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{liu2026eohs,
  title={EoH-S: Evolution of Heuristic Set using {LLM}s for Automated Heuristic Design},
  author={Fei Liu, Yilu Liu, Qingfu Zhang, Xialiang Tong, Mingxuan Yuan},
  booktitle={The Fortieth AAAI Conference on Artificial Intelligence},
  year={2026},
  url={https://openreview.net/forum?id=JiOY4d5ktq}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE] file for details.

---

## 🙏 Acknowledgments

- Built on the [LLM4AD](https://github.com/Optima-CityU/LLM4AD) platform
- Thanks to the AAAI 2026 reviewers for their valuable feedback

---

## 📧 Contact

For questions and feedback:
- Open an issue on [GitHub Issues](https://github.com/FeiLiu36/EoH-S/issues)
- Email: [fliu36-c@my.cityu.edu.hk]

---

