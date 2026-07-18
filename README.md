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

### Method in detail

#### 1. Wake stream and instance representation

At continual round `t`, the controller receives a wake batch
`B_t = {x_1, ..., x_m}`. A task-specific descriptor `d(x)` maps every instance
to a fixed-dimensional vector. The batch observation is the mean descriptor:

```text
phi_t = (1 / |B_t|) * sum(x in B_t) d(x).
```

The current runners use:

- **TSP (8 dimensions):** coordinate center and spread, plus mean, standard
  deviation, 10th percentile, and 90th percentile of pairwise distances.
- **CVRP (12 dimensions):** the eight geometric TSP statistics, customer-demand
  mean and standard deviation, total-demand/capacity ratio, and customer count.

Descriptors are used only by the open-world controller. Heuristics are still
evaluated on complete optimization instances.

#### 2. Regime observation model and novelty

Each known regime `z` stores one or more Gaussian descriptor components
`(mu_zk, Sigma_zk)` and an executable generator. Covariance matrices are
regularized by `covariance_reg`. For a wake observation, novelty is the nearest
squared Mahalanobis distance over all regimes and mixture components:

```text
D_t = min(z, k) (phi_t - mu_zk)^T pinv(Sigma_zk) (phi_t - mu_zk).
```

The novelty threshold is the `(1 - alpha)` quantile of a chi-square distribution
with `dim(phi_t)` degrees of freedom. A regime can be proposed after novelty is
observed for `novelty_confirm_rounds` consecutive rounds. An optional RBF-MMD
permutation test checks whether generated and observed descriptor samples could
come from the same distribution before accepting that regime.

The released TSP/CVRP experiment configs set `always_synthesize_regime: true`.
Consequently, they synthesize and accept one new regime per wake round; novelty
is still measured and logged, but it does not gate synthesis. They also set
`skip_mmd_verification: true`. These are experimental choices rather than
limitations of the controller.

#### 3. LLM-based mixture regime synthesis

To model a new regime, OW-CAHD asks the LLM for executable Python functions
named `generate_instance`. Each prompt contains a deterministic random subset
of observed instances, their descriptors, complete coordinates, task context,
and any error returned by the previous attempt.

The controller can request `K` generator hypotheses. Components use different
temperatures and structural focuses such as global geometry, clustering,
anisotropy, holes, multimodality, outliers, manifolds, and heterogeneous local
mechanisms. Every returned program is parsed and compiled, then must:

- define a callable `generate_instance` function;
- return exactly the requested number of samples;
- pass the task-specific instance validator; and
- successfully generate the samples used to fit its observation model.

Successful components form an equal-weight executable mixture. Sampling cycles
across components and randomizes the resulting order. For component `k`, the
controller fits `mu_zk` and `Sigma_zk` using both wake examples and generated
fit samples. If too few components succeed, synthesis fails; a bootstrap
resampling fallback is available through configuration.

#### 4. Bayesian belief tracking

Let `b_(t-1)(z)` be the previous belief over regimes. OW-CAHD first applies a
sticky transition model. With `rho = sticky_transition`, the self-transition
probability is `rho` and the remaining mass is distributed uniformly over the
other regimes:

```text
pred_t = T^T b_(t-1)
b_t(z) proportional to pred_t(z) * q_z(phi_t),
```

where `q_z` is the Gaussian or Gaussian-mixture descriptor likelihood. The
implementation evaluates mixture likelihoods with log-sum-exp and uses
pseudoinverses for numerical robustness. After normalization, `belief_floor`
keeps every known regime represented and reduces catastrophic forgetting.

When a new regime is accepted, it initially receives `new_regime_blend` belief
mass; the previous beliefs share the remainder.

#### 5. Sleep replay construction

OW-CAHD builds a synthetic sleep set to rehearse known regimes before evolving
the heuristic population. For each regime, it samples valid instances and
estimates an information-gain value:

```text
g_z = mean_x KL(p(regime | d(x)) || b_t),  x sampled from regime z
a_z = softmax(g_z / allocation_temperature).
```

The allocation `a_z` determines how many synthetic instances are drawn from
each regime, subject to `min_sleep_per_regime`. Invalid generated samples are
retried and can be replaced from the regime archive. Although information gain
controls sample counts, the evaluation weight of each retained instance is:

```text
w_i proportional to b_t(z) / n_z,
```

so the total optimization weight of a regime follows the current belief rather
than the number of samples allocated to it.

#### 6. Continual EoH-S evolution

The complete EoH-S population from the preceding round is rescored on the new
sleep set. Valid functions warm-start the next EoH-S run, allowing evolution to
continue instead of restarting from only the deployed portfolio. If an inner
run fails completely, OW-CAHD retains the previous population.

A single global sample budget is shared by all rounds. If `total_rounds` is
known, the next allowance is:

```text
round_budget = ceil((total_budget - samples_used) / rounds_remaining).
```

Otherwise the inner run may use the remaining budget, subject to its generation
limit. The controller stops once the global `max_sample_nums` budget is reached.

#### 7. Complementary portfolio selection

Each heuristic has a score vector over the sleep instances. Scores are
min-max-normalized separately for every instance to obtain utilities `u_hi`.
For a portfolio `P`, its weighted coverage objective is:

```text
F(P) = sum_i w_i * max(h in P) u_hi.
```

Starting from an empty set, OW-CAHD greedily adds the candidate with the largest
increase in `F(P)` until `portfolio_size` functions have been selected. This
retains complementary specialists: a heuristic is useful when it improves
coverage on instances not already handled well by the selected set.

#### 8. Hidden ID/OOD evaluation

Training and regime construction never use hidden test instances. After the
sample budget is exhausted, the final portfolio is evaluated on separate ID
and OOD files at each problem size. For every instance, evaluation takes the
best valid score achieved by any portfolio member. Constructive-task utility is
the relative gap to a centrally computed reference:

```text
utility = (reference_cost - heuristic_cost) / reference_cost.
```

Higher utility is better; zero matches the reference and a negative value is a
percentage excess cost. TSP uses LKH/elkai with nearest-neighbor fallback as its
reference. CVRP uses Clarke-Wright Savings followed by route-level 2-opt.

#### Algorithm summary

```text
Input: wake stream, descriptor d, total EoH-S budget, portfolio size M
Initialize regimes = {}, belief = {}, population = {}, portfolio = {}

for each wake batch B_t:
    compute phi_t and novelty statistic
    synthesize/verify/accept a regime when configured
    update Bayesian belief over all regimes
    allocate and generate the weighted sleep replay set
    rescore the inherited full population on replay
    run budgeted EoH-S, warm-started by valid inherited functions
    preserve the resulting full population
    greedily select an M-function complementary portfolio
    log regimes, beliefs, budgets, candidates, portfolio, and token usage
    stop when the global sample budget is exhausted

return final portfolio
```

#### Reference experimental settings

| Parameter | TSP/CVRP value | Role |
| --- | ---: | --- |
| Wake batch size | 32 | Observed instances per round |
| Sleep instances | 128 | Replay instances per round |
| Minimum per regime | 8 | Prevents a known regime from disappearing |
| Mixture components | 10 | LLM generator hypotheses per new regime |
| Prompt examples | 4 | Observed instances shown per synthesis call |
| Fit samples per component | 64 | Fits descriptor mean/covariance |
| Covariance regularization | 0.0001 | Stabilizes Gaussian models |
| Belief floor | 0.02 | Retains probability mass for old regimes |
| Sticky transition | 0.92 | Prior probability of staying in a regime |
| Allocation temperature | 0.5 | Controls information-gain allocation sharpness |
| Portfolio size | 10 | Number of deployed complementary heuristics |
| Total EoH-S samples | 500 | Shared continual-search budget |
| Inner generations | 8 | Maximum generations per round |
| EoH-S population | 10 | Full inherited evolutionary population |

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

