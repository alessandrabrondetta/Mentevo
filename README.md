
<div align="center">
    <img src="assets/banner.png" width="75%" alt="Mentevo logo" align="center" />
</div>

<div align="center">
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.9, 3.10, 3.11-efefef">
    </a>
    <a href="https://github.com/alessandrabrondetta/Mentevo/actions/workflows/tests.yml/badge.svg">
        <img alt="Tests" src="https://github.com/alessandrabrondetta/Mentevo/actions/workflows/tests.yml/badge.svg">
    </a>
    <a href="https://github.com/alessandrabrondetta/Mentevo/actions/workflows/publish.yml/badge.svg">
        <img alt="Pypi" src="https://github.com/alessandrabrondetta/Mentevo/actions/workflows/publish.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/Mentevo">
        <img alt="Pepy" src="https://static.pepy.tech/badge/Mentevo">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>

<p align="center">
    <a href="https://alessandrabrondetta.github.io/Mentevo/">
        📚 Explore the docs »
    </a>
</p>

 ---

👋  Mentevo is a compact library designed for studying the dynamic of balancing cognitive stability and flexibility in groups of agents, initially providing the implementation code for the research paper of [Brondetta et al, 2023](https://escholarship.org/uc/item/6b47b61g).

This repository also introduces various parametrization, visualization methods as well as metrics to compute performances of each agents. However, Mentevo emphasizes experimentation and is not an official reproduction of any other paper aside from Brondetta et al.

# Getting Started

To start with Mentevo, we propose multiple notebook that will help you familiarize with the library


- Starter [![Open](https://img.shields.io/badge/Starter-Notebook-green?style=flat&logo=jupyter)](notebooks/starter.ipynb)
- Study of optimal gain value depending on the task switching rate [![Open](https://img.shields.io/badge/Starter-Notebook-green?style=flat&logo=jupyter)](notebooks/optimal_g_homogenous.ipynb)
- Performance in details [![Open](https://img.shields.io/badge/Starter-Notebook-green?style=flat&logo=jupyter)](notebooks/performance.ipynb)
- Partially informed agents [![Open](https://img.shields.io/badge/Starter-Notebook-green?style=flat&logo=jupyter)](notebooks/partial_informed.ipynb)


Otherwise, you can simply start hacking with mentevo, it's as simple as:

```python
from mentevo import (Experiment, compute_performance, plot_curves)

# create an experiment object
experiment = Experiment(nb_agents=4)
simulation_results = experiment.solve()

# plots the simulation results
plot_curves(experiment, simulation_results)

# compute the performance
scores = compute_performance(experiment, simulation_results)
print('individual performance', scores[0])
print('group performance', scores[1])
```

When optimizing, it's crucial to fine-tune the hyperparameters. Parameters like the alpha, beta, d or tau significantly impact the output. We recommend ajusting the values according to the original paper to ensure comparable results.

# Citation

```
@inproceedings{brondetta2024benefits,
  title={On the Benefits of Heterogeneity in Cognitive Stability and Flexibility for Collaborative Task Switching},
  author={Brondetta, Alessandra and Bizyaeva, Anastasia and Lucas, Maxime and Petri, Giovanni and Musslick, Sebastian},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={46},
  year={2024}
}
```

# Authors

- Alessandra Brondetta - alessandra.brondetta@uni-osnabrueck.de, Candidate PhD Student under the supervision of Prof. Sebastian Musslick, Automated Scientific Discovery of Mind and Brain, Osnabrück University.