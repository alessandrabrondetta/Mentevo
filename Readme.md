# Mentevo

What we need to reproduce:

- Experiment 1: lower mu help
- ...

# todo

- implement the full equations
- implement the solver
- reproduce the curve evolutions
- implement the plot functions
- implement the performance
- test the lower mu
- test the heterogeneous
- start cue vector from zero (left padding)
- implement more tests
- do the documentation
- ...
- multi-threadind
- plot as gifs
- differentiable solver (pytorch or jax)

# Getting started


```bash
pip install mentevo
```

The to use it:
```python
from mentevo import XYZ
from mentevo.plots import YYY

users = XYZ()
results = user.run(ZZZ)

plot(results)
```