import numpy as np
import matplotlib.pyplot as plt


def plot_curves(experiment, curves):
    """
    todo: docstring
    """
    assert isinstance(curves, np.ndarray)

    na = experiment.number_of_agents
    no = experiment.number_of_tasks

    assert no == 2, 'this function works only for number_of_tasks = 2'

    # plot the curves
    # TO DO (Ale): refactor with performance funcion
    curves = [(curves[i*2] - curves[i*2+1]) / 2 for i in range(na)]
    for i in range(na):
        plt.plot(curves[i])

    # plot curves with lines
    for l in experiment.task_switching_times:
        plt.axvline(x=l, linestyle="--", color='black')
