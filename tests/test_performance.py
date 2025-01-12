import numpy as np

from mentevo.experiment import Experiment
from mentevo.metrics import compute_performance

from .utils import epsilon_equal

def test_performance_score():
    experiment = Experiment(number_of_agents=2,
                 number_of_tasks=2,
                 communication_graph=None,
                 task_graph=None,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 delta=1.0,
                 d=1.0,
                 tau=1.0,
                 g=None,
                 bias_value=1.0,
                 initial_state=None,
                 total_time=5,
                 nb_switches=1,
                 nb_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([5, 5]))

    # test with negative values
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([10, 10]))

    # test with all ones
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([0, 0]))

    # test when the agent got all wrong
    curves = np.array([
            # agent 1
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            # agent 2
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([-10, -10]))


def test_performance_1_switch():
    experiment = Experiment(number_of_agents=2,
                 number_of_tasks=2,
                 communication_graph=None,
                 task_graph=None,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 delta=1.0,
                 d=1.0,
                 tau=1.0,
                 g=None,
                 bias_value=0.1,
                 initial_state=None,
                 total_time=6,
                 nb_switches=2,
                 nb_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([0, 0]))

    # test with only 1 task active correctly
    curves = np.array([
            # agent 1
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            # agent 2
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([6, 6]))

    # test with only 2 task active correctly
    curves = np.array([
            # agent 1
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            # agent 2
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([12, 12]))

    # test with 1 task
    curves = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([0, 0]))

    # test when the agent got all wrong
    curves = np.array([
            # agent 1
            [-2.0, -3.0, -0.5, 1.0, 1.0, 2.0],
            [2.0, 3.0, 0.5, -1.0, -1.0, -2.0],
            # agent 2
            [-5.0, -0.1, -3.0, 2.0, 1.0, 1.0],
            [5.0, 0.1, 3.0, -2.0, -1.0, -1.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res, np.array([-19, -24.2]))