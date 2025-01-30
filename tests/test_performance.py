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
                 initial_steps=0,
                 reverse=False,
                 number_of_switches=1,
                 number_of_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([5, 5]))

    # test with negative values
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([10, 10]))

    # test with negative values, different values for each agent
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([10, 0]))

    # test with all ones
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([0, 0]))

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
    assert epsilon_equal(res[0], np.array([-10, -10]))

def test_performance_score_details_1():
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
                 initial_steps=0,
                 reverse=False,
                 number_of_switches=1,
                 number_of_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[2], np.array([
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0],
    ]))

    # test with negative values
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[2], np.array([
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
    ]))

    # test with negative values, different values for each agent
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[2], np.array([
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
    ]))

    # test with all ones
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0], 
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[2], np.array([
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [1.0, 1.0, 1.0, 1.0, 1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
    ]))

    # test when the agent got all wrong
    curves = np.array([
            # agent 1
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            # agent 2
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[2], np.array([
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
                                [-1.0, -1.0, -1.0, -1.0, -1.0],
    ]))


def test_performance_score_details_2():
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
                 initial_steps=0,
                 reverse=False,
                 number_of_switches=1,
                 number_of_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[3], np.array([5, 0, 5, 0]))

    # test with negative values
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[3], np.array([5, 5, 5, 5]))

    # test with negative values, different values for each agent
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[3], np.array([5, 5, 5, -5]))

    # test with all ones
    curves = np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[3], np.array([5, -5, 5, -5]))

    # test when the agent got all wrong
    curves = np.array([
            # agent 1
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            # agent 2
            [-1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves, detailed=True)
    assert epsilon_equal(res[3], np.array([-5, -5, -5, -5]))


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
                 bias_value=1.0,
                 initial_state=None,
                 total_time=6,
                 number_of_switches=2,
                 number_of_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([0, 0]))

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
    assert epsilon_equal(res[0], np.array([6, 6]))

    # test with 2 tasks active correctly
    curves = np.array([
            # agent 1
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            # agent 2
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([12, 12]))

    # test null with 1 task
    curves = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([0, 0]))

    # test asymmetric values and all wrongs
    curves = np.array([
            # agent 1
            [-2.0, -3.0, -0.5, 1.0, 1.0, 2.0],
            [2.0, 3.0, 0.5, -1.0, -1.0, -2.0],
            # agent 2
            [-5.0, -0.1, -3.0, 2.0, 1.0, 1.0],
            [5.0, 0.1, 3.0, -2.0, -1.0, -1.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[0], np.array([-19, -24.2]))


def test_group_performance():
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
                 total_time=6,
                 number_of_switches=2,
                 number_of_informed=None)
    # test with 1 task
    curves = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[1], 0)

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
    assert epsilon_equal(res[1], 12)

    # test with 2 tasks active correctly
    curves = np.array([
            # agent 1
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            # agent 2
            [1.0, 1.0, 1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[1], 24)

    # test null with 1 task
    curves = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[1], 0)

    # test asymmetric values and all wrongs
    curves = np.array([
            # agent 1
            [-2.0, -3.0, -0.5, 1.0, 1.0, 2.0],
            [2.0, 3.0, 0.5, -1.0, -1.0, -2.0],
            # agent 2
            [-5.0, -0.1, -3.0, 2.0, 1.0, 1.0],
            [5.0, 0.1, 3.0, -2.0, -1.0, -1.0],
    ])
    res = compute_performance(experiment, curves)
    assert epsilon_equal(res[1], -43.2)