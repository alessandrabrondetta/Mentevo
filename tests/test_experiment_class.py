import numpy as np
from math import floor
from scipy.integrate import solve_ivp

from mentevo.experiment import Experiment
from mentevo.utils import gaussian_g_vector, uniform_g_vector, build_forward_matrix, build_cue_vector

from .utils import epsilon_equal


def test_experiment_default():

    e = Experiment()

    assert e.number_of_agents == 4
    assert e.number_of_tasks == 2
    assert epsilon_equal(e.communication_graph,  np.ones((4, 4)))
    assert e.communication_graph.shape == (4, 4)
    assert epsilon_equal (e.task_graph, np.array([[1, -1], [-1, 1]]))
    assert e.task_graph.shape == (2, 2)
    assert e.alpha == 0.03
    assert e.beta == 0.01
    assert e.gamma == 0.02
    assert e.delta == 0.0
    assert e.d == 0.2
    assert e.tau == 10
    assert e.g.shape == (4,)
    assert e.g.min() > 0
    assert epsilon_equal (e.g.mean(), 3.0, epsilon=0.01)
    assert e.bias_value == 0.1
    assert epsilon_equal (e.initial_state, np.zeros(4*2))
    assert e.initial_state.shape == (4*2,)
    assert e.total_time == 2000
    assert e.initial_steps == 0
    assert e.reverse == False
    assert e.number_of_switches == 4
    assert e.number_of_informed == 4 

    assert e.F.shape == (8,8)
    assert e.cue_vector.shape == (2000,8)
    assert e.task_switching_times.shape == (3,)
    assert e.task_switching_times[0] == 499
    assert e.task_switching_times[1] == 999
    assert e.task_switching_times[2] == 1499

    zs = e.solve()

    assert zs.shape == (8, 2000)


def test_experiment_costum():

    e = Experiment(5, 3, np.ones((5, 5)), np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]), 0.02, 0.02, 0.01, 0.01, 0.3, 12, gaussian_g_vector(4.0, 0.5, 5), 1.0, np.ones(5*3), 2700, 200, False, 5, 3)

    assert e.number_of_agents == 5
    assert e.number_of_tasks == 3
    assert epsilon_equal(e.communication_graph,  np.ones((5, 5)))
    assert e.communication_graph.shape == (5, 5)
    assert epsilon_equal (e.task_graph, np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    assert e.task_graph.shape == (3, 3)
    assert e.alpha == 0.02
    assert e.beta == 0.02
    assert e.gamma == 0.01
    assert e.delta == 0.01
    assert e.d == 0.3
    assert e.tau == 12
    assert e.g.shape == (5,)
    assert e.g.min() > 0
    assert epsilon_equal (e.g.mean(), 4.0, epsilon=0.01)
    assert e.bias_value == 1.0
    assert epsilon_equal (e.initial_state, np.ones(5*3))
    assert e.initial_state.shape == (5*3,)
    assert e.total_time == 2700
    assert e.initial_steps == 200
    assert e.reverse == False
    assert e.number_of_switches == 5
    assert e.number_of_informed == 3 

    assert e.F.shape == (15,15)
    assert e.cue_vector.shape == (2700,15)
    assert epsilon_equal (e.cue_vector[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[200], np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[700], np.array([-1, 1, -1, -1, 1, -1, -1, 1, -1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[1200], np.array([-1, -1, 1, -1, -1, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[1700], np.array([1, -1, -1, 1, -1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[2200], np.array([-1, 1, -1, -1, 1, -1, -1, 1, -1, 0, 0, 0, 0, 0, 0]))
    assert e.task_switching_times.shape == (5,)
    assert e.task_switching_times[0] == 199
    assert e.task_switching_times[1] == 699
    assert e.task_switching_times[2] == 1199
    assert e.task_switching_times[3] == 1699
    assert e.task_switching_times[4] == 2199

    zs = e.solve()

    assert zs.shape == (15, 2700)

def test_experiment_costum_reversed():

    e = Experiment(5, 3, np.ones((5, 5)), np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]), 0.02, 0.02, 0.01, 0.01, 0.3, 12, gaussian_g_vector(4.0, 0.5, 5), 1.0, np.ones(5*3), 2700, 200, True, 5, 3)

    assert e.number_of_agents == 5
    assert e.number_of_tasks == 3
    assert epsilon_equal(e.communication_graph,  np.ones((5, 5)))
    assert e.communication_graph.shape == (5, 5)
    assert epsilon_equal (e.task_graph, np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    assert e.task_graph.shape == (3, 3)
    assert e.alpha == 0.02
    assert e.beta == 0.02
    assert e.gamma == 0.01
    assert e.delta == 0.01
    assert e.d == 0.3
    assert e.tau == 12
    assert e.g.shape == (5,)
    assert e.g.min() > 0
    assert epsilon_equal (e.g.mean(), 4.0, epsilon=0.01)
    assert e.bias_value == 1.0
    assert epsilon_equal (e.initial_state, np.ones(5*3))
    assert e.initial_state.shape == (5*3,)
    assert e.total_time == 2700
    assert e.initial_steps == 200
    assert e.reverse == True
    assert e.number_of_switches == 5
    assert e.number_of_informed == 3 

    assert e.F.shape == (15,15)
    assert e.cue_vector.shape == (2700,15)
    assert epsilon_equal (e.cue_vector[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[200], np.array([-1, 1, 1, -1, 1, 1, -1, 1, 1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[700], np.array([1, -1, 1, 1, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[1200], np.array([1, 1, -1, 1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[1700], np.array([-1, 1, 1, -1, 1, 1, -1, 1, 1, 0, 0, 0, 0, 0, 0]))
    assert epsilon_equal (e.cue_vector[2200], np.array([1, -1, 1, 1, -1, 1, 1, -1, 1, 0, 0, 0, 0, 0, 0]))
    assert e.task_switching_times.shape == (5,)
    assert e.task_switching_times[0] == 199
    assert e.task_switching_times[1] == 699
    assert e.task_switching_times[2] == 1199
    assert e.task_switching_times[3] == 1699
    assert e.task_switching_times[4] == 2199

    zs = e.solve()

    assert zs.shape == (15, 2700)

# fare test generale ?


    





