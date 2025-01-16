import numpy as np
from mentevo.utils import build_cue_vector
from mentevo.utils import build_cue_vector_zero

from .utils import epsilon_equal


def test_build_cue_vector_zero():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 6
    v = build_cue_vector_zero(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 3 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
    ]))


def test_build_cue_vector_reversed_zero():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 6
    v = build_cue_vector_zero(na, nt, na, n_switches, total_time, reversed=True)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 2 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 3 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
    ]))


def test_build_cue_vector_3_tasks_zero():
    na = 2
    nt = 3
    n_switches = 3
    total_time = 6
    v = build_cue_vector_zero(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, -1, 1, -1, -1],
        [1, -1, -1, 1, -1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, -1, 1, -1],
        [-1, 1, -1, -1, 1, -1],
        # switch 3 (task 3)
        [-1, -1, 1, -1, -1, 1],
        [-1, -1, 1, -1, -1, 1],
    ]))


def test_build_cue_vector_half_agent_zero():
    na = 2
    nt = 2
    nb_informed = 1
    n_switches = 3
    total_time = 6

    v = build_cue_vector_zero(na, nt, nb_informed, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
        # switch 2 (task 2)
        [-1, 1, 0, 0],
        [-1, 1, 0, 0],
        # switch 3 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
    ]))


def test_zero_agent_zero():
    na = 2
    nt = 2
    nb_informed = 0
    n_switches = 3
    total_time = 6

    v = build_cue_vector_zero(na, nt, nb_informed, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.zeros((total_time, na * nt)))


def test_scaling_zero():
    na = 100
    nt = 2
    n_switches = 4
    total_time = 1_000

    v = build_cue_vector_zero(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)

    # check that the first switch is the same for all agents
    for i in range(249):
        assert epsilon_equal(v[0], v[i])

    # check that the second switch is the same for all agents
    for i in range(250, 499):
        assert epsilon_equal(v[250], v[i])

    # check that the third switch is the same for all agents
    for i in range(500, 749):
        assert epsilon_equal(v[500], v[i])

    # check that the fourth switch is the same for all agents
    for i in range(750, 999):
        assert epsilon_equal(v[750], v[i])

    # now check that first and second switch are different
    assert not epsilon_equal(v[0], v[250])

    # now check that the first and third switch are the same
    assert epsilon_equal(v[0], v[500])

    # now check that the first and fourth switch are the same
    assert epsilon_equal(v[250], v[750])

    # check the first one activate task 1 only
    assert np.all(v[0] == np.array([1, -1] * na))

    # check the second one activate task 2 only
    assert np.all(v[250] == np.array([-1, 1] * na))


#------------------------------------------------------------

def test_build_cue_vector():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 6
    v = build_cue_vector(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 3 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
    ]))


def test_build_cue_vector_reversed():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 6
    v = build_cue_vector(na, nt, na, n_switches, total_time, reversed=True)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 2 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 3 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
    ]))


def test_build_cue_vector_3_tasks():
    na = 2
    nt = 3
    n_switches = 3
    total_time = 6
    v = build_cue_vector(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, -1, 1, -1, -1],
        [1, -1, -1, 1, -1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, -1, 1, -1],
        [-1, 1, -1, -1, 1, -1],
        # switch 3 (task 3)
        [-1, -1, 1, -1, -1, 1],
        [-1, -1, 1, -1, -1, 1],
    ]))


def test_build_cue_vector_half_agent():
    na = 2
    nt = 2
    nb_informed = 1
    n_switches = 3
    total_time = 6

    v = build_cue_vector(na, nt, nb_informed, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
        # switch 2 (task 2)
        [-1, 1, 0, 0],
        [-1, 1, 0, 0],
        # switch 3 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
    ]))


def test_zero_agent():
    na = 2
    nt = 2
    nb_informed = 0
    n_switches = 3
    total_time = 6

    v = build_cue_vector(na, nt, nb_informed, n_switches, total_time)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.zeros((total_time, na * nt)))


def test_scaling():
    na = 100
    nt = 2
    n_switches = 4
    total_time = 1_000

    v = build_cue_vector(na, nt, na, n_switches, total_time)

    assert v.shape == (total_time, na * nt)

    # check that the first switch is the same for all agents
    for i in range(249):
        assert epsilon_equal(v[0], v[i])

    # check that the second switch is the same for all agents
    for i in range(250, 499):
        assert epsilon_equal(v[250], v[i])

    # check that the third switch is the same for all agents
    for i in range(500, 749):
        assert epsilon_equal(v[500], v[i])

    # check that the fourth switch is the same for all agents
    for i in range(750, 999):
        assert epsilon_equal(v[750], v[i])

    # now check that first and second switch are different
    assert not epsilon_equal(v[0], v[250])

    # now check that the first and third switch are the same
    assert epsilon_equal(v[0], v[500])

    # now check that the first and fourth switch are the same
    assert epsilon_equal(v[250], v[750])

    # check the first one activate task 1 only
    assert np.all(v[0] == np.array([1, -1] * na))

    # check the second one activate task 2 only
    assert np.all(v[250] == np.array([-1, 1] * na))


#------------------------------------------------------------

def test_build_cue_vector_t0_zero():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 6
    t0 = 0
    v = build_cue_vector(na, nt, na, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert epsilon_equal(v, np.array([
        # switch 1 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 3 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
    ]))

def test_build_cue_vector_t0():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 8
    t0 = 2
    v = build_cue_vector(na, nt, na, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 6
    assert epsilon_equal(v, np.array([
        # switch 0 
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        # switch 1 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 3 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
    ]))


def test_build_cue_vector_reversed_t0():
    na = 2
    nt = 2
    n_switches = 3
    total_time = 8
    t0 = 2
    v = build_cue_vector(na, nt, na, n_switches, total_time, t0, reversed=True)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 6
    assert epsilon_equal(v, np.array([
        # switch 0 
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        # switch 1 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
        # switch 2 (task 1)
        [1, -1, 1, -1],
        [1, -1, 1, -1],
        # switch 3 (task 2)
        [-1, 1, -1, 1],
        [-1, 1, -1, 1],
    ]))


def test_build_cue_vector_3_tasks_t0():
    na = 2
    nt = 3
    n_switches = 3
    total_time = 8
    t0 = 2
    v = build_cue_vector(na, nt, na, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 6
    assert epsilon_equal(v, np.array([
        # switch 0 
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        # switch 1 (task 1)
        [1, -1, -1, 1, -1, -1],
        [1, -1, -1, 1, -1, -1],
        # switch 2 (task 2)
        [-1, 1, -1, -1, 1, -1],
        [-1, 1, -1, -1, 1, -1],
        # switch 3 (task 3)
        [-1, -1, 1, -1, -1, 1],
        [-1, -1, 1, -1, -1, 1],
    ]))


def test_build_cue_vector_half_agent_t0():
    na = 2
    nt = 2
    nb_informed = 1
    n_switches = 3
    total_time = 8
    t0 = 2

    v = build_cue_vector(na, nt, nb_informed, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 6
    assert epsilon_equal(v, np.array([
        # switch 0 
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        # switch 1 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
        # switch 2 (task 2)
        [-1, 1, 0, 0],
        [-1, 1, 0, 0],
        # switch 3 (task 1)
        [1, -1, 0, 0],
        [1, -1, 0, 0],
    ]))


def test_zero_agent_t0():
    na = 2
    nt = 2
    nb_informed = 0
    n_switches = 3
    total_time = 8
    t0 = 2

    v = build_cue_vector(na, nt, nb_informed, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 6
    assert epsilon_equal(v, np.zeros((total_time, na * nt)))


def test_scaling_t0():
    na = 100
    nt = 2
    n_switches = 4
    total_time = 1_200
    t0 = 200

    v = build_cue_vector(na, nt, na, n_switches, total_time, t0)

    assert v.shape == (total_time, na * nt)
    assert total_time - t0 == 1_000

    # check that the zero switch is the same for all agents
    for i in range(199):
        assert epsilon_equal(v[0], v[i])

    # check that the first switch is the same for all agents
    for i in range(200, 449):
        assert epsilon_equal(v[200], v[i])

    # check that the second switch is the same for all agents
    for i in range(450, 699):
        assert epsilon_equal(v[450], v[i])

    # check that the third switch is the same for all agents
    for i in range(700, 949):
        assert epsilon_equal(v[700], v[i])

    # check that the fourth switch is the same for all agents
    for i in range(950, 1199):
        assert epsilon_equal(v[950], v[i])

    # now check that first and second switch are different
    assert not epsilon_equal(v[200], v[450])

    # now check that the first and third switch are the same
    assert epsilon_equal(v[200], v[700])

    # now check that the first and fourth switch are the same
    assert epsilon_equal(v[450], v[950])

    # check the first one activate task 1 only
    assert np.all(v[200] == np.array([1, -1] * na))

    # check the second one activate task 2 only
    assert np.all(v[450] == np.array([-1, 1] * na))

    # check the zero one activate no task 
    assert np.all(v[0] == np.array([0, 0] * na))

