import numpy as np
from mentevo.utils import build_cue_vector

from .utils import epsilon_equal


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
