from mentevo.utils import gaussian_g_vector, uniform_g_vector

from .utils import epsilon_equal


def test_gaussian_g_vector():
    number_of_agents = 4
    for average in [1.0, 2.0, 3.0]:
        for deviation in [0.1, 0.5, 1.0]:
            g = gaussian_g_vector(average, deviation, number_of_agents)
            assert g.shape == (number_of_agents,)
            assert epsilon_equal(g.mean(), average)
            assert g.min() >= 0


def test_uniform_g_vector():
    for average in [1.0, 2.0, 3.0]:
        for delta in [0.1, 0.5, 1.0]:
            number_of_agents = 4
            g = uniform_g_vector(average, delta, number_of_agents)
            assert g.shape == (number_of_agents,)
            assert epsilon_equal(g.mean(), average)
            assert g.min() >= 0
