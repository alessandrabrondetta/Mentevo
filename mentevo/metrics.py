import numpy as np

def compute_performance(experiment, curves):
    """
    todo: add docstring
    """
    assert isinstance(curves, np.ndarray)

    na = experiment.number_of_agents
    no = experiment.number_of_tasks

    assert no == 2, 'this function works only for number_of_tasks = 2'

    # use the cue vector to measure the performance
    labels = np.sign(experiment.cue_vector)
    assert labels.shape == (experiment.total_time, 2 * na)
    assert curves.shape == (2 * experiment.number_of_agents, experiment.total_time)

    # compute the score using labels and curves
    score = np.sum(labels * curves.T, 0)
    score = score.reshape(na, 2).sum(1)

    return score