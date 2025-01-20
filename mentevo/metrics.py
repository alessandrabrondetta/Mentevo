import numpy as np

def compute_performance(experiment, simulation_results, detailed=False):
    """
    Compute the performance of the agents in the experiment using the simulation results.
    The metric used is the dot product between the sign of cue vector and the simulation results, 
    which is the score of the agent. This means counting in a positive way the areas where the agent 
    is focusing more on the correct task and in a negative way the areas where the agent is 
    doing the wrong task. The group performance is simply the sum of the scores of all agents.

    Parameters
    ----------
    experiment : Experiment
        The experiment object that generated the simulation_results.
    simulation_results : 2D numpy array
        The simulation results used to compute the performance. 
        The shape should be (number_of_agents * number_of_tasks, total_time).
    detailed : bool, optional
        Whether to return detailed information about the performance (performance values at
        each time step). The default is False.

    Returns
    -------
    individual_performance : 1D numpy array
        The performance of each agent. The shape is (number_of_agents,).
    group_performance : float
        The performance of the group.
    detailed_score : 2D numpy array
        The performance of each agent, at each time step, on each task. 
        The shape is (total_time, number_of_agents*number_of_tasks).
        This is returned only if detailed=True.     
    
    """
    assert isinstance(simulation_results, np.ndarray), 'simulation_results must be a numpy array'

    na = experiment.number_of_agents
    no = experiment.number_of_tasks

    assert no == 2, 'this function works only for number_of_tasks = 2'
    assert simulation_results.shape == (na * no, experiment.total_time), 'simulation_results has the right shape'

    # use the cue vector to measure the performance
    labels = np.sign(experiment.cue_vector)
    assert labels.shape == (experiment.total_time, na * no), 'cue_vector has the right shape'
    
    # compute the score using labels and curves
    detailed_score = labels * simulation_results.T
    individual_performance = np.sum(detailed_score, 0)
    individual_performance = individual_performance.reshape(na, 2).sum(1)

    # compute group performance
    group_performance = individual_performance.sum()

    if detailed:
        return individual_performance, group_performance, detailed_score

    return individual_performance, group_performance