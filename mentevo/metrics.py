import numpy as np

def compute_performance(experiment, simulation_results, detailed=False):
    """
    Compute the performance of the agents in the experiment using the simulation results.
    The metric used is the dot product between the sign of cue vector and the simulation results, 
    indeed counting in a positive way the areas where the agent is focusing more on the correct task and in a negative way 
    the areas where the agent is doing the wrong task. 
    The individual performance is the sum of the scores of each agent on both tasks.
    The group performance is simply the sum of the scores of all agents.

    Parameters
    ---------- 
    experiment : Experiment class object
        The experiment object that generated the simulation_results.
    simulation_results : 2D numpy array
        The simulation results used to compute the performance. 
        The shape should be (number_of_agents * number_of_tasks, total_time). 
        The order is [agent1_task1, agent1_task2, agent2_task1, agent2_task2, ...]
        in case of two agents and two tasks.
    detailed : bool, optional
        Whether to return detailed information about the performance (performance values at
        each time step). The default is False.

    Returns
    -------
    individual_performance : 1D numpy array
        The performance of each agent. The shape is (number_of_agents,).
    group_performance : float
        The performance of the group.
    detailed_score.T : 2D numpy array
        The performance of each agent, at each time step, on each task. 
        The shape is (number_of_agents*number_of_tasks, total_time).
        This is returned only if detailed=True.  
    individual_performance_t : 1D numpy array
        The performance of each agent on the two different tasks, sum over time. 
        The shape is (number_of_agents*number_of_tasks,).
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
    
    # compute the score using labels and simulation results
    detailed_score = labels * simulation_results.T
    individual_performance_t = np.sum(detailed_score, 0) # sum over time
    individual_performance = individual_performance_t.reshape(na, 2).sum(1) # sum over tasks

    # compute group performance
    group_performance = individual_performance.sum() # sum over agents

    if detailed:
        return individual_performance, group_performance, detailed_score.T, individual_performance_t

    return individual_performance, group_performance