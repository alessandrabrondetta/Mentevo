import numpy as np


def gaussian_g_vector(average, deviation, number_of_agents):
    """
    Create a vector of g values following Gaussian distribution.
    The average value of the vector is forced to be the average value given as input.

    Parameters
    ----------
    average : float
        The average value of the Gaussian distribution.
        The average should be greater than 0.0.
    deviation : float
        The standard deviation of the Gaussian distribution.
        The deviation should be non-negative.
    number_of_agents : int
        The number of agents in the system. 
        The number of agents should be greater than 0.
    
    Returns
    -------
    g : 1D numpy array
        A numpy array of size (number_of_agents,) with the g values for all agents in the system.
        All g values are non-negative. The average value of the g vector is equal to the average value given as input.          
    """
    assert average > 0, 'average should be greater than 0'
    assert deviation >= 0, 'deviation should be non-negative'
    assert number_of_agents > 0, 'number_of_agents should be greater than 0'

    g = np.random.normal(average, deviation, number_of_agents)
    g = np.clip(g, 0, None)
    # used range in Brondetta et al. 2024
    g[g == 0] = np.random.uniform(0.5, 8.5, size=np.sum(g == 0))
    g = g / (np.mean(g) + 1e-6)
    g = g * average
    return g


def uniform_g_vector(average, delta, number_of_agents):
    """
    Create a vector of g values following uniform distribution.
    The average value of the vector is forced to be the average value given as input.
    
    Parameters
    ----------
    average : float
        The average value of the uniform distribution.
        The average should be greater than 0.0.
    delta : float
        The deviation of the uniform distribution.
        The deviation should be non-negative. The deviation should be less or equal to the average.
    number_of_agents : int
        The number of agents in the system.
        The number of agents should be greater than 0.
    
    Returns
    -------
    g : 1D numpy array
        A numpy array of size (number_of_agents,) with the g values for all agents in the system. 
        All g values are non-negative. The average value of the g vector is equal to the average value given as input.       
    """

    assert average > 0, 'average should be greater than 0'
    assert delta >= 0, 'deviation should be non-negative'
    assert delta <= average, 'deviation should be equal or less than average'
    assert number_of_agents > 0, 'number_of_agents should be greater than 0'

    g = np.random.uniform(average - delta, average + delta, number_of_agents)
    # used range in Brondetta et al. 2024
    g[g == 0] = np.random.uniform(0.5, 8.5, size=np.sum(g == 0))
    g = g / (np.mean(g) + 1e-6)
    g = g * average
    return g

def build_forward_matrix(number_of_agent, number_of_tasks, alpha, beta, gamma, delta, 
                         task_graph, communication_graph):
    """
    Build the forward matrix of the system, where the parameter alpha, beta, 
    gamma and delta are the same for all the agents.
    ------------------------------ 
    The forward matrix is a matrix of size (Na * No, Na * No), where Na is the number of agents
    and No is the number of tasks. The forward matrix represents the interaction between all agents 
    and tasks. The next state of the agents is given by the matrix-vector product
    of the forward matrix and the current state of the agents, besides other terms.
    ------------------------------
    The basic formulas to get the forward matrix is:
    alpha * I + beta * (1 - I)   for the in-diagonal block
    gamma * I + delta * (1 - I)  for the out-diagonal block
    ------------------------------
    A more advanced formulas that use the graphs is the following:
    alpha * (G_o * I_o) + beta * (G_o - G_o * I_o)   for the in-diagonal block
    gamma * (G_o * I_o) + delta * (G_o - G_o * I_o)  for the out-diagonal block
    where G_o is the task graph, and I_o is the identity matrix of size No.
    ------------------------------
    Furthermore, each block is multiplied by their corresponding scalar value in
    the communication graph G_a.
    ------------------------------

    Parameters
    ----------
    number_of_agent : int
        Number of agents in the system. 
        The number of agents should be greater than 0.
    number_of_tasks : int
        Number of tasks the agents have to perform. 
        The number of tasks should be greater than 0.
    alpha : float
        The scalar value that weights the same agent-same task interaction. 
        The value should be non-negative.
    beta : float
        The scalar value that weights the same agent-different task interaction.
        The value should be non-negative.    
    gamma : float
        The scalar value that weights the different agent-same task interaction.
        The value should be non-negative.
    delta : float
        The scalar value that weights the different agent-different task interaction.   
        The value should be non-negative.
    task_graph : 2D numpy array
        The graph between tasks. A positive value means that the tasks are positively correlated, 
        a negative value means that the tasks are negatively correlated. 
        A null value means that the tasks are not correlated.
        The task graph should be of size (No, No).
    communication_graph : 2D numpy array
        The graph between agents. A positive value means that the agents have a positive interaction, 
        a negative value means that the agents have a negative interaction. 
        A null value means that the agents can not communicate.
        The communication graph should be of size (Na, Na).

    Returns
    -------
    F : 2D numpy array
        The forward matrix of the system representing the interactions 
        between agents and tasks. The forward matrix is of size (Na * No, Na * No). 
    """

    assert number_of_agent > 0, 'number_of_agent should be greater than 0'
    assert number_of_tasks > 0, 'number_of_tasks should be greater than 0'
    assert alpha >= 0, 'alpha should be non-negative'
    assert beta >= 0, 'beta should be non-negative'
    assert gamma >= 0, 'gamma should be non-negative'
    assert delta >= 0, 'delta should be non-negative'
    assert task_graph.shape == (number_of_tasks, number_of_tasks), 'task_graph should be of size (No, No)'
    assert communication_graph.shape == (number_of_agent, number_of_agent), 'communication_graph should be of size (Na, Na)' 
    
    # diagonal blocks of the forward matrix (intra-agent interactions)
    diagonal_block = alpha * (task_graph * np.eye(number_of_tasks)
                              ) + beta * (task_graph - task_graph * np.eye(number_of_tasks))

    # off-diagonal blocks of the forward matrix (inter-agent interactions)
    off_diagonal_block = gamma * (task_graph * np.eye(number_of_tasks)
                                  ) + delta * (task_graph - task_graph * np.eye(number_of_tasks))

    # construct the full block matrix using Kronecker products
    F = np.kron(communication_graph * np.eye(number_of_agent), diagonal_block) \
        + np.kron((communication_graph - communication_graph * np.eye(number_of_agent)), off_diagonal_block)

    return F


def build_cue_vector(number_of_agents, number_of_tasks, number_of_informed,
                     number_of_switches, total_time, initial_steps = 0, reversed=False):
    """
    Build the cue vector for the experiment. The cue vector is a 2D vector of size (total_time, Na * No) 
    that informs the agents about the tasks they should perform at each time unit. 
    Na is the number of agents and No is the number of tasks.
    ------------------------------
    The cue vector has the following characteristics: <br>
    - The first initial_steps time units are vectors of zero, meaning no tasks are prioritized
    for all agents. <br>
    - After, the cue vector has a step function shape with n_switches regular steps, indeed leading
    agents to switch tasks at regular intervals. Every step has the same length given by 
    (total_time - initial_steps) // number_of_switches. <br>
    - The first switching step is a vector of 1 for the first task and -1 for all the other tasks,
    meaning that Task 1 is prioratized. <br>
    - The vector is then rotated by one position for each step, meaning that now Task 2 is prioratized.
    etc. <br>
    - When reversed is True, the cue vector is reversed. <br>
    - The informed agents are the first number_of_informed agents in the system and are the only ones 
    receiving the task cue. For the other agents, the cue vector elements are zeros.
    ------------------------------

    Parameters
    ----------
    number_of_agents : int
        Number of agents in the system. 
        The number of agents should be greater than 0.
    number_of_tasks : int
        Number of tasks the agents have to perform. 
        The number of tasks should be greater than 0.
    number_of_informed : int
        Number of agents that are informed about the tasks (that receive the task cue). 
        The number of informed agents should be non-negative and less than or equal to the number 
        of agents.
    number_of_switches : int
        Number of switches in the cue vector.
        The number of switches should be positive.
        The number of switches should be less than or equal to total_time - initial_steps.
    total_time : int
        Total time of the experiment in time units. 
        The total time should be greater than 0.
    initial_steps : int, optional
        Number of initial time steps where no task is prioritized. 
        The number of initial steps should be non-negative.
        The number of initial steps should be less than or equal to total_time.
        Default is 0.
    reversed : bool, optional
        This options is used to reverse the cue vector. If True, the cue vector is reversed.
        Default is False.  
        This option should be used only when the number of tasks is 2.

    Returns
    -------
    cue_vector : 2D numpy array
        The cue vector of the experiment. The cue vector is of size (total_time, Na * No).
        The cue vector informs the agents about the tasks they should perform at each time unit.
        The cue vector is the same for all agents, except for the uninformed agents that have a cue vector of zeros.
    """

    assert number_of_agents > 0, 'number_of_agents should be greater than 0'
    assert number_of_tasks > 0, 'number_of_tasks should be greater than 0'
    assert number_of_informed >= 0, 'number_of_informed should be non-negative'
    assert number_of_informed <= number_of_agents, 'number_of_informed should be less than or equal to number_of_agents'
    assert number_of_switches > 0, 'number_of_switches should be positive'
    assert number_of_switches <= total_time - initial_steps, 'number_of_switches should be less than or equal to total_time - initial_steps'
    assert total_time > 0, 'total_time should be greater than 0'
    assert initial_steps >= 0, 'initial_steps should be non-negative'
    assert initial_steps <= total_time, 'initial_steps should be less than or equal to total_time'

    number_of_switches = int(number_of_switches)  
    number_of_informed = int(number_of_informed)  
    assert (total_time - initial_steps) % number_of_switches == 0, 'total_time - initial_steps should be divisible by number_of_switches'

    switch_len = (total_time - initial_steps) // number_of_switches  

    cue_vector = np.zeros((total_time, number_of_agents * number_of_tasks))

    # first value is 1 for the first task and -1 for all the other tasks
    val = -1.0 * np.ones(number_of_tasks)
    val[0] = 1.0
    val = np.array(list(val) * number_of_agents)
    # if number informed is less than number of agents, we need to set the remaining agents to 0
    if number_of_informed < number_of_agents:
        val[number_of_informed * number_of_tasks:] = 0

    # fill the first time steps with zeros (no tasks prioritized)
    cue_vector[:initial_steps, :] = 0

    # now start the task switching after the initial steps
    for i in range(number_of_switches):
        cue_vector[initial_steps + i*switch_len:initial_steps + (i+1)*switch_len, :] = val
        val = val.reshape((number_of_agents, number_of_tasks))
        val = np.roll(val, 1, axis=1)
        val = val.flatten()

    if reversed:
        cue_vector = -1.0 * cue_vector

    return cue_vector
