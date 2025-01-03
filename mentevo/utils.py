import numpy as np


def gaussian_g_vector(average, deviation, number_of_agents):
    g = np.random.normal(average, deviation, number_of_agents)
    g = np.clip(g, 0, None)
    g = g / (np.mean(g) + 1e-6)
    g = g * average
    return g


def uniform_g_vector(average, delta, number_of_agents):
    assert delta <= average
    g = np.random.uniform(average - delta, average + delta, number_of_agents)
    g = g / (np.mean(g) + 1e-6)
    g = g * average
    return g


def build_communication_tensor(number_of_agents, number_of_tasks,
                               alpha, beta, gamma, delta,
                               task_graph, communication_graph):
    A1 = np.kron(np.eye(number_of_agents), alpha * np.eye(number_of_tasks) + beta * task_graph)
    A2 = np.kron(communication_graph, gamma * np.eye(number_of_tasks) + delta * task_graph)
    A = A1 + A2
    return A


def build_forward_matrix(number_of_agent, number_of_tasks, alpha, beta, gamma, delta, task_graph, communication_graph):
    """
    todo Ale: do the documentation.

    Tehe forward matrix is a matrix of size (Na * No) x (Na * No)
    that represents the interaction between agents and tasks.
    The next state of the agents is given by the matrix-vector product
    of the forward matrix and the current state of the agents.

    The basic formulas to get the forward matrix is:
    alpha * I + beta * (1 - I)   for the in-diagonal block
    gamma * I + delta * (1 - I)  for the out-diagonal block

    A more advnced formulas that use the graphs is the following:
    alpha * (G_o * I_o) + beta * (G_o - G_o * I_o)   for the in-diagonal block
    gamma * (G_o * I_o) + delta * (G_o - G_o * I_o)  for the out-diagonal block

    where G_o is the task graph, and I_o is the identity matrix of size No.

    Furthermore, each block is multiplied by their corresponding scalar value in
    the communication graph.

    """
    # todo: check the shapes of graphs!!
    # diagonal blocks of the forward matrix
    diagonal_block = alpha * (task_graph * np.eye(number_of_tasks)
                              ) + beta * (task_graph - task_graph * np.eye(number_of_tasks))

    # off-diagonal blocks of the forward matrix
    off_diagonal_block = gamma * (task_graph * np.eye(number_of_tasks)
                                  ) + delta * (task_graph - task_graph * np.eye(number_of_tasks))

    # construct the full Na x Na block matrix using Kronecker products
    F = np.kron(communication_graph * np.eye(number_of_agent), diagonal_block) \
        + np.kron((communication_graph - communication_graph * np.eye(number_of_agent)), off_diagonal_block)

    return F


def build_cue_vector(number_of_agents, number_of_tasks, n_informed,
                     n_switches, total_time, reversed=False):
    """
    todo the documentation
    """
    n_switches = int(n_switches)  # todo: ale see if we should assert here
    n_informed = int(n_informed)  # todo: ale see if we should assert here
    switch_len = total_time // n_switches

    cue_vector = np.zeros((total_time, number_of_agents * number_of_tasks))

    # first value is 1 for the first task and -1 for all the other tasks
    val = -1.0 * np.ones(number_of_tasks)
    val[0] = 1.0
    val = np.array(list(val) * number_of_agents)
    # val = np.ones((number_of_agents * number_of_tasks))
    # for i in range(number_of_tasks):
    #    val[i * number_of_tasks + 1: (i + 1) * number_of_tasks] = -1
    # if number informed is less than number of agents, we need to set the
    # remaining agents to 0
    if n_informed < number_of_agents:
        val[n_informed * number_of_tasks:] = 0

    for i in range(int(n_switches)):
        cue_vector[i*switch_len:(i+1)*switch_len, :] = val
        # val = np.roll(val, 1)
        # val = np.roll(val, number_of_tasks)
        val = val.reshape((number_of_agents, number_of_tasks))
        val = np.roll(val, 1, axis=1)
        val = val.flatten()

    if reversed:
        cue_vector = -1.0 * cue_vector

    return cue_vector
