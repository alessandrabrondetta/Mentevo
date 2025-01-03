import numpy as np
from scipy.integrate import solve_ivp

from .utils import gaussian_g_vector, build_forward_matrix, build_cue_vector


class Experiment():
    """
    We need to do a small explanations here.

    Parameters
    ----------
    number_of_agents : int
        Number of agents in the system.
    number_of_tasks : int
        Number of tasks the agents have to perform.
    TODO: Ale finish this :)
    communication_graph : 2D array, optional
        The communication graph between agents. 1 means that the agents can communicate
        and 0 means that they cannot. Usually, the diagonal is 1 (intra-communication).
        If None, then the default is a fully connected graph.
    task_graph : 2D array, optional
        The task graph between tasks. 1 means that the tasks are positively correlated
        and -1 means that the tasks are negatively correlated. Usually, the diagonal is 1
        and the rest is -1. If None, then the default is diagonal 1 and non diagonal -1.

    """

    def __init__(self,
                 number_of_agents=4,
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
                 total_time=1_000,
                 rate=0.1,
                 nb_informed=None,
                 ):
        # perform some sanity checks
        assert number_of_agents > 0
        assert number_of_tasks > 0
        assert 0.0 <= rate <= 1.0
        # todo: finish assertions

        self.number_of_agents = number_of_agents
        self.number_of_tasks = number_of_tasks

        if communication_graph is None:
            # default communication graph is every agent communicate to each other
            communication_graph = np.ones((number_of_agents, number_of_agents))

        if task_graph is None:
            # default task graph is every task is negatively correlated with each other
            # except themselves : matrix of -1 and 1 on the diagonal
            task_graph = -1.0 * np.ones((number_of_tasks, number_of_tasks))
            task_graph.fill_diagonal(1)

        # equations parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.d = d
        self.tau = tau

        self.bias_value = bias_value

        if g is None:
            self.g = gaussian_g_vector(1.0, 0.1, number_of_agents)

        if initial_state is None:
            # default is zero for all agents on all tasks
            self.initial_state = np.zeros((number_of_agents * number_of_tasks))

        self.total_time = total_time
        self.rate = rate

        if nb_informed is None:
            self.nb_informed = number_of_agents

        self.F = build_forward_matrix(number_of_agents, number_of_tasks,
                                      alpha, beta, gamma, delta,
                                      task_graph, communication_graph)

        self.cue_vector = build_cue_vector(number_of_agents, number_of_tasks,
                                           self.nb_informed, total_time)

    def solve(self):
        """
        Solve the experiment.
        todo: Ale: do the documentation.
        """
        g = self.g.repeat(self.number_of_tasks)

        def diff(t, z):
            cue = self.cue_vector[round(t)]
            f = self.F @ z
            f = f / self.number_of_agents
            f = - self.d * z + np.tanh(g * f + cue)
            f = f * (1.0 / self.tau)
            return f

        # solve using scipy
        zs = solve_ivp(diff, [0, self.total_time], self.initial_state,
                       dense_output=False, max_step=1_000, method='Radau',
                       t_eval=np.arange(0, self.total_time, 1))

        return zs
