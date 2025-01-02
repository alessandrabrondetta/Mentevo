import numpy as np

from .utils import gaussian_g_vector


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
                 percentage_informed=100.0,
                 ):
        # perform some sanity checks
        assert number_of_agents > 0
        assert number_of_tasks > 0
        assert 0.0 < percentage_informed <= 100.0
        assert 0.0 <= rate <= 1.0

        self.number_of_agents = number_of_agents
        self.number_of_tasks = number_of_tasks

        if communication_graph is None:
            # default communication graph is every agent communicate to each other
            # except themselves : matrix of 1 and remove the diagonal
            communication_graph = np.ones((number_of_agents, number_of_agents))
            communication_graph.fill_diagonal(0)

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
            self.initial_state = np.zeros((number_of_agents, number_of_tasks))

        self.total_time = total_time
        self.rate = rate

        self.percentage_informed = percentage_informed
