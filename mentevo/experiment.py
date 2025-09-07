import numpy as np
from math import floor
from scipy.integrate import solve_ivp

from .utils import gaussian_g_vector, build_forward_matrix, build_cue_vector


class Experiment():
    """
    Class to represent an experiment. This Experiment class models and simulates a system with 
    interacting agents and tasks, solving a nonlinear dynamical system defined by agent behaviors, 
    communication structure and task correlations, system parameters and external inputs.
    The Experiment class simulates the evolution of the task activities (agents' focus) of the 
    agents over time, and is also defined by the initial state of the system and the parameters 
    of the simulation, such as total time of the simulation and number of task switches. 
    The class contains a solver for simulating system evolution over time.

    Parameters
    ----------
    number_of_agents : int, optional 
        Number of agents in the system. Must be greater than 0. Default is 4. 
    number_of_tasks : int, optional
        Number of tasks the agents have to perform. Must be greater than 0. Default is 2.
    communication_graph : 2D numpy array, optional
        The graph between agents. A positive value means that the agents have a positive interaction, 
        a negative value means that the agents have a negative interaction. A null value means that 
        the agents can not communicate.   
        Typically, the diagonal is 1 (intra-agent-communication).
        If None, then the default is a fully connected graph where all agents can communicate.
        The shape of the matrix is (number_of_agents, number_of_agents).
    task_graph : 2D numpy array, optional
        The graph between tasks. A positive value means that the tasks are positively correlated, 
        a negative value means that the tasks are negatively correlated. A null value means that 
        the tasks are not correlated.
        Typically, the diagonal is 1 (same-task correlation). 
        If None, then the default is diagonal 1 (positive self correlation) and 
        off-diagonal values are -1 (negative correlations with the other tasks).
        The shape of the matrix is (number_of_tasks, number_of_tasks). 
    alpha : float, optional
        Parameter of the dynamical system's equations that represents the weight of the same agent-
        same task interactions. Must be greater than or equal to 0. Default is 0.03.  
    beta : float, optional
        Parameter of the dynamical system's equations that represents the weight of the same agent-
        different task interactions. Must be greater than or equal to 0. Default is 0.01. 
    gamma : float, optional
        Parameter of the dynamical system's equations that represents the weight of the different agent-
        same task interactions. Must be greater than or equal to 0. Default is 0.02. 
    delta : float, optional     
        Parameter of the dynamical system's equations that represents the weight of the different agent-
        different task interactions. Must be greater than or equal to 0. Default is 0.0. 
    d : float, optional
        Parameter of the dynamical system's equations that represents the weight of the decay term.
        Must be greater than or equal to 0. Default is 0.2.    
    tau : float, optional
        Parameter of the dynamical system's equations that represents the time constant. 
        Must be greater than 0. Default is 10.0.
    g : 1D numpy array, optional
        The g vector of the dynamical system's equations, representing the gain values of the agents 
        and regulating the slope of the saturation function. 
        The shape of the array is (number_of_agents,).
        Default is a gaussian vector with mean 3.0 and standard deviation 1.0. 
    bias_value : float, optional
        The bias value of the dynamical system's equations that represents the weight of the cue 
        vector of the experiment. Must be greater than or equal to 0. Default is 0.1.
    initial_state : 1D numpy array, optional
        The initial state of the system, representing the task activity states at the start
        of the experiment. The shape of the array is (number_of_agents*number_of_tasks,).
        Default is an array of zero for all agents on all tasks.
    total_time : int, optional
        The total time of the simulation in time units. Must be greater than 0. Default is 2_000. 
    initial_steps : int, optional
        The number of initial steps in the experiment, where the agents don't receive the task cue. 
        Must be greater than or equal to 0. Must be less than total_time. Default is 0.
    reverse : bool, optional
        If True, the task cue vector is reversed in the experiment. Default is False.
    number_of_switches : int, optional
        The number of task-cue switches in the experiment. Must be greater than 0. 
        Must be less than or equal to total_time. Default is 4.
        The first switch correspond to the start of the experiment and 
        occurs at time unit 0 if initial_steps is 0, otherwise at time unit initial_steps.
        Attention: the number of switches corresponds to the number of switches in the task cue vector
        and correspond to the number of blocks of tasks in the experiment, and is different
        from the number of task-to-task switches (which is number_of_switches - 1).
    number_of_informed : int, optional
        The number of agents that are informed in the experiment (agents that receive the task
        cue vector). Must be less than or equal to number_of_agents and non-negative.
        Default is number_of_agents. 

    Attributes
    ----------
    F : 2D numpy array
        The forward matrix of the dynamical system, representing the interactions 
        between agents and tasks. The shape of the matrix is 
        (number_of_agents*number_of_tasks, number_of_agents*number_of_tasks).
    cue_vector : 2D numpy array
        The cue vector of the experiment, representing the external input to the system.
        The shape of the array is (total_time, number_of_agents*number_of_tasks).
        The cue vector is built according to the number of switches, initial steps,
        number of informed agents and reverse parameters and scaled by the bias_value.
        The default shape of the cue vector on a single task is a step function that switches 
        between tasks at regular intervals, and it is positive for the relevant task for the informed agents,
        while negative for the other non-relevant tasks for the informed agents, and zero for the non-informed agents. 
        It is zero during the initial steps.
    task_switching_times : 1D numpy array
        The time units at which the task switches occur in the experiment. 
        The shape of the array is (number_of_task_switches,). 
        The default is evenly spaced time units between initial_steps and total_time.
    """

    def __init__(self,
                 number_of_agents=4,
                 number_of_tasks=2,
                 communication_graph=None,
                 task_graph=None,
                 alpha=0.03,
                 beta=0.01,
                 gamma=0.02,
                 delta=0.0,
                 d=0.2,
                 tau=10.0,
                 g=None,
                 bias_value=0.1,
                 initial_state=None,
                 total_time=2000,
                 initial_steps=0,
                 reverse=False,
                 number_of_switches=4,
                 number_of_informed=None,
                 ):
        # perform some sanity checks
        assert number_of_agents > 0
        assert number_of_tasks > 0
        assert alpha >= 0
        assert beta >= 0
        assert gamma >= 0
        assert delta >= 0
        assert d >= 0
        assert tau > 0
        assert bias_value >= 0
        assert total_time > 0
        assert initial_steps >= 0
        assert initial_steps < total_time
        assert number_of_switches > 0
        assert number_of_switches <= total_time

        if communication_graph is None:
            # default communication graph is every agent communicate to each other
            communication_graph = np.ones((number_of_agents, number_of_agents))
        assert communication_graph.shape == (number_of_agents, number_of_agents)

        if task_graph is None:
            # default task graph is every task is negatively correlated with each other
            # except themselves : matrix of -1 and 1 on the diagonal
            task_graph = -1.0 * np.ones((number_of_tasks, number_of_tasks))
            np.fill_diagonal(task_graph, 1)
        assert task_graph.shape == (number_of_tasks, number_of_tasks)

        if g is None:
            g = gaussian_g_vector(3.0, 1.0, number_of_agents)
        assert g.shape == (number_of_agents,)

        if initial_state is None:
            # default is zero for all agents on all tasks
            initial_state = np.zeros((number_of_agents * number_of_tasks))
        assert initial_state.shape == (number_of_agents * number_of_tasks,)

        if number_of_informed is None:
            number_of_informed = number_of_agents
        assert number_of_informed >= 0
        assert number_of_informed <= number_of_agents

        # define attributes

        self.number_of_agents = number_of_agents
        self.number_of_tasks = number_of_tasks

        self.communication_graph = communication_graph
        self.task_graph = task_graph

        # equations parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.d = d
        self.tau = tau
        self.g = g

        # simulation parameters
        self.bias_value = bias_value
        self.initial_state = initial_state
        self.total_time = total_time
        self.initial_steps = initial_steps
        self.reverse = reverse
        self.number_of_switches = number_of_switches
        self.number_of_informed = number_of_informed
  
        # build the forward matrix 
        self.F = build_forward_matrix(self.number_of_agents, self.number_of_tasks,
                                      self.alpha, self.beta, self.gamma, self.delta,
                                      self.task_graph, self.communication_graph)
        assert self.F.shape == (self.number_of_agents * self.number_of_tasks,
                                self.number_of_agents * self.number_of_tasks)

        # build the cue vector
        self.cue_vector = build_cue_vector(self.number_of_agents, self.number_of_tasks,
                                           self.number_of_informed, self.number_of_switches,
                                           self.total_time, self.initial_steps, self.reverse)
        self.cue_vector = self.cue_vector * self.bias_value
        assert self.cue_vector.shape == (self.total_time, self.number_of_agents * self.number_of_tasks)

        # compute the time when the task switching occurs (first steps with new task)
        diff_cue = np.abs(self.cue_vector[1:] - self.cue_vector[:-1]).sum(-1)
        task_switching_times = np.argwhere(diff_cue > 1e-9).flatten() + 1
        # if initial_steps == 0, then the first task starts at time 0 and 0 is included in the task switching times
        if self.initial_steps == 0:
            self.task_switching_times = np.r_[0, task_switching_times]
        # if initial_steps > 0, then the first task starts at time initial_steps and 0 is not included in the task switching times
        # the first task switching time is initial_steps
        else:
            self.task_switching_times = task_switching_times
        assert self.task_switching_times.shape == (self.number_of_switches,)

    def solve(self, **kwargs):
        """
        Solve the dynamical system defined by the experiment's parameters and initial state. 
        The system is solved using the scipy solve_ivp function. The system is solved for the
        total time of the experiment, with a maximum step of 1_000 and using the Radau 
        integration method.
        The system is solved for each time unit, and the task activity states of the agents
        are returned over time in the experiment.

        Parameters
        ----------
        kwargs : dictionary, optional
            Additional keyword arguments to pass to the solve_ivp function.

        Returns
        -------
        zs.y : 2D numpy array
            The task activity states of the agents over time in the experiment.
            The shape of the array is (number_of_agents*number_of_tasks, total_time).
        """
        g = self.g.repeat(self.number_of_tasks)
        assert g.shape == (self.number_of_agents * self.number_of_tasks,)

        def diff(t, z):
            cue = self.cue_vector[int(floor(t))]
            
            f = self.F @ z
            f = f / self.number_of_agents
            f = - self.d * z + np.tanh(g * f + cue)
            f = f * (1.0 / self.tau)
            return f

        # solve using scipy
        zs = solve_ivp(diff, [0, self.total_time-1], self.initial_state,
                       dense_output=False, max_step=1_000, method='Radau',
                       t_eval=np.arange(0, self.total_time, 1), **kwargs)

        return np.array(zs.y)
