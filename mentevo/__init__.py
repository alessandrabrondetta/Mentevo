"""
Mentevo is a compact library designed for studying the dynamic of balancing cognitive stability 
and flexibility in task-switching environments within groups of agents. 

"""

__version__ = '0.0.2'

from .experiment import Experiment
from .metrics import compute_performance
from .plots import plot_curves
from .utils import gaussian_g_vector, build_forward_matrix, build_cue_vector