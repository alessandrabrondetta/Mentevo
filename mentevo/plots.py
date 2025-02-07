import numpy as np
import matplotlib.pyplot as plt


def plot_curves(experiment, 
                simulation_results,
                title=None,
                x_label=None,
                y_label=None,
                legend_labels=None,
                show_legend=False,
                legend_location='inside',  
                legend_fontsize=10,  
                show_vertical_lines=True,
                task_switching_linewidth=1,
                task_switching_line_color='black', 
                task_switching_line_style='--', 
                show_cue_vector=False,
                **kwargs,
                ):
    

    """
    Plots simulation curves for a two-task experiment, specifically the time evolution of the normalized 
    difference between the two task activities for each agent.

    Parameters
    ----------
    experiment : Experiment
        The experiment object that generated the simulation_results.
    simulation_results : 2D numpy array
        The simulation results use to create the plot. 
        The shape should be (number_of_agents * number_of_tasks, total_time).
    title : str, optional
        The title of the plot. The default is None.
    x_label : str, optional
        The x-axis label. The default is None.
    y_label : str, optional
        The y-axis label. The default is None.
    legend_labels : list of str, optional
        The labels for the legend. The default 'Agent i+1' whit i in range(number_of_agents).
    show_legend : bool, optional
        Whether to show the legend. The default is False (the legend is not shown).
    legend_location : str, optional
        The location of the legend. The default is 'inside' (the legend is inside the plot). 
        If 'outside', the legend is outside the plot.
    legend_fontsize : int, optional
        The font size of the legend. The default is 10.
    show_vertical_lines : bool, optional
        Whether to show vertical lines for task-switching times. The default is True (the vertical
        lines are plotted).
    task_switching_linewidth : int, optional    
        The width of the task-switching vertical lines. The default is 1.
    task_switching_line_color : str, optional
        The color of the task-switching vertical lines. The default is 'black'. 
    task_switching_line_style : str, optional
        The line style of the task-switching vertical lines. The default is '--'.
    show_cue_vector : bool, optional
        Whether to plot the cue vector. The default is False (the cue vector is not plotted).
    kwargs : dictionary, optional
        Additional keyword arguments to pass to the plot function (e.g., color, linestyle, etc.).

    """
    assert isinstance(simulation_results, np.ndarray), 'simulation_results must be a numpy array'

    na = experiment.number_of_agents
    no = experiment.number_of_tasks
    total_time = experiment.total_time
    cue_vector = experiment.cue_vector

    assert no == 2, 'this visualization works only for number_of_tasks = 2'
    assert simulation_results.shape == (na * no, total_time), 'simulation_results has the right shape'
    assert cue_vector.shape == (total_time, na * no), 'cue_vector has the right shape' 

    # Compute curves for each agent
    curves = [(simulation_results[i*2] - simulation_results[i*2+1]) / 2 for i in range(na)]

    # Plot the curves
    for i, curve in enumerate(curves):
        label = legend_labels[i] if legend_labels and i < len(legend_labels) else f"Agent {i + 1}"
        plt.plot(curve, label=label, **kwargs)

    # Add vertical lines for task-switching times
    if show_vertical_lines:
        for t in experiment.task_switching_times:
            plt.axvline(
                x=t, 
                linestyle=task_switching_line_style, 
                color=task_switching_line_color, 
                linewidth=task_switching_linewidth,
                label="Task Switching Time" if t == experiment.task_switching_times[0] else None,
            )

    # Set optional plot details
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    # Add cue vector if requested
    if show_cue_vector is True:
        cue = cue_vector[:, 0] - cue_vector[:, 1]
        plt.plot(cue, label='Cue Vector', linestyle='-', color='black')

    # Add legend if labels are provided
    if show_legend is True:
        if legend_location == 'outside':
            plt.legend(fontsize=legend_fontsize, bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()  # adjust layout to make room for the legend
        else:  # default "inside" behavior
            plt.legend(fontsize=legend_fontsize)
