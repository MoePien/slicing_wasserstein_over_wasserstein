import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import FuncFormatter
from functools import partial

def four_symbols(x, pos, strnum=4):
    return f"{x:.10f}"[:strnum]  
    
def plot_means_with_errorbars(x, means, stds, xlabel="", ylabel="", save_path=None, x_target=0., x_min=None, ylim = 0, legend=True, ystrnum=4, xstrnum=None):
    """
    Create a scatter plot with error bars from means and stds.
    
    Parameters
    ----------
    x : array-like
        X values.
    means : array-like
        Mean values for Y.
    stds : array-like
        Standard deviations (for error bars).
    xlabel : str, optional
        Label for the X axis.
    ylabel : str, optional
        Label for the Y axis.
    save_path : str or None, optional
        If provided, saves the plot to this path. Otherwise shows the plot.
    """
    x = np.asarray(x)
    means = np.asarray(means)
    stds = np.asarray(stds)

    # standard error of the mean
    stderrs = stds / np.sqrt(len(means))

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, means, yerr=stderrs, fmt="o", capsize=10, markersize=14, 
                 elinewidth=2, label="Means ± Std")

    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)
    plt.ylim(ylim)
    if x_min is None:
        x_min = -0.05
    plt.xlim(x_min)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if x_target is not None:
        plt.vlines(x_target, 0, means.max() + stds.max(), color="red", label="Reference", linestyles="dotted")
    y_sym_func = partial(four_symbols, strnum=ystrnum)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_sym_func))
    if xstrnum is not None:
        x_sym_func = partial(four_symbols, strnum=xstrnum)
        plt.gca().xaxis.set_major_formatter(FuncFormatter(x_sym_func))
    if legend:
        plt.legend(fontsize=25)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()