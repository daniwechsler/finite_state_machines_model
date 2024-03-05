import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def getColorsFromColorMap (colorMap, N, min=0.0, max=0.0):
    """
    Returns N equally distant colors from the colormap defiend
    by colorMapString
    :param colorMap: String name of colormap or colormap
    :param N:
    :return:
    """

    if isinstance(colorMap, str):
        cmap = mpl.cm.get_cmap(colorMap)
    else:
        cmap = colorMap

    colors = list()

    interval = (1.0-max-min)/N
    intervals = np.arange(min, 1.0-max, interval)
    for i in intervals:
        colors.append(cmap(i))

    return colors


def contourPlot (dataFrame, ax, fig, levels=10, colorMap='jet', roundLevels=2, alpha=0.9, addColorBar=True, colorMin=0.0, colorMax=0.0, args=None):
    """
    Creates a contour plot from the data frame. The index of dataFrame
    is the y-axis and the columns are the x-axis

    :param dataFrame:
    :param ax:
    :param fig:
    :param levels:
    :param colorMap:
    :return:
    """

    args_defaults = {}
    args_defaults["line_color"] = "black"
    args_defaults["line_linewidht"] = 0.5
    args_defaults["line_alpha"] = 1.0

    if not args is None:
        for key in args:
            args_defaults[key] = args[key]

    X, Y = np.meshgrid(dataFrame.columns, dataFrame.index)
    res = dataFrame.values

    maxValue = dataFrame.max().max()
    minValue = dataFrame.min().min()


    if isinstance(levels, int):
        levels = np.linspace(minValue, maxValue, levels)

    colors = getColorsFromColorMap(colorMap, len(levels), min=colorMin, max=colorMax)
    CSF = ax.contourf(X, Y, res, levels=levels, alpha=alpha, colors=colors)
    CS = ax.contour(X, Y, res, levels=levels, colors=args_defaults["line_color"], alpha=args_defaults["line_alpha"])

    if addColorBar:
        cbar = fig.colorbar(CSF, ax=ax, ticks=levels)

        format = "%." + str(roundLevels) + "f"
        levelsFormated = [ format % elem for elem in levels ]

        if roundLevels == 0:
            levelsFormated = [ int(elem) for elem in levels ]

        cbar.ax.set_yticklabels(levelsFormated)

        return cbar

    return None

def plot_binary_matrix(M, ax, xlabel=None, ylabel=None, x_group=None, y_group=None, colormap='hsv'):
    """
    Plots the given binary matrix (entries either 0 or 1) such that
    the rows are on the y-axis and the columns on the x-axis.
    Interactions (1) are denoted by a black and no interactions (0) by
    a white dot.
    :param M:
    :param ax:
    :return:
    """

    if not x_group is None and not y_group is None:
        num_groups =  max(len(np.unique(x_group)), len(np.unique(y_group)))

        cMap = [(0.0, 'white')]
        for value, colour in zip(list(range(1, num_groups+1)), getColorsFromColorMap(colormap, num_groups, min=0.0, max=0.0)):
            cMap.append((value / (num_groups), colour))

        cMap.append((1.0, 'grey'))

        customColourMap = LinearSegmentedColormap.from_list("custom", cMap)
        M_colors = np.zeros(M.shape)

        for k, group in enumerate(np.unique(x_group)):

            x = (x_group == group).astype(int)
            x_mask = np.stack([x.copy() for i in range(len(y_group))], axis=1)

            y = (y_group == group).astype(int)
            y_mask = np.stack([y.copy() for i in range(len(x_group))], axis=0)

            M_group = np.multiply(np.multiply(x_mask, y_mask), M) * (k+1) / (num_groups+1)
            M_colors += M_group

        M_colors += np.multiply((M_colors == 0).astype(int), M)
        ax.imshow(M_colors, interpolation="nearest", cmap=customColourMap, vmin=0, vmax=1)

    else:
        ax.imshow(M, interpolation="nearest", cmap='binary', vmin=0, vmax=1)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    if not xlabel is None:
        ax.set_xlabel(xlabel)

    if not ylabel is None:
        ax.set_ylabel(ylabel)


def plot_binary_matrix_sort_modules(G, ax, xlabel=None, ylabel=None, colormap='hsv', modularity_function=None):
    """
    A wrapper for the previous function that computes the modules first.
    :param G:
    :param ax:
    :param xlabel:
    :param ylabel:
    :param colormap:
    :return:
    """
    G, row_indices, col_indices = sort_matrix_by_modules(G, modularity_function=modularity_function, return_groups=True)
    plot_binary_matrix(G, ax, xlabel=xlabel, ylabel=ylabel, x_group=row_indices, y_group=col_indices, colormap=colormap)


def sort_matrix_by_modules(G, modularity_function=None, row_groups=None, col_groups=None, return_groups=False):

    if row_groups is None or col_groups is None:
        Q_stats = modularity_function(G)
        row_groups = Q_stats['groups']
        col_groups = Q_stats['groups']

    row_in_group_degrees = np.array([0] * G.shape[0])
    col_in_group_degrees = np.array([0] * G.shape[1])

    row_group_sizes = np.array([0] * G.shape[0])
    col_group_sizes = np.array([0] * G.shape[1])

    row_group_ids, row_group_cnt = np.unique(row_groups, return_counts=True)

    for i, row_group in enumerate(row_group_ids):
        for row in range(G.shape[0]):
            if row_groups[row] == row_group:
                row_group_sizes[row] = row_group_cnt[i]
        for col in range(G.shape[1]):
            if col_groups[col] == row_group:
                col_group_sizes[col] = row_group_cnt[i]

    for group in np.unique(row_groups):

        rows_in_group = [int(row == group) for row in row_groups]
        cols_in_group = [int(col == group) for col in col_groups]
        degrees = np.zeros(G.shape, dtype=int)

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                degrees[i, j] = rows_in_group[i]*cols_in_group[j]*G[i, j]

        row_in_group_degrees += degrees.sum(axis=1)
        col_in_group_degrees += degrees.sum(axis=0)

    G = G.copy()
    row_indices = list(range(len(row_groups)))
    row_indices_groups = [(index, module) for g_size, module, degree, index in list(sorted(zip(row_group_sizes, row_groups, row_in_group_degrees, row_indices), reverse=True))]
    row_indices, row_groups_s = map(list, zip(*row_indices_groups))
    G = G[row_indices,:]
    G = G.T

    col_indices = list(range(len(col_groups)))
    col_indices_groups = [(index, module) for g_size, module, degree, index in sorted(zip(col_group_sizes, col_groups, col_in_group_degrees, col_indices), reverse=True)]
    col_indices, col_groups_s = map(list, zip(*col_indices_groups))
    G = G[col_indices,:]
    G = G.T

    if return_groups:
        return G, row_groups_s, col_groups_s
    return G
