import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# Define a custom colormap from light gray to dark gray
custom_gray = mcolors.LinearSegmentedColormap.from_list(
    "custom_gray", ["#D3D3D3", "#303030"]  # Light gray → Dark gray
)
import pickle
# TODO (low priority) general refactor


def create_grid_plot(ax, grid, color_map="binary"):
    if color_map == "binary":
        grid = 1 - grid
    size_y = grid.shape[0]
    size_x = grid.shape[1]
    vmax = max(float(np.max(grid)), 1)
    vmin = 0

    mat = ax.matshow(np.flip(grid, 0), cmap=plt.get_cmap(color_map), extent=[0, size_x, 0, size_y], vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(0, size_x))
    ax.set_xticks(np.arange(0.5, size_x + 0.5), minor=True)
    # ax.set_xticklabels(np.arange(0, size_x), minor=True)
    plt.setp(ax.get_xmajorticklabels(), visible=False)
    ax.set_yticks(np.arange(0, size_y))
    ax.set_yticks(np.arange(0.5, size_y + 0.5), minor=True)
    # ax.set_yticklabels(np.arange(0, size_y), minor=True)
    ax.invert_yaxis()
    plt.setp(ax.get_ymajorticklabels(), visible=False)

    ax.tick_params(axis='both', which='both', length=0)
    ax.grid(color="black")
    return mat


def create_grid_plot_values(ax, grid, color_map, coords, probs, values=True):
    grid = np.ma.masked_array(grid, grid == 0)
    for i in range(len(coords)):
        if grid[coords[i][0], coords[i][1]] != 0:
            x = coords[i][1]
            y = coords[i][0]
            grid[y][x] = probs[i]
            if values:
                ax.text(x + 0.5, y + 0.5, "{}".format(int(probs[i] * 100)), horizontalalignment='center',
                        verticalalignment='center', color="black", fontsize=7)
    return create_grid_plot(ax, grid, color_map=color_map)


def plot_options(ax, probs, coords, grid, title_suffix="", colorbar_size='10%'):
    create_grid_plot(ax, grid)
    grid = np.array(grid, float)
    mat = create_grid_plot_values(ax, grid, "YlGn", coords, probs.numpy())
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    plt.colorbar(mat, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.0, 1.1, 0.25))
    # ax.set_title(("Probability of choosing an option in states" + title_suffix))


def plot_terminations(ax, probs, coords, grid, title_suffix="", values=True, colorbar_size='10%'):
    create_grid_plot(ax, grid)
    grid = np.array(grid, float)
    mat = create_grid_plot_values(ax, grid, "OrRd", coords, probs.numpy(), values=values)
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    # plt.colorbar(mat, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.0, 1.1, 0.25))
    # ax.set_title(("Termination probabilities in states" + title_suffix))
    return mat


def plot_policy(ax, arrow_data, grid, title_suffix="", values=True, headwidth=9, headlength=20, colorbar_size='10%'):
    create_grid_plot(ax, grid)
    x_pos, y_pos, x_dir, y_dir, color = arrow_data
    quiv = ax.quiver(x_pos, y_pos, x_dir, y_dir, color, cmap=plt.get_cmap("viridis"),
                     norm=colors.Normalize(vmin=color.min(), vmax=color.max()), angles='xy', scale_units='xy',
                     scale=1, pivot='middle', clim=(0.3, 1), headwidth=headwidth, headaxislength=headlength, headlength=headlength)# width=0.1)
    divider = make_axes_locatable(ax)

    if values:
        for i in range(len(x_pos)):
            x = x_pos[i]
            y = y_pos[i]
            if x_dir[i] == 0:
                x -= 0.25
            else:
                y -= 0.25
            ax.text(x, y, "%2d" % (color[i] * 100), horizontalalignment='center',
                    verticalalignment='center', color="black", fontsize=4)
    # cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    # plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.3, 1.1, 0.1))
    # ax.set_title(("Maximum likelihood actions in states" + title_suffix))

    return quiv


def plot_policy_alternative(ax, arrow_data, grid, title_suffix="", values=True, colorbar_size='10%', fixed_colors=True):
    create_grid_plot(ax, grid)
    cmap = plt.get_cmap('viridis')
    x_pos, y_pos, x_dir, y_dir, color = arrow_data

    MAX_ARROW_WIDTH = 0.5
    MAX_ARROW_FORWARD = 0.4
    MAX_ARROW_BACKWARD = 0.5

    for i in range(len(x_pos)):
        arrow_width = MAX_ARROW_WIDTH * color[i]
        arrow_forward = MAX_ARROW_FORWARD * color[i]
        arrow_backward = MAX_ARROW_BACKWARD * color[i]

        x = float(x_pos[i])
        y = float(y_pos[i])
        if x_dir[i] == 1 and y_dir[i] == 0: # right
            c = 'limegreen' if fixed_colors else cmap(color[i])
            ax.add_patch(patches.Polygon(np.array([[x + arrow_forward, y],
                                                   [x - arrow_backward, y - arrow_width / 2],
                                                   [x - arrow_backward, y + arrow_width / 2]]),
                                         edgecolor=c, facecolor=c))
        elif x_dir[i] == -1 and y_dir[i] == 0: # left
            c = 'magenta' if fixed_colors else cmap(color[i])
            ax.add_patch(patches.Polygon(np.array([[x - arrow_forward, y_pos[i]],
                                                   [x + arrow_backward, y_pos[i] - arrow_width / 2],
                                                   [x + arrow_backward, y_pos[i] + arrow_width / 2]]),
                                         edgecolor=c, facecolor=c))
        elif x_dir[i] == 0 and y_dir[i] == 1: # top
            c = 'b' if fixed_colors else cmap(color[i])
            ax.add_patch(patches.Polygon(np.array([[x, y + arrow_forward],
                                                   [x - arrow_width / 2, y - arrow_backward],
                                                   [x + arrow_width / 2, y - arrow_backward]]),
                                         edgecolor=c, facecolor=c))
        elif x_dir[i] == 0 and y_dir[i] == -1: # bottom
            c = 'r' if fixed_colors else cmap(color[i])
            ax.add_patch(patches.Polygon(np.array([[x, y - arrow_forward],
                                                   [x - arrow_width / 2, y + arrow_backward],
                                                   [x + arrow_width / 2, y + arrow_backward]]),
                                         edgecolor=c, facecolor=c))
    if values:
        for i in range(len(x_pos)):
            x = x_pos[i]
            y = y_pos[i]
            if x_dir[i] == 0:
                x -= 0.25
            else:
                y -= 0.25
            ax.text(x, y, "%2d" % (color[i] * 100), horizontalalignment='center',
                    verticalalignment='center', color="black", fontsize=7)

    if not fixed_colors:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
        sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        plt.colorbar(sm, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0, 1.1, 0.25))

    ax.set_title(("Maximum likelihood actions in states" + title_suffix))



def get_plot_arrow_params(q_table, w, grid_env):
    x_pos = []
    y_pos = []
    x_dir = []
    y_dir = []
    color = []

    for coords,q_vals in q_table.items():
        max_val = np.max(q_vals @ w)
        max_index = np.argmax(q_vals @ w)

        # print(coords, q_vals, max_index)

        x_d = y_d = 0
        if max_index == grid_env.DOWN:
            y_d = 1
        elif max_index == grid_env.UP:
            y_d = -1
        elif max_index == grid_env.RIGHT:
            x_d = 1
        elif max_index == grid_env.LEFT:
            x_d = -1

        x_pos.append(coords[1] + 0.5)
        y_pos.append(coords[0] + 0.5)
        x_dir.append(x_d)
        y_dir.append(y_d)
        color.append(max_val)
    # down, up , right, left
    return np.array(x_pos), np.array(y_pos), np.array(x_dir), np.array(y_dir), np.array(color)

def create_grid_plot(ax, grid, cmap=custom_gray):
    """
    Plots a 2D grid representation of the environment using a custom gray colormap.

    Args:
        ax: Matplotlib axis object.
        grid: 2D numpy array representation of the environment.
    """
    size_y, size_x = grid.shape
    vmax = np.max(grid)
    vmin = 0

    # Use the custom grayscale colormap
    mat = ax.matshow(np.flip(grid, 0), cmap=cmap, extent=[0, size_x, 0, size_y], vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(0, size_x))
    ax.set_xticks(np.arange(0.5, size_x + 0.5), minor=True)
    plt.setp(ax.get_xmajorticklabels(), visible=False)

    ax.set_yticks(np.arange(0, size_y))
    ax.set_yticks(np.arange(0.5, size_y + 0.5), minor=True)
    plt.setp(ax.get_ymajorticklabels(), visible=False)

    ax.tick_params(axis='both', which='both', length=0)
    ax.set_ylim(size_y, 0)  # Ensures correct y-axis orientation
    ax.grid(color="black")

    return mat

def plot_policy(ax, arrow_data, grid, title_suffix="", values=False, headwidth=9, headlength=20, colorbar_size='10%', max_index=None):
    """
    Plots arrows representing the optimal actions based on the Q-values.

    Args:
        ax: Matplotlib axis object.
        arrow_data: Tuple containing (x_pos, y_pos, x_dir, y_dir, color).
        grid: The environment's 2D grid representation.
        title_suffix: Title suffix.
        values: Whether to display the Q-values as numbers.
        headwidth: Arrow head width.
        headlength: Arrow head length.
        colorbar_size: Size of the colorbar.
    """
    x_pos, y_pos, x_dir, y_dir, color = arrow_data
    norm = colors.Normalize(vmin=color.min(), vmax=color.max())
    cmap = cm.get_cmap('viridis')
    # quiv = ax.quiver(x_pos, y_pos, x_dir, y_dir, color, cmap=cmap,
    #                  norm=norm, angles='xy', scale_units='xy',
    #                  scale=1, pivot='middle', clim=(0.3, 1), headwidth=headwidth, headaxislength=headlength, headlength=headlength)
    quiv = ax.quiver(
        x_pos, y_pos, x_dir, y_dir, color, cmap=cmap,
        norm=norm, angles='xy', scale_units='xy',
        scale=1, pivot='middle', clim=(0.3, 1),
        headwidth=6, headlength=10, headaxislength=7, width=0.005  # Customize arrow shape
    )

    # Only plot values if show_values is True
    if values:
        for i in range(len(x_pos)):
            x = x_pos[i]
            y = y_pos[i]
            if x_dir[i] == 0:
                x -= 0.25
            else:
                y -= 0.25
            ax.text(x, y, "%.3f" % color[i], horizontalalignment='center',
                    verticalalignment='center', color="black", fontsize=7)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0.3, 1.1, 0.1))
    return quiv

def add_legend(ax, mapping, cmap=custom_gray):
    """
    Adds a legend to the plot based on the provided mapping, positioning it outside the grid.

    Args:
        ax: Matplotlib axis object.
        mapping: Dictionary mapping symbols to numeric values.
        cmap: The colormap used for the grid.
    """
    # Remove keys for empty spaces and starting location
    filtered_mapping = {k: v for k, v in mapping.items() if k not in [" ", "_"]}

    norm = plt.Normalize(min(filtered_mapping.values()), max(filtered_mapping.values()))

    legend_patches = [
        mpatches.Patch(color=cmap(norm(value)), label=f"{key}")  # Map symbol to color
        for key, value in filtered_mapping.items()
    ]

    # Move legend to the **left** of the plot, outside the grid area
    ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(-0.3, 0.5),  # Moves the legend further left
        title="Legend",
        fontsize=8,
        frameon=False  # Removes legend box outline
    )

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


