import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from envs.utils import convert_map_to_grid


# Define a custom colormap from light gray to dark gray
custom_gray = mcolors.LinearSegmentedColormap.from_list(
    "custom_gray", ["#D3D3D3", "#303030"]  # Light gray → Dark gray
)
# TODO (low priority) general refactor

RBF_COLORS = {
    "A": "cyan",
    "B": "magenta",
    "C": "lime",
    "D": "orange",
    "E": "blue",
}


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
    # plt.colorbar(mat, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)),
    #              ticks=np.arange(0.0, 1.1, 0.25))
    # ax.set_title(("Termination probabilities in states" + title_suffix))
    return mat


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
        cmap: Color map for the env.
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


def plot_policy(ax, arrow_data, values=False, headwidth=6, headlength=10, headaxislength=7, colorbar_size='10%'):
    """
    Plots arrows representing the optimal actions based on the Q-values.

    Args:
        ax: Matplotlib axis object.
        arrow_data: Tuple containing (x_pos, y_pos, x_dir, y_dir, color).
        values: Whether to display the Q-values as numbers.
        headwidth: Arrow head width.
        headlength: Arrow head length.
        headaxislength: Head axis length.
        colorbar_size: Size of the colorbar.
    """
    x_pos, y_pos, x_dir, y_dir, color = arrow_data
    color = np.array(color)
    color = (color - color.min()) / (color.max() - color.min())
    # norm = colors.Normalize(vmin=color.min(), vmax=color.max())

    cmap = cm.get_cmap('viridis')
    quiv = ax.quiver(
        x_pos, y_pos, x_dir, y_dir, color, cmap=cmap,
        angles='xy', scale_units='xy',
        scale=1, pivot='middle',
        headwidth=headwidth, headlength=headlength, headaxislength=headaxislength, width=0.005  # Customize arrow shape
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
    plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0, 1.1, 0.1))
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


def add_rbf_activations(ax, rbf_data, env, only_add_rbf_on_its_goal=True):
    """
    For each cell that has one or more RBF activations, plot a small marker
    (circle or square) in one of the four corners. The color is unique for each RBF,
    and the marker's transparency (alpha) is proportional to the activation value.
    """
    # First, collect activations per cell and gather the unique RBF identifiers.
    # An RBF is uniquely identified by (symbol, center_coords).
    cell_to_rbfs = {}   # key: (y,x) cell coordinate; value: list of tuples (rbf_id, activation)
    unique_rbfs = []    # list of all unique (symbol, center_coords) pairs

    for symbol, centers in rbf_data.items():
        for center_coords, cell_dict in centers.items():
            rbf_id = (symbol, center_coords)
            unique_rbfs.append(rbf_id)
            for (y, x), activation in cell_dict.items():
                if only_add_rbf_on_its_goal and env.MAP[y, x] != symbol:
                    continue
                if (y, x) not in cell_to_rbfs:
                    cell_to_rbfs[(y, x)] = []
                cell_to_rbfs[(y, x)].append((rbf_id, activation))

    # Create a unique color for each RBF using a colormap.
    cmap = plt.get_cmap('tab10')
    rbf_colors = {}
    # (If you have more than 10, consider using a different or extended colormap.)
    for i, rbf_id in enumerate(unique_rbfs):
        rbf_colors[rbf_id] = cmap(i % 10)

    # Now, for each cell, plot markers at the corners.
    # We assume that each cell (with top-left at (x,y)) spans x -> x+1 and y -> y+1.
    # Adjust the offsets if your grid is different.
    # Here we use fixed offsets (0.2 and 0.8) so that up to 4 markers (one per corner) fit.
    for (y, x), rbf_list in cell_to_rbfs.items():
        # Sort for consistency (so the same RBF always goes to the same corner).
        rbf_list = sorted(rbf_list, key=lambda tup: (tup[0][0], tup[0][1]))
        # Define corner offsets; here the order is: top-left, top-right, bottom-left, bottom-right.
        # (Adjust these as needed for your plotting coordinate system.)
        corner_offsets = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        for i, (rbf_id, activation) in enumerate(rbf_list):
            if i >= len(corner_offsets):
                break  # In case there are >4 overlapping activations.
            offset_x, offset_y = corner_offsets[i]
            # The marker will be placed at (x + offset_x, y + offset_y)
            marker_x = x + offset_x
            marker_y = y + offset_y
            marker_size = 10 + 8 * activation  # 10 when activation=0, 18 when activation=1
            # If the current cell is the RBF center, use a square ('s'); otherwise, use a circle ('o').
            marker_style = 's' if (y, x) == rbf_id[1] else 'o'
            ax.scatter(marker_x, marker_y, s=marker_size, marker=marker_style,
                       color=rbf_colors[rbf_id],
                       alpha=activation,
                       edgecolors='k', zorder=3)


def plot_q_vals(policy_index, policy, w, env, rbf_data=None):
    """
    Plot the Q-values (with arrows) on top of a grid, and optionally also plot the
    RBF activation markers in the cell corners.
    """
    arrow_data = get_plot_arrow_params(policy.q_table, w, env)  # e.g., returns (x_pos, y_pos, x_dir, y_dir, color)

    fig, ax = plt.subplots()

    grid, mapping = convert_map_to_grid(env)
    create_grid_plot(ax, grid)  # Draw the grid cells.
    add_legend(ax, mapping)     # Add legend (obstacles, goals, etc.)

    # Format the weight vector as a string for the title.
    weight_str = np.array2string(w, precision=3, separator=", ")
    ax.set_title(f"Policy {policy_index} | Weights: {weight_str}")

    # Plot the arrows that indicate the policy's best actions.
    quiv = plot_policy(ax, arrow_data, values=False)

    # If RBF data is provided, overlay the small markers.
    if rbf_data is not None:
        add_rbf_activations(ax, rbf_data, env)

    plt.show()


def plot_all_rbfs(rbf_data, grid_size, env, aggregation="sum", skip_non_goal=True, colors_symbol_centers=None):
    """
    Plots all RBF activations on a single heatmap.

    Args:
        rbf_data: Dictionary of RBF activations per center.
        grid_size: (grid_height, grid_width) of the environment.
        colors_symbol_centers: Color map for mapping RBF centers by their symbol to a color.
    """
    if colors_symbol_centers is None:
        colors_symbol_centers = RBF_COLORS

    grid_height, grid_width = grid_size
    combined_activation_grid = np.zeros((grid_height, grid_width))

    fig, ax = plt.subplots(figsize=(8, 8))

    # Aggregate all RBF activations
    for symbol, centers in rbf_data.items():
        for center_coords, activations in centers.items():
            for (y, x), activation_value in activations.items():
                if skip_non_goal and not env.MAP[y, x] == symbol:
                    activation_value = 0 #  Set to 0, so it isn't added / maxed over

                if aggregation == "sum":
                    combined_activation_grid[y, x] += activation_value  # Sum overlapping RBFs
                elif aggregation == "max" and activation_value > combined_activation_grid[y, x]:
                    combined_activation_grid[y, x] = activation_value

    # Plot heatmap
    im = ax.imshow(combined_activation_grid, cmap="hot", origin="upper", extent=(0, grid_width, 0, grid_height))

    # Plot RBF centers with distinct colors per symbol
    for symbol, centers in rbf_data.items():
        center_color = colors_symbol_centers.get(symbol, "white")  # Default to white if symbol not found
        for (cy, cx) in centers.keys():
            ax.scatter(cx + 0.5, grid_height - cy - 0.5, color=center_color, s=100, edgecolors="black",
                       label=f"RBF {symbol}" if f"RBF {symbol}" not in ax.get_legend_handles_labels()[1] else None)

    # Formatting
    ax.set_title("All RBF Activations Combined")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    plt.colorbar(im, label="RBF Activation Intensity")
    ax.grid(False)

    # Ensure legend only shows unique symbols
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Remove duplicates
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left")

    plt.show()
