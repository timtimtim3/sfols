import os

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
    coords_list = []

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
        elif max_index == grid_env.TERMINATE:
            pass

        x_pos.append(coords[1] + 0.5)
        y_pos.append(coords[0] + 0.5)
        x_dir.append(x_d)
        y_dir.append(y_d)
        color.append(max_val)
        coords_list.append(coords)
    # down, up , right, left
    return np.array(x_pos), np.array(y_pos), np.array(x_dir), np.array(y_dir), np.array(color), coords_list


def get_plot_arrow_params_from_eval(actions, qvals, grid_env):
    x_pos = []
    y_pos = []
    x_dir = []
    y_dir = []
    color = []
    coords_list = []

    for coords, max_index in actions.items():
        max_val = qvals[coords]
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
        coords_list.append(coords)
    # down, up , right, left
    return np.array(x_pos), np.array(y_pos), np.array(x_dir), np.array(y_dir), np.array(color), coords_list


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
    Terminate actions (where the agent stays in place) are indicated with a smaller hollow circle (donut marker).

    Args:
        ax: Matplotlib axis object.
        arrow_data: Tuple containing (x_pos, y_pos, x_dir, y_dir, color, coords_list).
        values: Whether to display the Q-values as numbers.
        headwidth: Arrow head width.
        headlength: Arrow head length.
        headaxislength: Head axis length.
        colorbar_size: Size of the colorbar.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import cm
    from matplotlib.ticker import FuncFormatter

    x_pos, y_pos, x_dir, y_dir, color, coords_list = arrow_data
    color = np.array(color)
    normed_color = (color - color.min()) / (color.max() - color.min())
    cmap = cm.get_cmap('viridis')

    # Identify indices for normal actions and terminate actions
    normal_indices = np.where((x_dir != 0) | (y_dir != 0))[0]
    term_indices = np.where((x_dir == 0) & (y_dir == 0))[0]

    # Plot arrows for normal actions
    if len(normal_indices) > 0:
        quiv = ax.quiver(
            np.array(x_pos)[normal_indices],
            np.array(y_pos)[normal_indices],
            np.array(x_dir)[normal_indices],
            np.array(y_dir)[normal_indices],
            normed_color[normal_indices],
            cmap=cmap,
            angles='xy', scale_units='xy', scale=1, pivot='middle',
            headwidth=headwidth, headlength=headlength, headaxislength=headaxislength, width=0.005
        )

    # Plot termination actions with a smaller hollow circle ("donut" marker)
    if len(term_indices) > 0:
        scatter = ax.scatter(
            np.array(x_pos)[term_indices],
            np.array(y_pos)[term_indices],
            facecolors='none',  # Hollow circle (donut)
            edgecolors=cmap(normed_color[term_indices]),
            marker='o', s=40,  # Smaller marker size
            linewidths=1.5
        )
        if values:
            for i in term_indices:
                ax.text(x_pos[i], y_pos[i], "%.3f" % normed_color[i],
                        horizontalalignment='center', verticalalignment='center',
                        color='black', fontsize=7)

    # Optionally, annotate normal action arrows with their Q-values
    if values:
        for i in normal_indices:
            x = x_pos[i]
            y = y_pos[i]
            if x_dir[i] == 0:
                x -= 0.25
            else:
                y -= 0.25
            ax.text(x, y, "%.3f" % normed_color[i],
                    horizontalalignment='center', verticalalignment='center',
                    color='black', fontsize=7)

    # Add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
    if len(normal_indices) > 0:
        plt.colorbar(quiv, cax=cax, ax=ax,
                     format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)),
                     ticks=np.arange(0, 1.1, 0.1))
    else:
        plt.colorbar(scatter, cax=cax, ax=ax,
                     format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)),
                     ticks=np.arange(0, 1.1, 0.1))
    return ax


# def plot_policy(ax, arrow_data, values=False, headwidth=6, headlength=10, headaxislength=7, colorbar_size='10%'):
#     """
#     Plots arrows representing the optimal actions based on the Q-values.
#
#     Args:
#         ax: Matplotlib axis object.
#         arrow_data: Tuple containing (x_pos, y_pos, x_dir, y_dir, color).
#         values: Whether to display the Q-values as numbers.
#         headwidth: Arrow head width.
#         headlength: Arrow head length.
#         headaxislength: Head axis length.
#         colorbar_size: Size of the colorbar.
#     """
#     x_pos, y_pos, x_dir, y_dir, color, coords_list = arrow_data
#     color = np.array(color)
#     color = (color - color.min()) / (color.max() - color.min())
#     # norm = colors.Normalize(vmin=color.min(), vmax=color.max())
#
#     cmap = cm.get_cmap('viridis')
#     quiv = ax.quiver(
#         x_pos, y_pos, x_dir, y_dir, color, cmap=cmap,
#         angles='xy', scale_units='xy',
#         scale=1, pivot='middle',
#         headwidth=headwidth, headlength=headlength, headaxislength=headaxislength, width=0.005  # Customize arrow shape
#     )
#
#     # Only plot values if show_values is True
#     if values:
#         for i in range(len(x_pos)):
#             x = x_pos[i]
#             y = y_pos[i]
#             if x_dir[i] == 0:
#                 x -= 0.25
#             else:
#                 y -= 0.25
#             ax.text(x, y, "%.3f" % color[i], horizontalalignment='center',
#                     verticalalignment='center', color="black", fontsize=7)
#
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size=colorbar_size, pad=0.025)
#     plt.colorbar(quiv, cax=cax, ax=ax, format=FuncFormatter(lambda y, _: '{:.0%}'.format(y)), ticks=np.arange(0, 1.1, 0.1))
#     return quiv


def add_legend(ax, mapping, cmap=custom_gray):
    """
    Adds a legend to the plot based on the provided mapping, positioning it outside the grid.

    Args:
        ax: Matplotlib axis object.
        mapping: Dictionary mapping symbols to numeric values.
        cmap: The colormap used for the grid.
    """
    # Remove keys for empty spaces and starting location
    filtered_mapping = {k: v for k, v in mapping.items() if k not in ["_"]}
    empty_color = filtered_mapping[" "]
    filtered_mapping.pop(" ")
    filtered_mapping["Empty"] = empty_color

    norm = plt.Normalize(min(filtered_mapping.values()), max(filtered_mapping.values()))

    legend_patches = [
        mpatches.Patch(color=cmap(norm(value)), label=f"{key}")  # Map symbol to color
        for key, value in filtered_mapping.items()
    ]

    legend1 = ax.legend(
        handles=legend_patches,
        loc="center left",
        bbox_to_anchor=(-0.3, 0.5),  # Move the legend left
        title="Legend",
        fontsize=8,
        frameon=False  # Remove legend box outline
    )
    ax.add_artist(legend1)  # keep legend1 on the axes


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def add_activations(ax, activations, env, only_add_feat_on_its_goal=True, unique_symbol_for_centers=False):
    """
    For each cell that has one or more feature activations, plot a small marker
    (circle or square) in one of the four corners. The color is unique for each feat,
    and the marker's transparency (alpha) is proportional to the activation value.
    """
    # First, collect activations per cell and gather the unique feat identifiers.
    # A feat is uniquely identified by (symbol, feat).
    cell_to_feats = {}   # key: (y,x) cell coordinate; value: list of tuples (feat_id, activation)
    unique_feats = []    # list of all unique (symbol, feat) pairs

    for symbol, features in activations.items():
        for feat, cell_dict in features.items():
            feat_id = (symbol, feat)
            unique_feats.append(feat_id)
            for (y, x), activation in cell_dict.items():
                if only_add_feat_on_its_goal and env.MAP[y, x] != symbol:
                    continue
                if (y, x) not in cell_to_feats:
                    cell_to_feats[(y, x)] = []
                cell_to_feats[(y, x)].append((feat_id, activation))

    # Create a unique color for each feature using a colormap.
    cmap = plt.get_cmap('tab10')
    feat_colors = {}
    for i, feat_id in enumerate(unique_feats):
        feat_colors[feat_id] = cmap(i % 10)

    # Now, for each cell, plot markers at the corners.
    # We assume that each cell (with top-left at (x,y)) spans x -> x+1 and y -> y+1.
    # Here we use fixed offsets (0.2 and 0.8) so that up to 4 markers (one per corner) fit.
    for (y, x), feat_list in cell_to_feats.items():
        # Sort for consistency (so the same RBF always goes to the same corner).
        feat_list = sorted(feat_list, key=lambda tup: (tup[0][0], tup[0][1]))
        # Define corner offsets; here the order is: top-left, top-right, bottom-left, bottom-right.
        corner_offsets = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
        for i, (feat_id, activation) in enumerate(feat_list):
            if i >= len(corner_offsets):
                break  # In case there are >4 overlapping activations.
            offset_x, offset_y = corner_offsets[i]
            # The marker will be placed at (x + offset_x, y + offset_y)
            marker_x = x + offset_x
            marker_y = y + offset_y
            marker_size = 10 + 8 * activation  # 10 when activation=0, 18 when activation=1
            # If the current cell is the RBF center we use a square ('s'); otherwise, we use a circle ('o').
            marker_style = 's' if unique_symbol_for_centers and (y, x) == (feat_id[1][0], feat_id[1][1]) else 'o'
            ax.scatter(marker_x, marker_y, s=marker_size, marker=marker_style,
                       color=feat_colors[feat_id],
                       alpha=activation,
                       edgecolors='k', zorder=3)
    return unique_feats

def add_policy_indices(ax, policy_indices, arrow_data, fontsize=4, color="black"):
    """
    Adds policy indices as text annotations to each cell in the grid.
    The position of the text depends on the arrow's direction in arrow_data:
      - For arrows pointing left/right, the text is placed at the top middle of the cell.
      - For arrows pointing up/down, the text is placed on the right-hand side of the cell.

    Args:
        ax: Matplotlib axis object.
        policy_indices: Dictionary mapping (y, x) cell coordinates to an integer.
        arrow_data: Tuple of (x_pos, y_pos, x_dir, y_dir, arrow_colors, arrow_coords).
                    The arrow_coords element is a list of (y, x) tuples corresponding to the arrows.
        fontsize: Font size of the text.
        color: Color of the text.
    """
    # Unpack the arrow_data tuple.
    x_pos, y_pos, x_dir, y_dir, arrow_colors, arrow_coords = arrow_data

    for (y, x), index in policy_indices.items():
        # Default placement if no arrow info is available.
        text_x = x + 0.5
        text_y = y + 0.5

        try:
            # Look for the cell in the arrow_coords.
            idx = arrow_coords.index((y, x))
            # Get the corresponding arrow direction.
            xd = x_dir[idx]
            yd = y_dir[idx]
            # If the arrow is pointing left or right.
            if abs(xd) == 1 and yd == 0:
                text_x = x + 0.5  # center horizontally
                text_y = y + 0.2  # near the top edge
            # If the arrow is pointing up or down.
            elif abs(yd) == 1 and xd == 0:
                text_x = x + 0.8  # near the right edge
                text_y = y + 0.5  # center vertically
            # (You can add additional logic for diagonal arrows if needed.)
        except ValueError:
            # If the current cell is not in arrow_coords, keep the default.
            pass

        ax.text(text_x, text_y, str(index),
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=fontsize,
                color=color,
                zorder=4)


def plot_maxqvals(w, env, q_table=None, arrow_data=None, policy_index=None, policy_indices=None, rbf_data=None,
                save_path=None, show=True):
    """
    Plot the Q-values (with arrows) on top of a grid, and optionally also plot the
    RBF activation markers in the cell corners. Optionally save the plot if save_path is given,
    and only show the plot if show is True.

    Args:
        w: Weight vector.
        env: Environment object.
        q_table: The Q-table (sf).
        arrow_data: Either pass Q-table or pass arrow data
        policy_index: Index of the current policy.
        policy_indices: A dictionary mapping each state to the GPI policy index of the max Q-val over policies and
        actions.
        rbf_data: (Optional) RBF activation data.
        save_path: (Optional) Path to save the figure.
        show: (Optional) If True, display the plot (default True).
    """
    if q_table is None and arrow_data is None:
        raise Exception("Pass a q-table or arrow data")

    if arrow_data is None:
        arrow_data = get_plot_arrow_params(q_table, w, env)  # e.g., returns (x_pos, y_pos, x_dir, y_dir, color)

    fig, ax = plt.subplots()

    grid, mapping = convert_map_to_grid(env, custom_mapping=env.QVAL_COLOR_MAP)
    create_grid_plot(ax, grid)  # Draw the grid cells.
    add_legend(ax, mapping)     # Add legend (obstacles, goals, etc.)

    # Format the weight vector as a string for the title
    if rbf_data is None:
        rounded_weights = np.round(w, decimals=2)
        weight_str = np.array2string(rounded_weights, precision=2, separator=", ")
        title = f"Policy {policy_index} | Weights: {weight_str}" if policy_index is not None else f"Weights: {weight_str}"
        ax.set_title(title)

    # Plot the arrows that indicate the policy's best actions
    quiv = plot_policy(ax, arrow_data, values=True)
    plt.show()


def plot_q_vals(w, env, q_table=None, arrow_data=None, policy_index=None, policy_indices=None, activation_data=None,
                save_path=None, show=True, unique_symbol_for_centers=False):
    """
    Plot the Q-values (with arrows) on top of a grid, and optionally also plot the
    RBF activation markers in the cell corners. Optionally save the plot if save_path is given,
    and only show the plot if show is True.

    Args:
        w: Weight vector.
        env: Environment object.
        q_table: The Q-table (sf).
        arrow_data: Either pass Q-table or pass arrow data
        policy_index: Index of the current policy.
        policy_indices: A dictionary mapping each state to the GPI policy index of the max Q-val over policies and
        actions.
        rbf_data: (Optional) RBF activation data.
        save_path: (Optional) Path to save the figure.
        show: (Optional) If True, display the plot (default True).
    """
    if q_table is None and arrow_data is None:
        raise Exception("Pass a q-table or arrow data")

    if arrow_data is None:
        arrow_data = get_plot_arrow_params(q_table, w, env)  # e.g., returns (x_pos, y_pos, x_dir, y_dir, color)

    fig, ax = plt.subplots()

    grid, mapping = convert_map_to_grid(env, custom_mapping=env.QVAL_COLOR_MAP)
    create_grid_plot(ax, grid)  # Draw the grid cells.
    add_legend(ax, mapping)     # Add legend (obstacles, goals, etc.)

    # Format the weight vector as a string for the title
    if activation_data is None:
        rounded_weights = np.round(w, decimals=2)
        weight_str = np.array2string(rounded_weights, precision=2, separator=", ")
        title = f"Policy {policy_index} | Weights: {weight_str}" if policy_index is not None else f"Weights: {weight_str}"
        ax.set_title(title)

    # Plot the arrows that indicate the policy's best actions
    quiv = plot_policy(ax, arrow_data, values=False)

    # If policy_indices are provided, add them to each cell.
    if policy_indices is not None:
        add_policy_indices(ax, policy_indices, arrow_data)

    # If RBF data is provided, overlay the RBF activation markers
    if activation_data is not None:
        unique_feats = add_activations(ax, activation_data, env, unique_symbol_for_centers=unique_symbol_for_centers)

        # build a mapping from each feat_id to its unique color
        cmap = plt.get_cmap('tab10')
        feat_colors = {feat_id: cmap(i % 10) for i, feat_id in enumerate(unique_feats)}

        # Build a mapping from weight index to the color of its corresponding RBF
        # env.rbf_indices maps center_coords -> weight index
        weight_to_color = {}
        for prop in env.PHI_OBJ_TYPES:
            for feat in env.FEAT_DATA[prop]:
                weight_idx = env.get_feat_idx(prop, feat)
                feat_id = (prop, feat)
                weight_to_color[weight_idx] = feat_colors[feat_id]

        from matplotlib.lines import Line2D

        # Create custom legend handles using square markers
        handles = []
        for i, weight in enumerate(w):
            color = weight_to_color.get(i, 'gray')
            handle = Line2D([], [], marker='s', color='none',
                            markerfacecolor=color, markersize=8,
                            label=f"w[{i}]={weight:.2f}")
            handles.append(handle)

        # Place the legend above the plot
        legend2 = ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.10),
            ncol=len(handles),
            handlelength=1,
            handletextpad=0.2,
            columnspacing=0.5,
            frameon=False
        )

    # Save the figure if a save_path is provided.
    if save_path is not None:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, bbox_inches='tight')

    # Show or close the plot.
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_rbfs(rbf_data, grid_size, env, aggregation="sum", skip_non_goal=True,
                  colors_symbol_centers=None, seperate_plots_for_goals=True):
    """
    Plots RBF activations. If `seperate_plots_for_goals` is True, creates a subplot for each symbol;
    otherwise, aggregates all activations into a single heatmap.

    Args:
        rbf_data: Dictionary of RBF activations per center.
        grid_size: Tuple (grid_height, grid_width) of the environment.
        env: Environment object that contains the MAP attribute.
        aggregation: Method to combine overlapping activations ("sum" or "max").
        skip_non_goal: If True, ignores activations where the map cell doesn't match the symbol.
        colors_symbol_centers: Mapping from symbol to color for the center markers.
        seperate_plots_for_goals: If True, plots each symbol on a separate subplot.
    """
    if colors_symbol_centers is None:
        colors_symbol_centers = RBF_COLORS

    grid_height, grid_width = grid_size

    if seperate_plots_for_goals:
        num_symbols = len(rbf_data)
        # Create one row of subplots for each symbol, with an increased overall figure size.
        fig, axes = plt.subplots(nrows=1, ncols=num_symbols, figsize=(10 * num_symbols, 8), squeeze=False)
        axes = axes[0]  # Only one row of subplots

        for ax, (symbol, features) in zip(axes, rbf_data.items()):
            activation_grid = np.zeros((grid_height, grid_width))
            # Compute activations for this symbol.
            for _, activations in features.items():
                for (y, x), activation_value in activations.items():
                    if skip_non_goal and not env.MAP[y, x] == symbol:
                        activation_value = 0  # Ignore activations not matching the symbol.
                    if aggregation == "sum":
                        activation_grid[y, x] += activation_value
                    elif aggregation == "max" and activation_value > activation_grid[y, x]:
                        activation_grid[y, x] = activation_value

            im = ax.imshow(activation_grid, cmap="hot", origin="upper",
                           extent=(0, grid_width, 0, grid_height))
            # Plot RBF centers without legend.
            center_color = colors_symbol_centers.get(symbol, "white")
            for (cy, cx, d) in features.keys():
                ax.scatter(cx + 0.5, grid_height - cy - 0.5, color=center_color, s=100,
                           edgecolors="black")
            ax.set_title(f"RBF Activations for {symbol}", fontsize=14)
            ax.grid(False)
            # Add a colorbar without axis labels.
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Adjust layout: add extra horizontal spacing between the subplot groups.
        plt.tight_layout(pad=3.0, w_pad=3.0)
        fig.subplots_adjust(wspace=0.2)
        plt.show()
    else:
        # Aggregated plot for all RBF activations in one heatmap.
        combined_activation_grid = np.zeros((grid_height, grid_width))
        fig, ax = plt.subplots(figsize=(10, 10))

        for symbol, features in rbf_data.items():
            for _, activations in features.items():
                for (y, x), activation_value in activations.items():
                    if skip_non_goal and not env.MAP[y, x] == symbol:
                        activation_value = 0
                    if aggregation == "sum":
                        combined_activation_grid[y, x] += activation_value
                    elif aggregation == "max" and activation_value > combined_activation_grid[y, x]:
                        combined_activation_grid[y, x] = activation_value

        im = ax.imshow(combined_activation_grid, cmap="hot", origin="upper",
                       extent=(0, grid_width, 0, grid_height))
        # Plot RBF centers and add legend entry only once per symbol.
        legend_added = {}
        for symbol, features in rbf_data.items():
            center_color = colors_symbol_centers.get(symbol, "white")
            for (cy, cx, d) in features.keys():
                label = f"RBF {symbol}" if symbol not in legend_added else None
                ax.scatter(cx + 0.5, grid_height - cy - 0.5, color=center_color, s=100,
                           edgecolors="black", label=label)
                legend_added[symbol] = True

        ax.set_title("All RBF Activations Combined", fontsize=14)
        ax.grid(False)
        # Remove duplicate legend entries.
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout(pad=3.0)
        plt.show()
