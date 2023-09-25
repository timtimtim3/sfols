import pickle as pkl
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import gym
import envs
from plotting.plotting import create_grid_plot, plot_policy, get_plot_arrow_params

action_dict = ["LEFT", "UP", "RIGHT", "DOWN"]
PLOTS = True
# LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

if __name__ == "__main__":

    # This is to check how the SF representation of the discovered policies look like

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    dir = args.input
    env = gym.make(args.input)
    map = env.MAP

    dirpath = os.path.abspath(f"policies/{dir}")
    for i, file in enumerate(f for f in os.listdir(dirpath) if os.path.isfile(f"{dirpath}/{f}")):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)

        print(f"\nPolicy {i}")
        # print(policy["reward"])
        q = policy["q_table"]
        int_keys = [key for key in q.keys() if isinstance(key, int)]
        for key in int_keys:
            del q[key]
        ss = sorted(q.keys())

        for obs in ss:
            for j in range(q[obs].shape[0]):

                print(obs, action_dict[j], np.round(q[obs][j], 16))

            print(15 * '--')
        if PLOTS:
            plt.figure(i)
            ax = plt.subplot(111)
            create_grid_plot(ax=ax, grid=map != 'X')
            start_map = np.zeros_like(map, dtype=bool)
            for char in env.PHI_OBJ_TYPES:
                start_map = np.logical_or(start_map, map == char)

            create_grid_plot(ax=ax, grid=start_map, color_map="YlGn")
            quiv = plot_policy(
                ax=ax, arrow_data=get_plot_arrow_params(q, policy["w"]), grid=map,
                values=False, max_index=False
            )
            # plt.show()
            plot_dir = f"{dirpath}/plots"
            if not os.path.isdir(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(f"{plot_dir}/policy{i}.pdf", format='pdf', dpi=1000, pad_inches=0, bbox_inches='tight')