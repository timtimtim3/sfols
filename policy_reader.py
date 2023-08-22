import pickle as pkl
import numpy as np
import os
import argparse


action_dict = ["LEFT", "RIGHT"]
# LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3

if __name__ == "__main__":

    # This is to check how the SF representation of the discovered policies look like

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)

    args = parser.parse_args()
    dir = args.input

    dirpath = os.path.abspath(f"policies/{dir}")

    for i, file in enumerate(os.listdir(dirpath)):
        with open(os.path.join(dirpath, file), "rb") as fp:
            policy = pkl.load(fp)

        print(f"\nPolicy {i}")
        print(policy["reward"])
        q = policy["q_table"]

        ss = sorted(q.keys())

        for obs in ss:

            for i in range(q[obs].shape[0]):

                print(obs, action_dict[i], np.round(q[obs][i], 16))

            print(15 * '--')
