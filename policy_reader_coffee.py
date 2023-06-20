import pickle as pkl
import numpy as np
import os
import argparse

if __name__ == "__main__":

    # This is to check how the SF representation of the discovered policies look like

    actions = ["LEFT", "UP", "RIGHT", "DOWN"]

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
        w = np.asarray([0.7737809366318175, 0.95])
        qc1 = np.dot(q[(6, 6, 0, 0)], w)
        # qc2 = np.dot(q[(3, 6, 0, 0)], w)
        print('c1', np.max(qc1), actions[np.argmax(qc1)], qc1)
        # print('c2', np.max(qc2), np.argmax(qc2))
        print(15 * '--')
