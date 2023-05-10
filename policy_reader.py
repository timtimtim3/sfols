import pickle as pkl
import numpy as np


if __name__ == "__main__":

    # This is to check how the SF representation of the discovered policies look like

    policies = [1, 2, 3, 4, 5]

    for i in policies:
        with open(f"policies/hallway-multiple/discovered_policy_{i}.pkl", "rb") as fp:
            policy = pkl.load(fp)

        print(f"\nPolicy {i}")
        print(policy["reward"])
        q = policy["q_table"]

        ss = sorted(q.keys())

        for obs in ss:
            print(obs, 'LEFT', np.round(q[obs][0], 16))
            print(obs, 'RIGHT', np.round(q[obs][1], 16))
            print(15 * '--')
