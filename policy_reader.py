import pickle as pkl
import numpy as np


if __name__ == "__main__":


    policies = [1,2]

    for i in policies:
            with open(f"policies/hallway/discovered_policy_{i}.pkl", "rb") as fp:
                policy = pkl.load(fp)
            print(policy["reward"])
            q = policy["q_table"]

            for obs in q:
                print(obs, q[obs])