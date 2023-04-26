import pickle as pkl
import numpy as np


if __name__ == "__main__":

    with open("discovered_policy_1.pkl", "rb") as fp:
        policy = pkl.load(fp)

    q = policy["q_table"]

    for obs in q:
        print(obs, q[obs])