from itertools import combinations
from typing import List, Optional

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import wandb

import wandb as wb
from sfols.rl.utils.utils import random_weights, hypervolume, policy_evaluation_mo

np.set_printoptions(precision=4)

class OLS:
    # Section 3.3 of http://roijers.info/pub/thesis.pdf
    def __init__(
        self,
        m: int,
        epsilon: float = 0.0,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        reverse_extremum: bool = False,
        restriction : Optional[dict] = None, 
    ):
        self.m = m
        self.epsilon = epsilon
        self.W = []
        self.ccs = []
        self.ccs_weights = []
        self.queue = []
        self.iteration = 0
        self.max_value = max_value
        self.min_value = min_value
        self.worst_case_weight_repeated = False
        self.extremum_weights = reversed(self.extrema_weights(restriction)) if reverse_extremum else self.extrema_weights(restriction)
        for w in self.extremum_weights:
            self.queue.append((float("inf"), w))
        self.define_wandb_metrics()
    def next_w(self) -> np.ndarray:
        return self.queue.pop(0)[1]

    def get_ccs_weights(self) -> List[np.ndarray]:
        return self.ccs_weights.copy()

    def get_corner_weights(self, top_k: Optional[int] = None) -> List[np.ndarray]:
        weights = [w for (p, w) in self.queue]
        if top_k is not None:
            return weights[:top_k]
        else:
            return weights

    def ended(self) -> bool:
        return len(self.queue) == 0 #or self.worst_case_weight_repeated

    def add_solution(self, value, w, gpi_agent=None, env=None, learn_all_extremum=False) -> int:
        self.iteration += 1
        self.W.append(w)

        if self.is_dominated(value):  # This is not in the pseudoode
            w_table = wandb.Table(columns=[f"w{i}" for i in range(self.m)], data=[(c.tolist()) for c in self.ccs])
            wandb.log({
                "ols/|W_del|": 1,
                "ols/|W_corner|": 0,
                "ols/|Q|": len(self.queue),
                "ols/ccs_size": len(self.ccs),
                "ols/ccs": w_table,
                "ols_iter": self.iteration

            })
            print("Deleting new policy because it is dominated")
            return [len(self.ccs)]

        for i, v in enumerate(self.ccs):
            if np.allclose(v, value):
                w_table = wandb.Table(columns=[f"w{i}" for i in range(self.m)], data=[(c.tolist()) for c in self.ccs])
                wandb.log({
                    "ols/|W_del|": 1,
                    "ols/|W_corner|": 0,
                    "ols/|Q|": len(self.queue),
                    "ols/ccs_size": len(self.ccs),
                    "ols/ccs": w_table,
                    "ols_iter": self.iteration

                })
                print("Deleting new policy because it has same val as old one")
                return [len(self.ccs)]  # delete new policy as it has same value as an old one

        if learn_all_extremum and len(self.queue) > 0 and self.queue[0][0] == float("inf"):
            print("Skipping removing obsolete weights from queue")
            W_del = []
        else:
            W_del = self.remove_obsolete_weights(new_value=value)
            W_del.append(w)

        removed_indx = self.remove_obsolete_values(value)  # Not sure why this is here
        self.ccs.append(value)
        self.ccs_weights.append(w)

        if learn_all_extremum and len(self.queue) > 0 and self.queue[0][0] == float("inf"):
            print("Skipping adding new weigths to queue")
            W_corner = []
        else:
            W_corner = self.new_corner_weights(value, W_del)

            for wc in W_corner:
                priority = self.get_priority(wc, gpi_agent, env)
                print("improv.", priority)
                if priority > self.epsilon:
                    self.queue.append((priority, wc))
            self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority

        rows = [
            np.asarray(c).reshape(-1).tolist()
            for c in self.ccs
        ]
        w_table = wb.Table(
            columns=[f"w{i}" for i in range(self.m)],
            data=rows
        )
        wandb.log({
            "ols/|W_del|": len(W_del),
            "ols/|W_corner|": len(W_corner),
            "ols/|Q|": len(self.queue),
            "ols/ccs_size": len(self.ccs),
            "ols/ccs": w_table,
            "ols_iter": self.iteration

        })
        return removed_indx


    def check_solution(self, gpi_agent=None, env=None) -> int:
        W_corner = self.new_corner_weights_new(None)

        print("W_corner", W_corner)
        for wc in W_corner:
            priority = self.get_priority(wc, gpi_agent, env)
            print("improv.", priority)
            if priority > self.epsilon:
                self.queue.append((priority, wc))
        self.queue.sort(key=lambda t: t[0], reverse=True)  # Sort in descending order of priority

        print("ccs:", self.ccs)
        print("ccs size:", len(self.ccs))

        return len(self.queue)


    def new_corner_weights_new(self, v_new: np.ndarray) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []

        W_new = self.corner_weights_new()

        filter_fn = lambda wc: (wc is not None) and (
            not any([np.allclose(wc, w_old) for w_old in self.W] + [np.allclose(wc, w_old) for p, w_old in self.queue]))
        # (np.isclose(np.dot(wc, v_new), self.max_scalarized_value(wc))) and \
        W_new = list(filter(filter_fn, W_new))
        W_new = np.unique(W_new, axis=0)
        return W_new


    def corner_weights_new(self):
        # Based on https://stackoverflow.com/questions/65343771/solve-linear-inequalities
        A = np.vstack(self.ccs)
        A = np.round_(A, decimals=4)
        A = np.concatenate((A, -np.ones(A.shape[0]).reshape(-1, 1)), axis=1)

        bla = np.ones(A.shape[1]).reshape(1, -1)
        bla[0, -1] = 0
        A = np.concatenate((A, bla), axis=0)
        bla = -np.ones(A.shape[1]).reshape(1, -1)
        bla[0, -1] = 0
        A = np.concatenate((A, bla), axis=0)

        for i in range(self.m):
            bla = np.zeros(A.shape[1]).reshape(1, -1)
            bla[0, i] = -1
            A = np.concatenate((A, bla), axis=0)

        b = np.zeros(len(self.ccs) + 2 + self.m)
        b[len(self.ccs)] = 1
        b[len(self.ccs) + 1] = -1

        import cdd
        def compute_poly_vertices(A, b):
            b = b.reshape((b.shape[0], 1))
            mat = cdd.Matrix(np.hstack([b, -A]), number_type='float')
            mat.rep_type = cdd.RepType.INEQUALITY
            P = cdd.Polyhedron(mat)
            g = P.get_generators()
            V = np.array(g)
            vertices = []
            for i in range(V.shape[0]):
                if V[i, 0] != 1: continue
                if i not in g.lin_set:
                    vertices.append(V[i, 1:])
            return vertices

        vertices1 = compute_poly_vertices(A, b)
        corner = []
        for v in vertices1:
            corner.append(v[:-1])
        return corner

    def get_priority(self, w, gpi_agent=None, env=None) -> float:
        max_optimistic_value = self.max_value_lp(w)
        max_value_ccs = self.max_scalarized_value(w)
        # upper_bound_nemecek = self.upper_bound_policy_caches(w)
        # print(f'optimistic: {max_optimistic_value} policy_cache_up: {upper_bound_nemecek}')
        # if gpi_agent is not None:
        #     gpi_value = policy_evaluation_mo(gpi_agent, env, w, rep=1)
        #     gpi_value = np.dot(gpi_value, w)
        #     print(f"optimistic: {max_optimistic_value:.4f} smp: {max_value_ccs:.4f} gpi: {gpi_value:.4f}")
            # max_value_ccs = max(max_value_ccs, gpi_value)
        priority = max_optimistic_value - max_value_ccs  # / abs(max_optimistic_value)
        return priority

    def max_scalarized_value(self, w: np.ndarray) -> float:
        if not self.ccs:
            return None
        return np.max([np.dot(v, w) for v in self.ccs])

    def get_set_max_policy_index(self, w: np.ndarray) -> int:
        if not self.ccs:
            return None
        return np.argmax([np.dot(v, w) for v in self.ccs])

    def remove_obsolete_weights(self, new_value: np.ndarray) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []
        W_del = []
        inds_remove = []
        for i, (priority, cw) in enumerate(self.queue):
            if np.dot(cw, new_value) > self.max_scalarized_value(cw):  # and priority != float('inf'):
                W_del.append(cw)
                inds_remove.append(i)
        for i in reversed(inds_remove):
            self.queue.pop(i)
        return W_del

    def remove_obsolete_values(self, value: np.ndarray) -> List[int]:
        # self.ccs holds previous average GPI SF vector (phi_dim,) over inital states under corresponding weight
        # that was added in that iteration to self.css_weights

        removed_indx = []
        # Loop in reverse over previous GPI SF vectors (each computed by maxing over actions under w', max_a phi * w')
        # where w' is the one that was learned and added in that iteration to self.ccs_weights
        for i in reversed(range(len(self.ccs))):
            current_gpi_best_in_all = True
            gpi_sf_i = self.ccs[i]

            # Loop over all previously learned w vectors
            for j in range(len(self.W)):
                w = self.W[j]
                if np.dot(value, w) < np.dot(gpi_sf_i, w):
                    current_gpi_best_in_all = False
                    break

            # if current_gpi_best_in_all is True this means it beats the average SF of gpi_sf_i in all tasks
            # we have previously learned and we remove the gpi_sf_i

            # I don't know if this makes sense though because the SF under current GPI policy depend on and are computed
            # under a specific weight vector w (the one that just learned), the same goes for the GPI policy gpi_sf_i
            # that we are comparing to, it was computed under its own w' vector and depends on it
            # So then computing their value under some other weight vector w'' above using SF that depend on totally
            # different weight vectors w' and w? Maybe makes more sense to compute SF of current_gpi_sf and gpi_sf_i
            # using the weight vector w'' we're using to compute the value. e.g. if we're dealing with the last element
            # of self.ccs (so -1) we would simply use self.policies[:-1] and for current_gpi_sf we use self.policies[:]?
            if current_gpi_best_in_all:
                print("removed value", self.ccs[i])
                print("removed weights", self.ccs_weights[i])
                removed_indx.append(i)
                self.ccs.pop(i)
                self.ccs_weights.pop(i)
        return removed_indx

    def max_value_lp(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        v = cp.Variable(self.m)
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Maximize(w @ v)
        constraints = [W @ v <= V]
        if self.max_value is not None:
            constraints.append(v <= self.max_value)
        if self.min_value is not None:
            constraints.append(v >= self.min_value)
        prob = cp.Problem(objective, constraints)
        return prob.solve(verbose=False)

    def upper_bound_policy_caches(self, w_new: np.ndarray) -> float:
        if len(self.ccs) == 0:
            return float("inf")
        w = cp.Parameter(self.m)
        w.value = w_new
        alpha = cp.Variable(len(self.W))
        W_ = np.vstack(self.W)
        V_ = np.array([self.max_scalarized_value(weight) for weight in self.W])
        W = cp.Parameter(W_.shape)
        W.value = W_
        V = cp.Parameter(V_.shape)
        V.value = V_
        objective = cp.Minimize(alpha @ V)
        constraints = [alpha @ W == w, alpha >= 0]
        prob = cp.Problem(objective, constraints)
        upper_bound = prob.solve()
        if prob.status == cp.OPTIMAL:
            return upper_bound
        else:
            return float("inf")

    def worst_case_weight(self) -> np.ndarray:
        if len(self.W) == 0:
            return random_weights(dim=self.m)
        w = None
        min = float("inf")
        w_var = cp.Variable(self.m)
        params = []
        for v in self.ccs:
            p = cp.Parameter(self.m)
            p.value = v
            params.append(v)
        for i in range(len(self.ccs)):
            objective = cp.Minimize(w_var @ params[i])
            constraints = [0 <= w_var, cp.sum(w_var) == 1]
            # constraints = [cp.norm(w_var) - 1 <= 0, 0 <= w_var]
            for j in range(len(self.ccs)):
                if i != j:
                    constraints.append(w_var @ (params[j] - params[i]) <= 0)
            prob = cp.Problem(objective, constraints)
            value = prob.solve()
            if value < min and prob.status == cp.OPTIMAL:
                min = value
                w = w_var.value.copy()

        if np.allclose(w, self.W[-1]):
            self.worst_case_weight_repeated = True

        return w

    def new_corner_weights(self, v_new: np.ndarray, W_del: List[np.ndarray]) -> List[np.ndarray]:
        if len(self.ccs) == 0:
            return []
        V_rel = []
        W_new = []
        for w in W_del:
            best = [self.ccs[0]]
            for v in self.ccs[1:]:
                if np.allclose(np.dot(w, v), np.dot(w, best[0])):
                    best.append(v)
                elif np.dot(w, v) > np.dot(w, best[0]):
                    best = [v]
            V_rel += best
        B_rel = self.extrema_weights()
        V_rel.extend(B_rel)
        V_rel = np.unique(V_rel, axis=0)
        # V_rel = self.ccs.copy()
        # for comb in range(1, self.m):
        for x in combinations(V_rel, self.m-1):
            if not x:
                continue
            wc = self.corner_weight(v_new, x)
            if wc is None:
                continue
            new_v = np.dot(wc, v_new)
            max_v = self.max_scalarized_value(wc)
            if np.allclose(new_v, max_v):
                W_new.append(wc)
        filter_fn = lambda wc: (wc is not None) and (not any([np.allclose(wc, w_old) for w_old in self.W] + [np.allclose(wc, w_old) for p, w_old in self.queue]))
        # (np.isclose(np.dot(wc, v_new), self.max_scalarized_value(wc))) and \
        W_new = list(filter(filter_fn, W_new))
        W_new = np.unique(W_new, axis=0)
        return W_new

    def corner_weight(self, v_new: np.ndarray, v_set: List[np.ndarray]) -> np.ndarray:
        wc = cp.Variable(self.m)
        v_n = cp.Parameter(self.m)
        v_n.value = v_new
        objective = cp.Minimize(v_n @ wc)  # cp.Minimize(0)
        constraints = [0 <= wc, cp.sum(wc) == 1]
        for v in v_set:
            v_par = cp.Parameter(self.m)
            v_par.value = v
            constraints.append(v_par @ wc == v_n @ wc)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver='SCS', verbose=False, eps=1e-5)  # (solver='SCS', verbose=False, eps=1e-5)
        if prob.status == cp.OPTIMAL:
            weight = np.clip(wc.value, 0, 1)  # ensure range [0,1]
            weight /= weight.sum()  # ensure sum to one
            return weight
        else:
            return None

    def extrema_weights(self, restriction=None) -> List[np.ndarray]:
        extrema_weights = []
        for i in range(self.m):
            w = np.zeros(self.m)
            w[i] = 1.0
            if restriction:
                for c in restriction:
                    w[c] = restriction[c]
            extrema_weights.append(w)
        return extrema_weights

    def is_dominated(self, value):
        for v in self.ccs:
            if (v >= value).all() and (v > value).any():
                return True
        return False

    @staticmethod
    def define_wandb_metrics():
        wandb.define_metric("ols_iter")
        wandb.define_metric("ols/*", step_metric="ols_iter")
        wandb.define_metric("ols/css", summary="last", step_metric="ols_iter")

    def plot_ccs(self, ccs, ccs_weights, gpi_agent=None, eval_env=None):
        import seaborn as sns
        params = {
            "text.latex.preamble": r"\usepackage{amsmath}",
            "mathtext.fontset": "cm",
            "figure.figsize": (1.5 * 4.5, 1.5 * 3),
            "xtick.major.pad": 0,
            "ytick.major.pad": 0,
        }
        plt.rcParams.update(params)
        sns.set_style("white", rc={"xtick.bottom": False, "ytick.left": False})
        sns.set_context(
            "paper",
            rc={
                "text.usetex": True,
                "lines.linewidth": 2,
                "font.size": 15,
                "figure.autolayout": True,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "axes.titlesize": 12,
                "axes.labelsize": 15,
                "lines.markersize": 12,
                "legend.fontsize": 14,
            },
        )
        colors = ["#E6194B", "#5CB5FF"]
        sns.set_palette(colors)

        x_css, y_css = [], []
        for i in range(len(ccs)):
            x_css.append(ccs[i][0])
            y_css.append(ccs[i][1])

        x, y = [], []
        for i in range(len(self.ccs)):
            x.append(self.ccs[i][0])
            y.append(self.ccs[i][1])

        if gpi_agent is not None:
            x_gpi, y_gpi = [], []
            for w in ccs_weights:
                value = policy_evaluation_mo(gpi_agent, eval_env, w, rep=5)
                x_gpi.append(value[0])
                y_gpi.append(value[1])

        plt.figure()
        if gpi_agent is not None:
            plt.scatter(
                x_gpi,
                y_gpi,
                label="$\Psi^{\mathrm{GPI}}$ (GPI-expanded SF set)",
                color=colors[0],
            )
        plt.scatter(
            x,
            y,
            label="$\Psi$ (SF set at iteration {})".format(self.iteration),
            marker="^",
            color=colors[1],
        )
        plt.scatter(x_css, y_css, label="CCS", marker="x", color="black")
        plt.legend(loc="lower left", fancybox=True, framealpha=0.5)
        plt.xlabel("$\psi^{\pi}_{1}$ (Treasure Value)")
        plt.ylabel("$\psi^{\pi}_{2}$ (Time Value)")
        sns.despine()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(f"figs/ccs_dst{self.iteration}.pdf", format="pdf", bbox_inches="tight")

        wb.log(
            {
                "metrics/ccs": wb.Image(plt),
                "global_step": gpi_agent.policies[-1].num_timesteps,
                "iteration": self.iteration,
            }
        )


if __name__ == "__main__":

    def solve(w):
        return np.array(list(map(float, input().split())), dtype=np.float32)

    m = 4
    ols = OLS(m=m, epsilon=0.0001) #, min_value=0.0, max_value=1 / (1 - 0.95) * 1)
    while not ols.ended():
        w = ols.next_w()
        print("w:", w)
        value = solve(w)
        ols.add_solution(value, w)

        print("hv:", hypervolume(np.zeros(m), ols.ccs))