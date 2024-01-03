import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from tqdm.notebook import tqdm as tqdm_notebook

from recidivism import *

plt.rcParams["font.size"] = "16"
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 200

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"


def generate_gamma_p(n):
    p0 = np.random.gamma(2, 2, n) + 0.1
    p0 = p0 / (p0.max() + 0.1) / 2
    h = logit(p0)
    b = h.min() - 13
    a = 1
    sigma0 = (h - b) / a
    p = LinearRecidProb(a, b, sigma0.copy())
    return p


def run(
    scope,
    state,
    actions=None,
    verbose=False,
    options_mix=2,
    zeta=None,
    func_cost_function=None,
    rate_exogenous_arrival=0.4,
    rate_state_decay=1.0,
    rate_state_decay_last=0.5,
):
    traj = []
    # keep input untouched
    st: RecidState = copy.deepcopy(state)

    # iterate on st
    for t in range(0, scope):
        ###############################################################################
        # nomenclature:
        #  x(t) := sto
        #  x(t+1) := st
        ###############################################################################
        # return the old state: sto
        #   record the new population: st
        sto: RecidState = st.forward(
            rate_exogenous_arrival=rate_exogenous_arrival,
            rate_state_decay=rate_state_decay,
            rate_state_decay_last=rate_state_decay_last,
        )
        action: Decision = actions[t]

        # y(t)
        # apply action to the old state, x(t)
        action.forward(sto)
        # update if needed, e.g., policy optimization
        action.backward()
        sto.evaluate(action, func_cost=func_cost_function)
        # keep the trajectory at t
        traj.append(sto)

        pr: RecidProb = st.p
        # probability recalculation
        # this corresponds to \rho(t+1) since the population remixes
        pr.p = (1 / st.x) * (sto.Phi.T @ (pr.p * sto.x) + pr.Q * sto.lbda)
        pr.p = pr.p * (st.x > 1e-2)
        pr.p = np.nan_to_num(pr.p, nan=1.0)

        pr.reverse()
        ###################################################
        # mix-in effects
        ###################################################
        # mix-in effects will be affective on x(t+1): st
        # compute an intermediate probability
        # this corresponds to \tilde \rho(t)
        if options_mix == 0:
            # random policy, choose gamma to decease
            # reset these people to zero
            raise ValueError("not known for option 0")
        elif options_mix == 1:
            # the mix-in happens to everybody
            pr.sigma = pr.sigma + (sto.recid_treat.sum(0) @ zeta / st.n * 20 * 0.2)
            pr.update()
        elif options_mix == 2:
            # the mix-in happens to everybody except for incarcerated ones

            pr.sigma = pr.sigma + (
                sto.recid_treat.sum(0)
                @ zeta
                / st.n
                * (1 - sto.recid_no_treat.sum(0))
                * 0.2
            )
            pr.update()
        else:
            raise ValueError(f"not implemented :options_mix = {options_mix}")

    return traj


def grid_search(s0, P0, T, d, space, run_func, **kwargs):
    perfs = []
    for s1 in tqdm_notebook(space):
        for s2 in space:
            actions = [
                RandomizedHierarchicalDecision(d, theta=np.array([s1, s2]))
                for _ in range(T)
            ]
            P = copy.deepcopy(P0)
            state = copy.deepcopy(s0)
            traj = run_func(T, P, state, actions, **kwargs)
            perf = evaluate(traj)
            perfs.append(pd.DataFrame(perf))
    return perfs


def show_traj(traj):
    def query_metric(metric, reducer=lambda x: x.sum()):
        if reducer is not None:
            mm = [reducer(getattr(s, metric)) for s in traj]
        else:
            mm = [getattr(s, metric) for s in traj]
        xaxis = range(len(traj))
        return xaxis, mm

    metric = [("s", None), ("total_x", None), ("ri", None), ("rt", None)]
    fig, axs = plt.subplots(2, len(metric) // 2)
    fig.set_size_inches(16, 8)
    for idx, ax in enumerate(axs.reshape(len(metric))):
        _x, _y = query_metric(metric[idx][0], reducer=metric[idx][1])
        ax.plot(_x, _y)
        ax.set_title(f"trajectory of {RecidState.ALIAS[metric[idx][0]]}")


def show_grids(grids, perfs, figure_id, df=None, plot=True):
    # records = [
    #     {
    #         "r": sum(perf["r"]),
    #         "z": sum(perf["z"]).sum(),
    #         "f": sum(perf["f"]),
    #         "ri": sum(perf["ri"]),
    #         "rt": sum(perf["rt"]),
    #     }
    #     for perf in perfs
    # ]

    T = perfs[0]["r"].shape[0]
    print(f"trajectory size: {T}")
    records = [
        {
            "r": np.power(perf["r"], 2).mean(),
            "s": np.power(perf["s"], 2).mean(),
            "f": np.power(perf["f"], 2).mean(),
            "ri": np.power(perf["ri"], 2).mean(),
            "rt": np.power(perf["rt"], 2).mean(),
        }
        for perf in perfs
    ]
    t1, t2 = grids
    if df is None:
        df = pd.DataFrame.from_records(records)
    if not plot:
        return df
    metric = ["rt", "ri", "s", "f"]
    fig, axs = plt.subplots(2, len(metric) // 2)
    fig.set_size_inches(15, 7 * len(metric) // 2)
    plt.rcParams.update({"font.size": 15})
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    for idx, ax in enumerate(axs.reshape(len(metric))):
        sum_c = df[metric[idx]].values.reshape(t1.shape)

        ax.contour(
            1 - t1,
            1 - t2,
            sum_c,
            levels=40,
            linestyles="dashed",
            linewidths=0.2,
            colors="k",
        )
        cntr0 = ax.contourf(1 - t1, 1 - t2, sum_c, levels=40, cmap="RdBu_r")

        fig.colorbar(cntr0, ax=ax)
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"long-run {RecidState.ALIAS[metric[idx]]}")
        ax.set_xlabel("fraction of people treated in $L$")
        ax.set_ylabel("fraction of people treated in $H$")

    plt.show()
    fig.savefig(f"runid-{figure_id}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    return df


def evaluate(traj):
    metrics = ["r", "z", "ri", "rt", "f", "s"]
    return {m: [getattr(r, m) for r in traj] for m in metrics}
