"""
receive treatment and then recidivate

"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import copy
from tqdm.notebook import tqdm as tqdm_notebook

import recidivism_bak as bak

plt.rcParams["font.size"] = "12"
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.dpi"] = 200
import os

os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OMP_NUM_THREADS"] = "6"
from optimization import *


########################################
# recidivism probability
########################################
class RecidProb(object):
    pass


class LinearRecidProb(RecidProb):
    def __init__(self, *args, **kwargs) -> None:
        self.a, self.b, self.sigma, *_ = args
        self.update()

    def update(self, da=None, S=None):
        if da is not None:
            self.a += da
        if S is not None:
            self.sigma = S * self.sigma
        self.P = expit(self.b + self.a * self.sigma)

    def forward(self):
        """recidivate

        Returns:
            _type_: _description_
        """
        return np.random.binomial(1, self.P)


def beta_init_p(n):
    p0 = np.random.gamma(2, 2, n) + 0.1
    p0 = p0 / (p0.max() + 0.1) / 2
    return p0


class RecidWithScaler(RecidProb):
    def __init__(self, P0, n=0, scaler=0) -> None:
        self.P0 = P0
        self.update(n, scaler)

    def update(self, n, scaler):
        self.n = n
        self.scaler = scaler
        self.P = self.basep(n) + scaler

    def forward(self):
        """recidivate

        Returns:
            _type_: _description_
        """
        return np.random.binomial(1, self.P)


########################################
# mix-in effect
########################################
class MixIn(object):
    pass


class LinearMixIn(MixIn):
    def __init__(self, *args, **kwargs) -> None:
        """a, b, s are callables"""
        self.a, self.b, self.s, *_ = args
        self.va = np.vectorize(self.a)
        self.vb = np.vectorize(self.b)
        self.vs = np.vectorize(self.s)

    def update(self, state):
        self.da = self.va(state.x, state.y)
        self.db = self.vb(state.x, state.y)
        self.ds = self.vs(state.x, state.y)


########################################
# justice decision
########################################


class DecisionCost(object):
    @staticmethod
    def func_dual_cost(state, action):
        c_recid_no_treat = np.array([5, 10])
        c_recid_treat = np.array([1, 3])
        cost_no_treat = c_recid_no_treat @ action.recid_no_treat
        cost_treat = c_recid_treat @ action.recid_treat
        return cost_treat + cost_no_treat, cost_treat, cost_no_treat


class Decision(object):
    pass


class HierarchicalDecision(Decision):
    # partition the into
    def __init__(self, shape, theta=None, *args, **kwargs) -> None:
        self.shape = shape
        self.theta = np.zeros(shape) if theta is None else theta
        assert self.theta.shape[0] == shape
        self.func_classification = kwargs.get("func", None)

    # we assume top k% is the high risk category
    @staticmethod
    def func_top_k_category(
        state,
        perc=0.1,
        *args,
        **kwargs,
    ):
        """return index of top perc as high risk
        Args:
            x (_type_): _description_
            perc (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        x = state.x
        ind = np.argsort(x)
        q = np.zeros_like(x)
        q[ind[-int(x.shape[0] * perc) :]] = 1
        return np.vstack([1 - q, q])

    @staticmethod
    def func_random_category(
        state,
        perc=0.1,
        *args,
        **kwargs,
    ):
        """return a random fraction as high risk
        Args:
            x (_type_): _description_
            perc (float, optional): _description_. Defaults to 0.1.

        Returns:
            _type_: _description_
        """
        x = state.x
        ind = np.argsort(x)
        q = np.zeros_like(x)
        q[ind[np.random.choice(ind, size=int(ind.shape[0] * 0.1), replace=False)]] = 1
        return np.vstack([1 - q, q])


class RandomizedHierarchicalDecision(HierarchicalDecision):
    def forward(self, state, *args, **kwargs):
        """The forward step makes a decision y
        Args:
            state (_type_): _description_
        """
        if self.shape == 2:
            if self.func_classification == "random":
                self.c = (
                    HierarchicalDecision.func_random_category(
                        state,
                        *args,
                        **kwargs,
                    )
                    * state.z
                )
            elif self.func_classification == "single":
                # in this case you do not have any H-L classification,
                #   there is only one class here.
                n = state.x.shape[0]
                _c = np.zeros((2, n))
                _c[1, :] = 1
                self.c = _c * state.z
            else:
                self.c = (
                    HierarchicalDecision.func_top_k_category(
                        state,
                        *args,
                        **kwargs,
                    )
                    * state.z
                )

        else:
            raise NotImplementedError(f"d > 2 - dim decision is not implemented")
        x, y = self.c.nonzero()
        self.y = np.zeros_like(self.c)

        self.y[x, y] = self._y = np.array(
            [np.random.random() > self.theta[i] for i in x]
        )

        self.y = state.y = self.y.T
        self.recid_treat = self.y.T * self.c
        self.recid_no_treat = self.c - self.recid_treat

        self.eval(state, *args, **kwargs)

    def eval(self, state, *args, **kwargs):
        # cost evaluation
        func_cost = kwargs.get("func_cost")

        if func_cost is None:
            raise ValueError("provide a function to calculate cost")

        self.rp, self.rt, self.ri = func_cost(state, self)
        state.rt = self.rt.sum()
        state.ri = self.ri.sum()
        state.r = self.rt + self.ri

    def backward(self):
        """This is null backward step,
            we use a pure randomized policy
        Returns:
            _type_: _description_
        """
        pass


########################################
# state of a group
########################################
class GroupState(object):
    ALIAS = {
        "r": "treatment + incarceration cost",
        "ri": "incarceration cost",
        "rt": "treatment cost",
        "z": "recidivism number",
        "s": "screening cost",
        "f": "total cost",
    }

    # a state of a group is its current status at t
    def __init__(
        self,
        idx,  # group id
        t,  # time
        z=None,
        x=None,  # z, x, represent by n-d arrays
        P=None,  #
        y=None,
        c=None,
        a=None,
        s=None,
        r=None,
        ri=None,
        rt=None,
        f=None,
    ):
        self.idx = idx
        self.t = t
        self.z = z
        self.x = x
        self.P = P
        self.y = y
        self.c = c
        self.a = a
        self.f = f
        self.r = r
        self.ri = ri
        self.rt = rt
        self.s = s

    def evaluate(self, t):
        pass


########################################
# utilities
########################################
def fpr(theta, y_pred, y_true, target=0.1):
    c = y_pred > theta
    n = (1 - y_true).sum()
    cp = c[y_true < 1].sum()
    return cp / n - target


def find_theta(y_pred, y_true, target=0.1):
    args = (y_pred, y_true, target)
    # use a bisection procedure to find the right threshold θ
    l, u = 0, 1
    while (u - l) > 1e-2:
        t = (u + l) / 2
        ff = fpr(t, *args)
        if ff < -1e-2:
            u = t
        elif ff > 0:
            l = t
        else:
            break
    return t


########################################
# main
########################################
from enum import IntEnum


class LinearMixInType(IntEnum):
    WorsenCommunity = 0
    BenefiCommunity = 1
    BenefiSelf = 2


class FuncMixCommunity(object):
    @staticmethod
    def signexp(x, p=None):
        """
        p is a callable to produce the 0-1 indicator vector, 1 means active
        """
        if p is None:
            xs = x.sum()
        else:
            xs = x.T @ p()
        sgn = np.sign(xs)
        return sgn * (1 - np.exp(-np.abs(xs))) * 1e-1


class PopulationType(IntEnum):
    Decease = 0
    ExogenousBirth = 1


def run(
    T,
    P,
    state,
    actions=None,
    verbose=False,
    func_cost_function=None,
    options_mix=1,
    options_population=[],
):
    traj = []
    bool_population_decease = PopulationType.Decease in options_population
    P_start = copy.deepcopy(P)
    for t in range(0, T):
        # sample
        state.z = P.forward()

        # find policy based on z, x
        action: Decision = actions[t]

        action.forward(state, func_cost=func_cost_function)
        action.backward()

        # compute mix-in effect

        if verbose:
            if t == 0:
                print(
                    f"""
            {'='*100}
            actions: 
                    c: {action.c.shape}
                    y: {action.y.shape}
            states 
                    z: {state.z.shape}
                    x: {state.z.shape}
                """
                )
            print(
                f"""
            {'='*100}
            t: {t}
            z, y, z\y: {state.z.sum(), action.recid_treat.sum(), action.recid_no_treat.sum()},
            r, f: {state.r, state.r + state.z.sum() / 2}
            p: {P.p.min(), P.p.max()}
            σ: {P.sigma.min(), P.sigma.max()}
            {'='*100}
            """
            )

        # keep the trajectory at t
        traj.append(
            GroupState(
                state.idx,
                t,
                z=state.z.copy(),
                s=state.z.sum(),
                x=state.x.copy(),
                r=state.r.sum(),
                ri=state.ri.sum(),
                rt=state.rt.sum(),
                f=state.r.sum() + (state.z.sum() ** 2) / 2,
                a=action,
            )
        )

        # update record
        state.x += state.z

        # decease and birth
        n = state.x.shape[0]
        if bool_population_decease:
            gamma = 0.4
            idx_size = int(n * gamma)
            # random policy, choose gamma to decease
            idx_decease = np.random.choice(range(0, n), idx_size, replace=False)
            # reset these people to zero
            state.x[idx_decease] = 0
            P.p[idx_decease] = 0
            # random policy, choose gamma to birth
            idx_birth = np.random.choice(range(0, n), idx_size, replace=False)
            P.p[idx_decease] = P_start.p[idx_birth]
            P.sigma[idx_decease] = P_start.sigma[idx_birth]

        # mix-in effects
        gamma = 0.4
        idx_size = int(n * gamma)
        idx_change = np.random.choice(range(0, n), idx_size, replace=False)
        if options_mix == 0:
            # random policy, choose gamma to decease
            # reset these people to zero
            P.p[idx_change] = P_start.p[idx_change] + (
                (expit(action.recid_treat.sum() / n) - 0.5)
            )
        elif options_mix == 1:
            # update by initial "memoryless"
            P.sigma[idx_change] = P_start.sigma[idx_change].copy() + (
                action.recid_treat.sum() / n * 5
            )
            P.p[idx_change] = expit(P_start.b + P_start.a * P.sigma[idx_change])

        elif options_mix == 2:
            # update by initial "not memoryless"
            P.sigma[idx_change] = P.sigma[idx_change] + (
                action.recid_treat.sum() / n * 5
            )
            P.p[idx_change] = expit(P_start.b + P_start.a * P.sigma[idx_change])
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

    metric = [("s", None), ("f", None), ("ri", None), ("rt", None)]
    fig, axs = plt.subplots(2, len(metric) // 2)
    fig.set_size_inches(16, 8)
    for idx, ax in enumerate(axs.reshape(len(metric))):
        _x, _y = query_metric(metric[idx][0], reducer=metric[idx][1])
        ax.plot(_x, _y)
        ax.set_title(f"trajectory of {GroupState.ALIAS[metric[idx][0]]}")


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

        # ax[1].contour(1 - t1, 1 - t2, sum_z, levels=20, linewidths=0.2, colors="k")
        # cntr1 = ax[1].contourf(1 - t1, 1 - t2, sum_z, levels=20, cmap="RdBu_r")
        # ax[2].contour(1 - t1, 1 - t2, sum_f, levels=20, linewidths=0.2, colors="k")
        # cntr2 = ax[2].contourf(1 - t1, 1 - t2, sum_f, levels=20, cmap="RdBu_r")
        fig.colorbar(cntr0, ax=ax)
        # fig.colorbar(cntr1, ax=ax[1])
        # fig.colorbar(cntr2, ax=ax[2])
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"long-run {GroupState.ALIAS[metric[idx]]}")
        ax.set_xlabel("fraction of people treated in $L$")
        ax.set_ylabel("fraction of people treated in $H$")
    # ax[1].set(xlim=(0, 1), ylim=(0, 1))
    # ax[1].set_title("long-run recidivism number")
    # ax[1].set_xlabel("fraction of people treated in $L$")
    # ax[2].set_ylabel("fraction of people treated in $H$")
    # ax[2].set(xlim=(0, 1), ylim=(0, 1))
    # ax[2].set_title("long-run treatment, incarceration, screening cost")
    # ax[0].set_xlabel("fraction of people treated in $L$")
    # ax[0].set_ylabel("fraction of people treated in $H$")

    plt.show()
    fig.savefig(f"runid-{figure_id}.png", dpi=300, bbox_inches="tight", pad_inches=0)
    return df


def evaluate(traj):
    metrics = ["r", "z", "ri", "rt", "f", "s"]
    return {m: [getattr(r, m) for r in traj] for m in metrics}
