"""receive treatment if recidivate
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import copy
from tqdm.notebook import tqdm as tqdm_notebook

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


class LinearRecidProbWithBound(LinearRecidProb):
    scaler_a = lambda x: np.arctan(max(x, 0))

    def update(self, da=None, S=None):
        """
        we add a scaling func for a
        """
        if da is not None:
            self.a += da
        if S is not None:
            self.sigma = S * self.sigma
        self.P = expit(
            self.b + (LinearRecidProbWithBound.scaler_a(self.a) + 1) * self.sigma
        )


########################################
# mix-in effect
########################################
class MixIn(object):
    pass


class LinearMixIn(MixIn):
    def __init__(self, *args, **kwargs) -> None:
        """a, b, s are callables
        """
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
class Decision(object):
    pass


class HierarchicalDecision(Decision):
    # partition the into
    def __init__(self, shape, theta=None, *args, **kwargs) -> None:
        self.shape = shape
        self.theta = np.zeros(shape) if theta is None else theta
        assert self.theta.shape[0] == shape

    # we assume top k% is the high risk category
    @staticmethod
    def func_two_way_category(
        state, perc=0.1, *args, **kwargs,
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


class RandomizedHierarchicalDecision(HierarchicalDecision):
    def forward(self, state, *args, **kwargs):
        """The forward step makes a decision y
        Args:
            state (_type_): _description_
        """
        if self.shape == 2:
            self.c = (
                HierarchicalDecision.func_two_way_category(state, *args, **kwargs,)
                * state.z
            )
        # self._y = np.random.random([*state.x.shape, self.shape]) * self.c.T
        # self.y = state.y = np.apply_along_axis(
        #     lambda x: x > self.theta, 1, self._y
        # ).astype(np.int8)
        x, y = self.c.nonzero()
        self.y = np.zeros_like(self.c)

        self.y[x, y] = self._y = np.array(
            [np.random.random() > self.theta[i] for i in x]
        )

        self.y = state.y = self.y.T
        self.r = state.r = kwargs.get("func_cost", lambda x: x.sum())(state.y)

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
    # a state of a group is its current status at t
    def __init__(
        self,
        idx,  # group id
        t,  # time
        z=None,
        x=None,  # z, x, represent by n-d arrays
        P=None,  #
        y=None,
        r=None,
        f=None,
    ):
        self.idx = idx
        self.t = t
        self.z = z
        self.x = x
        self.P = P
        self.y = y
        self.r = r
        self.f = f

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


def run(
    T,
    P,
    state,
    actions=None,
    verbose=False,
    options=[LinearMixInType.WorsenCommunity, LinearMixInType.BenefiCommunity],
    # func_mix_self=lambda x: x,
    func_mix_community=None,
):
    traj = []
    bool_worse_community = LinearMixInType.WorsenCommunity in options
    bool_benef_community = LinearMixInType.BenefiCommunity in options
    bool_benef_self = LinearMixInType.BenefiSelf in options
    for t in range(0, T):
        # sample
        state.z = P.forward()

        # find policy based on z, x
        action = actions[t]

        action.forward(state, func_cost=lambda x: x.sum())
        action.backward()

        # compute mix-in effect
        q1 = np.array([0.001, 0.005])
        s = np.array([0.010, 0.005]) * bool_benef_self

        recid_no_treat = (1 - action.y).T * action.c
        dx = (
            q1 @ (recid_no_treat) * bool_worse_community
            - q1 @ (action.y.T * action.c) * bool_benef_community / 5
        )

        fmc = func_mix_community
        da = fmc(dx)
        S = np.ones_like(dx) - action.y @ s + s @ recid_no_treat / 5

        if verbose:
            print(
                f"""
            {'='*100}
            t: {t}
            ft: {bool_worse_community, bool_benef_community, bool_benef_self}
            q1: {q1}
            z, y, z\y: {state.z.sum(), action.y.sum(), recid_no_treat.sum()},
            dx: {dx.max()}
            da, (1-S): {da, (1-S).sum()}
            S: {S.min(), S.max()}
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
                x=state.x.copy(),
                r=state.r,
                f=state.r + state.z.sum() / 2,
            )
        )

        # apply effect
        P.update(da, S)

        # update record
        state.x += state.z

    return traj


def grid_search(s0, P0, T, d, space, options, func_mix_community):
    z = []
    perfs = []
    for s1 in tqdm_notebook(space):
        for s2 in space:
            actions = [
                RandomizedHierarchicalDecision(d, theta=np.array([s1, s2]))
                for _ in range(T)
            ]
            P = copy.deepcopy(P0)
            state = copy.deepcopy(s0)
            traj = run(
                T,
                P,
                state,
                actions,
                options=options,
                func_mix_community=func_mix_community,
            )
            perf = evaluate(traj)
            z.append(
                {"r": sum(perf["r"]), "z": sum(perf["z"]).sum(), "f": sum(perf["f"])}
            )
            perfs.append(perf)
    return z, perfs


def show(grids, z, perfs, idx):
    t1, t2 = grids
    df = pd.DataFrame.from_records(z)
    # plot cost and total number of recidivism
    sum_c = df["r"].values.reshape(t1.shape)
    sum_z = df["z"].values.reshape(t1.shape)
    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(16, 5)
    plt.rcParams.update({"font.size": 12})
    ax[0].contour(1 - t1, 1 - t2, sum_c, levels=14, linewidths=0.2, colors="k")
    cntr0 = ax[0].contourf(1 - t1, 1 - t2, sum_c, levels=20, cmap="RdBu_r")
    ax[1].contour(1 - t1, 1 - t2, sum_z, levels=14, linewidths=0.2, colors="k")
    cntr1 = ax[1].contourf(1 - t1, 1 - t2, sum_z, levels=20, cmap="RdBu_r")
    ax[2].contour(
        1 - t1, 1 - t2, sum_c + sum_z / 2, levels=14, linewidths=0.2, colors="k"
    )
    cntr2 = ax[2].contourf(1 - t1, 1 - t2, sum_c + sum_z / 2, levels=20, cmap="RdBu_r")
    fig.colorbar(cntr0, ax=ax[0])
    fig.colorbar(cntr1, ax=ax[1])
    fig.colorbar(cntr2, ax=ax[2])
    ax[0].set(xlim=(0, 1), ylim=(0, 1))
    ax[0].set_title("long-run treatment cost of the justice system")
    ax[0].set_xlabel("$1-\\theta_1$: fraction of people treated in $L$ group")
    ax[0].set_ylabel("$1-\\theta_2$: fraction of people treated in $H$ group")
    ax[1].set(xlim=(0, 1), ylim=(0, 1))
    ax[1].set_title("long-run recidivism number of the justice system")
    ax[1].set_xlabel("$1-\\theta_1$: fraction of people treated in $L$ group")
    ax[1].set_ylabel("$1-\\theta_2$: fraction of people treated in $H$ group")
    ax[2].set(xlim=(0, 1), ylim=(0, 1))
    ax[2].set_title("long-run total recidivism cost of the justice system")
    ax[2].set_xlabel("$1-\\theta_1$: fraction of people treated in $L$ group")
    ax[2].set_ylabel("$1-\\theta_2$: fraction of people treated in $H$ group")

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig(f"runid-{idx}.png", dpi=300)


def evaluate(traj):
    metrics = ["r", "z", "f"]
    return {m: [getattr(r, m) for r in traj] for m in metrics}

