import copy

import numpy as np
import scipy.sparse

from optimization import *
from scipy.special import expit, logit


########################################
# recidivism probability
########################################
class RecidProb(object):
    def forward(self):
        pass

    def reverse(self):
        """
        Compute the reverse step, i.e., the parameters.
        """
        pass

    def update(self, da=None, S=None):
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
        self.p = expit(self.b + self.a * self.sigma)
        self.Q = np.zeros_like(self.p)
        self.Q[0] = 0.1

    def reverse(self):
        """
        Reverses the calculation of `sigma` from the original value of `p`.

        Returns:
            float: The original value of `sigma`.
        """
        self.sigma = (logit(self.p) - self.b) / self.a

    def forward(self):
        """recidivate

        Returns:
            _type_: _description_
        """
        return self.p


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
    def func_linear_cost(state):
        c_recid_no_treat = np.array([5, 10])
        c_recid_treat = np.array([1, 3])
        cost_no_treat = c_recid_no_treat @ state.recid_no_treat
        cost_treat = c_recid_treat @ state.recid_treat
        return cost_treat, cost_no_treat


class Decision(object):
    pass


# class HierarchicalDecision(Decision):
#     # partition the into
#     def __init__(self, shape, theta=None, *args, **kwargs) -> None:
#         self.shape = shape
#         self.theta = np.zeros(shape) if theta is None else theta
#         assert self.theta.shape[0] == shape
#         self.func_classification = kwargs.get("func", None)

#     # we assume top k% is the high risk category
#     @staticmethod
#     def func_top_k_category(
#         state,
#         perc=0.1,
#         *args,
#         **kwargs,
#     ):
#         """return index of top perc as high risk
#         Args:
#             x (_type_): _description_
#             perc (float, optional): _description_. Defaults to 0.1.

#         Returns:
#             _type_: _description_
#         """
#         x = state.x
#         ind = np.argsort(x)
#         q = np.zeros_like(x)
#         q[ind[-int(x.shape[0] * perc) :]] = 1
#         return np.vstack([1 - q, q])

#     @staticmethod
#     def func_random_category(
#         state,
#         perc=0.1,
#         *args,
#         **kwargs,
#     ):
#         """return a random fraction as high risk
#         Args:
#             x (_type_): _description_
#             perc (float, optional): _description_. Defaults to 0.1.

#         Returns:
#             _type_: _description_
#         """
#         x = state.x
#         ind = np.argsort(x)
#         q = np.zeros_like(x)
#         q[ind[np.random.choice(ind, size=int(ind.shape[0] * 0.1), replace=False)]] = 1
#         return np.vstack([1 - q, q])


# class RandomizedHierarchicalDecision(HierarchicalDecision):
#     def forward(self, state, *args, **kwargs):
#         """The forward step makes a decision y
#         Args:
#             state (_type_): _description_
#         """
#         if self.shape == 2:
#             if self.func_classification == "random":
#                 self.c = (
#                     HierarchicalDecision.func_random_category(
#                         state,
#                         *args,
#                         **kwargs,
#                     )
#                     * state.z
#                 )
#             else:
#                 self.c = (
#                     HierarchicalDecision.func_top_k_category(
#                         state,
#                         *args,
#                         **kwargs,
#                     )
#                     * state.z
#                 )
#         else:
#             raise NotImplementedError(f"d>2 - dim decision is not implemented")
#         x, y = self.c.nonzero()
#         self.y = np.zeros_like(self.c)

#         self.y[x, y] = self._y = np.array(
#             [np.random.random() > self.theta[i] for i in x]
#         )

#         self.y = state.y = self.y.T
#         self.recid_treat = self.y.T * self.c
#         self.recid_no_treat = self.c - self.recid_treat

#     def backward(self):
#         """This is null backward step,
#             we use a pure randomized policy
#         Returns:
#             _type_: _description_
#         """
#         pass


class RandomizedDecision(Decision):
    # partition the into
    def __init__(self, shape, theta=None, *args, **kwargs) -> None:
        self.shape = shape
        self.theta = np.zeros(shape) if theta is None else theta
        assert self.theta.shape[0] == shape

    def forward(self, state, *args, **kwargs):
        """The forward step makes a decision y
        Args:
            state (_type_): _description_
        """
        if self.shape == 2:
            # in the same category (first)
            _x = state.x
            self.c = np.vstack([np.ones_like(_x), np.zeros_like(_x)]) * state.p.p
        else:
            raise NotImplementedError(f"d>2 - dim decision is not implemented")

        self.y = np.repeat(np.expand_dims(self.theta, 0), state.n, 0)
        self.recid_treat = self.y.T * self.c
        self.recid_no_treat = self.c - self.recid_treat

    def backward(self):
        """This is null backward step,
            we use a pure randomized policy
        Returns:
            _type_: _description_
        """
        pass


class RandomizedSimpleHierarchicalDecision(Decision):
    # partition the into
    def __init__(self, shape, theta=None, *args, **kwargs) -> None:
        self.shape = shape
        self.theta = np.zeros(shape) if theta is None else theta
        assert self.theta.shape[0] == shape

    def forward(self, state, *args, **kwargs):
        """The forward step makes a decision y
        Args:
            state (_type_): _description_
        """
        if self.shape == 2:
            _x = state.x
            n = _x.shape[0]
            self.c = np.zeros((2, n))
            self.c[0, 0 : n // 2] = 1
            self.c[1, n // 2 :] = 1
            self.c = self.c * state.p.p
        else:
            raise NotImplementedError(f"d>2 - dim decision is not implemented")

        self.y = np.repeat(np.expand_dims(self.theta, 0), state.n, 0)
        self.recid_treat = self.y.T * self.c
        self.recid_no_treat = self.c - self.recid_treat

    def backward(self):
        """This is null backward step,
            we use a pure randomized policy
        Returns:
            _type_: _description_
        """
        pass


########################################
# state of the system
########################################
class RecidState(object):
    ALIAS = {
        "r": "treatment + incarceration cost",
        "ri": "incarceration cost",
        "rt": "treatment cost",
        "s": "recidivism number",
        "f": "total cost",
        "total_x": "total population",
        "active_x": "active population",
    }

    # a state of a group is its current status at t
    def __init__(
        self,
        t,  # time
        x=None,  # current state probability
        p: RecidProb = None,  # current recidivate probability ρ
        a: Decision = None,  # current action [decision of the justice system]
        s=None,
        r=None,
        ri=None,
        rt=None,
        f=None,
    ):
        self.t = t
        self.x = x
        self.p = p
        self.a = a
        self.f = f
        self.r = r
        self.ri = ri
        self.rt = rt
        self.s = s
        self.n = self.x.shape[0]
        self.Phi = None
        self.rows = [*range(self.n), *range(self.n - 1)]
        self.cols = [*range(self.n), *range(1, self.n)]

        # the decision per state, multiplied by x
        #   unscaled decision see action <- state.a
        self.recid_no_treat = None
        self.recid_treat = None

    def compute_transition(
        self, rate_state_decay=1.0, rate_state_decay_last=0.5  # no decay by default
    ):
        # transition matrix formed by recidivate probability
        self.phi_data = rate_state_decay * np.hstack(
            ((1 - self.p.p), self.p.p[0 : self.n - 1])
        )
        # last entry not scale by probability
        #   for the last state it only transfers to itself (absorbing)
        # self.phi_data[self.n - 1] = rate_state_decay_last
        # create coo matrix
        self.Phi = scipy.sparse.coo_array(
            (self.phi_data, (self.rows, self.cols)),
        )

    def forward(
        self,
        rate_exogenous_arrival=0.4,
        rate_state_decay=1.0,
        rate_state_decay_last=0.5,
    ):
        self.compute_transition(
            rate_state_decay=rate_state_decay,
            rate_state_decay_last=rate_state_decay_last,
        )
        self.total_x = self.x.sum()
        self.active_x = self.x[:-1].sum()
        self.lbda = lbda = np.zeros_like(self.x)
        lbda[0] = rate_exogenous_arrival
        # make a copy to stage t
        sto = copy.deepcopy(self)
        # apply transition to t + 1
        self.x = self.Phi.T @ self.x + lbda

        if not np.all(self.p.p <= 1):
            print(self.p.p)
            print(self.Phi)
            print(self.x)
            raise ValueError("probability too large")
        return sto

    def evaluate(self, action: Decision, **kwargs):
        func_cost = kwargs.get("func_cost")

        self.a = action
        self.recid_no_treat = self.x * action.recid_no_treat
        self.recid_treat = self.x * action.recid_treat
        self.s = self.x @ self.p.p
        if func_cost is None:
            self.rt = self.recid_treat.sum()
            self.ri = self.recid_no_treat.sum()
            print("!!! did not provide a function to calculate cost, set to unit cost")
        else:
            self.rt, self.ri = func_cost(self)
            self.rt = self.rt.sum()
            self.ri = self.ri.sum()
        self.r = self.rt + self.ri


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
