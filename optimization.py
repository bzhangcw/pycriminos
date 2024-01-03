import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# this two functions are evaluations of loss and fairness
classification_loss = np.vectorize(lambda x, y, theta: float((x > theta) != y))
metric_objective = "classification"


def evaluation(group, theta):
    if theta in group.feval:
        return group.feval[theta]
    loss = classification_loss(group.df["x"], group.df["y"], theta)
    group.loss_arr = loss
    group.loss = loss.mean()
    group.feval[theta] = group.loss
    return group.loss


def fairness_simple(group, theta):
    group.gamma = theta
    return group.gamma


def fairness_equal_opt(group, theta):
    if theta in group.geval:
        return group.geval[theta]
    gamma_arr = group.df.query("y == 0")["x"] > theta
    group.gamma_arr = gamma_arr
    group.gamma = gamma_arr.mean()
    group.geval[theta] = group.gamma
    return group.gamma


def eval_obj_gamma(func, groups, *args, **kwargs):
    iterates_gamma = []
    iterates_theta = []
    iterates_loss = []
    _scale = [*[[*g.I0, *g.I1] for name, g in sorted(groups.items())]]
    _l, _u = np.min(_scale), np.max(_scale)
    _interval = np.arange(_l, _u, 0.1)
    for theta in _interval:
        iterates_theta.append([theta for name, g in sorted(groups.items())])
        iterates_gamma.append([func(g, theta) for name, g in sorted(groups.items())])
        iterates_loss.append(
            [evaluation(g, theta) for name, g in sorted(groups.items())]
        )
    iterates_theta = np.array(iterates_theta)
    iterates_gamma = np.array(iterates_gamma)
    iterates_loss = np.array(iterates_loss)
    iterates_loss_total = iterates_loss.sum(axis=1)
    iterates = np.column_stack([iterates_loss, iterates_loss_total])
    # visualize
    fig1, ax1 = plt.subplots(2, 1)
    for gid, (name, g) in enumerate(sorted(groups.items())):
        ax1[0].plot(iterates_theta[:, 0], iterates[:, gid], label=name)
        ax1[1].plot(iterates_theta[:, 0], iterates_gamma[:, gid], label=name)

    ax1[0].plot(iterates_theta[:, 0], iterates[:, -1], label="total")
    ax1[0].legend()
    ax1[1].legend()
    fig1.suptitle(
        r"Minimization of Metric \texttt{"
        + f"{metric_objective}"
        + r"}"
        + r"  w.t. Fairness := \texttt{"
        + f"{func.__name__}"
        + r"}"
    )


def optimize_simple(groups, tol_theta=0.05, verbose=False, *args, **kwargs):
    # search interval
    _scale = [*[[*g.I0, *g.I1] for name, g in sorted(groups.items())]]
    _l, _u = np.min(_scale), np.max(_scale)
    _interval = np.arange(_l, _u, tol_theta)
    loss = 1e6
    for theta in _interval:
        _loss = [evaluation(g, theta) for name, g in sorted(groups.items())]
        _loss_sum = sum(_loss)
        if _loss_sum < loss:
            loss = _loss_sum
            iterates_loss = np.array(_loss)
            iterates_theta = np.array([theta for name, g in sorted(groups.items())])
            iterates_gamma = np.array([theta for name, g in sorted(groups.items())])

    if verbose:
        print(
            f"""
            optimality found 
            solutions  (θ) :  {iterates_theta}
            objectives (L) :  {iterates_loss}
            gamma      (Γ) :  {iterates_gamma}

        """
        )
    return (
        iterates_loss,  # optimal loss
        iterates_theta,  # θ: iterates in size k * |G|
        iterates_gamma,  # Γ: gamma function in size k * |G|
    )


def optimize_eqopt(
    groups,
    target="a",
    verbose=False,
    tol_theta=0.05,
    tol_gamma=0.005,
    tol_digits=3,
    *args,
    **kwargs,
):
    # now ta may not = tb
    # set search interval of ta

    iterates_theta = []
    iterates_gamma = []
    iterates_loss = []
    _scale = [*[[*g.I0, *g.I1] for name, g in sorted(groups.items())]]
    _l, _u = np.min(_scale), np.max(_scale)
    _interval = np.arange(_l, _u, tol_theta)
    loss = 1e6
    for theta in _interval:
        ga = fairness_equal_opt(groups[target], theta)
        la = evaluation(groups[target], theta)
        _theta = [theta]
        _gamma = [ga]
        _loss = [la]
        for name, g in sorted(groups.items()):
            if name == target:
                continue
            # use bisection to find proper gamma
            lb = _l
            ub = _u
            while (ub - lb) > tol_theta:
                tt = round((lb + ub) / 2, tol_digits)
                gg = fairness_equal_opt(g, tt)
                if gg > ga + tol_gamma:
                    lb = tt
                elif gg < ga - tol_gamma:
                    ub = tt
                else:
                    break
            _theta.append(tt)
            _gamma.append(gg)
            _loss.append(evaluation(g, tt))
        _loss_sum = sum(_loss)
        if _loss_sum < loss:
            loss = _loss_sum
            iterates_loss = np.array(_loss)
            iterates_theta = np.array(_theta)
            iterates_gamma = np.array(_gamma)

    if verbose:
        print(
            f"""
            optimality found 
            solutions  (θ) :  {iterates_theta}
            objectives (L) :  {iterates_loss}
            gamma      (Γ) :  {iterates_gamma}

        """
        )
    return (
        iterates_loss,  # optimal loss
        iterates_theta,  # θ: iterates in size k * |G|
        iterates_gamma,  # Γ: gamma function in size k * |G|
    )
