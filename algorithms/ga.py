import numpy as np


def simplex_projection(y):
    u = sorted(y, reverse=True)
    cumsum_u = np.cumsum(u)
    ids = np.arange(len(u))[u + 1 / (np.arange(len(y)) + 1) * (1 - cumsum_u) > 0]
    rho = np.max(ids)
    lamb = 1 / (rho + 1) * (1 - cumsum_u[rho])
    x = np.maximum(y + lamb, 0)
    return x / sum(x)


def projection(strategy, limit):
    projected_strategy = strategy.copy()
    for k in range(len(strategy)):
        if strategy[k] > limit:
            projected_strategy[k] = limit
        if strategy[k] < -limit:
            projected_strategy[k] = -limit
    return projected_strategy


class GA(object):
    def __init__(self, strategy_space, num_actions, learning_rate, **kwargs):
        self.num_actions = num_actions
        self.cum_gradient = np.zeros(num_actions)
        self.gradient = np.zeros(num_actions)
        self.learning_rate = learning_rate
        if kwargs["random_init"]:
            k = np.random.exponential(scale=1.0, size=self.num_actions)
            self.strategy = k / k.sum()
        else:
            self.strategy = np.ones(num_actions) / num_actions
        self.limit = kwargs.get("limit", None)
        self.n = 0
        self.strategy_space = strategy_space

    def name(self):
        alg_name = self.__class__.__name__
        alg_name += "_lr{}".format(self.learning_rate)
        return alg_name

    def _ga(self, cum_gradient, gradient):
        if self.strategy_space == "simplex":
            self.strategy = simplex_projection(
                self.strategy + self.learning_rate * gradient
            )
        else:
            self.strategy = projection(
                self.strategy + self.learning_rate * gradient, self.limit
            )

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        self._ga(*self._calc_gradient())
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient
