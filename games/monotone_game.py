import numpy as np

from .base_game import BaseGame


class MonotoneGame(BaseGame):
    def __init__(
        self,
        n_actions,
        strategy_space,
        gradient_fn,
        nash_conv_fn,
        tangent_residual_fn=None,
        **kwargs,
    ):
        self.n_actions = n_actions
        self.strategy_space = strategy_space
        self.gradient_fn = gradient_fn
        self.nash_conv_fn = nash_conv_fn
        self.tangent_residual_fn = tangent_residual_fn
        self.noise_std = kwargs.get("noise_std", 0.1)
        self.limit = kwargs.get("limit", None)

    def num_players(self):
        return 2

    def num_actions(self, player_id):
        return self.n_actions[player_id]

    def full_feedback(self, strategies):
        return self.gradient_fn(strategies)

    def nash_conv(self, strategies):
        return self.nash_conv_fn(strategies)

    def tangent_residual(self, strategies):
        if self.tangent_residual_fn is None:
            return None
        else:
            return self.tangent_residual_fn(strategies)

    def noisy_feedback(self, strategies):
        feedback = self.full_feedback(strategies)
        noise = [
            np.random.normal(0, self.noise_std, len(strategies[i]))
            for i in range(len(strategies))
        ]
        return [feedback[i] + noise[i] for i in range(len(feedback))]

    def strategy_classes(self):
        return self._strategy_classes


def hard_concave_convex(num_actions=100, limit=200, payoff_mode="fix", **kwargs):
    limit = limit
    # the set is [-limit, limit]^n
    A = np.zeros((num_actions, num_actions))
    b = np.ones(num_actions) / 4
    h = np.zeros(num_actions)
    h[num_actions - 1] = 1 / 4
    for i in range(num_actions):
        for j in range(num_actions):
            value = 1
            if i + j == num_actions - 1:
                A[i][j] = value
                if j >= 1:
                    A[i][j - 1] = -value
    A = 1 / 4 * A
    H = 2 * np.dot(A.T, A)
    kwargs["limit"] = limit

    def gradient_fn(strategies):
        assert len(strategies) == 2
        x, y = strategies
        # print("max: ", np.max(-(np.dot(H, x) - h - np.dot(A, y))), np.max(-np.dot(A.T, x) + b))
        # print("min: ", np.min(-(np.dot(H, x) - h - np.dot(A, y))), np.min(-np.dot(A.T, x) + b))
        return -(np.dot(H, x) - h - np.dot(A, y)), -np.dot(A.T, x) + b

    def nash_conv_fn(strategies):
        return __dynamic_reg(strategies)

    def tangent_residual(strategies):
        fx, fy = gradient_fn(strategies)
        return np.sqrt(np.linalg.norm(fx, ord=2) ** 2 + np.linalg.norm(fy, ord=2) ** 2)

    def __dynamic_reg(strategies):
        x, y = strategies
        c = np.dot(A, x) - b
        sum = 0
        for i in range(num_actions):
            if c[i] <= 0:
                sum = sum - c[i] * (limit - y[i])
            else:
                sum = sum - c[i] * (-limit - y[i])
        return sum

    return MonotoneGame(
        [num_actions, num_actions],
        "limit",
        gradient_fn,
        nash_conv_fn,
        tangent_residual,
        **kwargs,
    )
