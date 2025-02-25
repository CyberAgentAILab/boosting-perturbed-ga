import numpy as np

from .base_game import BaseGame


class ZeroSumMatrixGame(BaseGame):
    def __init__(self, payoff, **kwargs):
        self.payoff = payoff
        self.noise_std = kwargs.get("noise_std", 0.1)
        self.strategy_space = "simplex"
        self.limit = kwargs.get("limit", None)

    def num_players(self):
        return 2

    def num_actions(self, player_id):
        return self.payoff.shape[player_id]

    def full_feedback(self, strategies):
        return [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]

    def noisy_feedback(self, strategies):
        feedback = [self.payoff @ strategies[1], -self.payoff.T @ strategies[0]]
        noise = [
            np.random.normal(0, self.noise_std, len(feedback[i]))
            for i in range(len(feedback))
        ]
        return [feedback[i] + noise[i] for i in range(len(feedback))]

    def nash_conv(self, strategies):
        return max(self.payoff @ strategies[1]) + max(-self.payoff.T @ strategies[0])

    def tangent_residual(self, strategies):
        return None


def random_payoff(n_actions=50):
    payoff = np.random.uniform(-1, 1, size=[n_actions, n_actions])
    return ZeroSumMatrixGame(payoff)
