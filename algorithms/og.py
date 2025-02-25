from .ga import GA, projection, simplex_projection


class OG(GA):
    def __init__(self, strategy_space, num_actions, learning_rate, **kwargs):
        super().__init__(strategy_space, num_actions, learning_rate, **kwargs)
        self.strategy_hat = self.strategy.copy()

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        gradient = self._calc_gradient()[1]
        if self.strategy_space == "simplex":
            self.strategy_hat = simplex_projection(
                self.strategy_hat + self.learning_rate * gradient
            )
            self.strategy = simplex_projection(
                self.strategy_hat + self.learning_rate * gradient
            )
        else:
            self.strategy_hat = projection(
                self.strategy_hat + self.learning_rate * gradient, self.limit
            )
            self.strategy = projection(
                self.strategy_hat + self.learning_rate * gradient, self.limit
            )
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient
