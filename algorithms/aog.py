from .ga import GA, projection, simplex_projection


class AOG(GA):
    def __init__(self, strategy_space, num_actions, learning_rate, **kwargs):
        super().__init__(strategy_space, num_actions, learning_rate, **kwargs)
        self.strategy_hat = self.strategy.copy()
        self.init_strategy = self.strategy.copy()

    def _calc_gradient(self):
        return self.cum_gradient, self.gradient

    def calc_strategy(self):
        gradient = self._calc_gradient()[1]
        if self.strategy_space == "simplex":
            self.strategy_hat = simplex_projection(
                self.strategy_hat
                + self.learning_rate * gradient
                - (self.strategy_hat - self.init_strategy) / (self.n + 1)
            )
            self.strategy = simplex_projection(
                self.strategy_hat
                + self.learning_rate * gradient
                - (self.strategy_hat - self.init_strategy) / (self.n + 2)
            )
        else:
            self.strategy_hat = projection(
                self.strategy_hat
                + self.learning_rate * gradient
                - (self.strategy_hat - self.init_strategy) / (self.n + 1),
                self.limit,
            )
            self.strategy = projection(
                self.strategy_hat
                + self.learning_rate * gradient
                - (self.strategy_hat - self.init_strategy) / (self.n + 2),
                self.limit,
            )
        self.n += 1
        return self.strategy

    def add_gradient(self, gradient):
        self.cum_gradient += gradient
        self.gradient = gradient
