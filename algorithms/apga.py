from .ga import GA


class APGA(GA):
    def __init__(
        self,
        strategy_space,
        num_actions,
        learning_rate,
        mutation_rate,
        update_slingshot_freq,
        **kwargs,
    ):
        super().__init__(strategy_space, num_actions, learning_rate, **kwargs)
        self.mutation_rate = mutation_rate
        self.update_slingshot_freq = update_slingshot_freq
        self.slingshot_strategy = self.strategy.copy()
        self.k = 0
        self.t = 0

    def calc_strategy(self):
        self.t += 1
        return super().calc_strategy()

    def name(self):
        alg_name = self.__class__.__name__
        if self.update_slingshot_freq is not None:
            alg_name += "_utf{}".format(self.update_slingshot_freq)
        alg_name += "_mu{}".format(self.mutation_rate)
        alg_name += "_lr{}".format(self.learning_rate)
        return alg_name

    def add_gradient(self, gradient):
        mutation = -self.mutation_rate * (self.strategy - self.slingshot_strategy)
        self.cum_gradient += gradient + mutation
        self.gradient = gradient + mutation
        self._update_slingshot_strategy_strategy()

    def _update_slingshot_strategy_strategy(self):
        if (
            self.update_slingshot_freq is not None
            and self.update_slingshot_freq < self.t
        ):
            self.t = 0
            self.k += 1
            self.slingshot_strategy = self.strategy.copy()
