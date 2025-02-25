from .apga import APGA


class GABP(APGA):
    def __init__(
        self,
        strategy_space,
        num_actions,
        learning_rate,
        mutation_rate,
        update_slingshot_freq,
        **kwargs,
    ):
        super().__init__(
            strategy_space,
            num_actions,
            learning_rate,
            mutation_rate,
            update_slingshot_freq,
            **kwargs,
        )
        self.init_strategy = self.strategy.copy()

    def _update_slingshot_strategy_strategy(self):
        if (
            self.update_slingshot_freq is not None
            and self.update_slingshot_freq < self.t
        ):
            self.t = 0
            self.k += 1
            self.slingshot_strategy = (self.k * self.strategy + self.init_strategy) / (
                self.k + 1
            )
