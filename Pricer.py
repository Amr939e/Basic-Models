import numpy as np

class MonteCarloPricer:
    def __init__(self, simulator, option):
        self.simulator = simulator
        self.option = option

    def price(self, steps=252, n_sims=100000, antithetic=True, discount_rate = None):
        paths = self.simulator.generate_paths(
            T=self.option.T,
            steps=steps,
            n_sims=n_sims,
            antithetic=antithetic
        )
        if discount_rate is None:
            discount_rate = self.simulator.r
      
        payoffs = self.option.payoff(paths)

        T = self.option.T
        price = np.exp(-discount_rate * T) * np.mean(payoffs)
        sem = 0
        if antithetic:

            n_pairs = len(payoffs) // 2
            payoffs_plus = payoffs[:n_pairs]  
            payoffs_minus = payoffs[n_pairs:]  
            pair_averages = (payoffs_plus + payoffs_minus) / 2

            price = np.exp(-discount_rate * T) * np.mean(pair_averages)
            sem = np.exp(-discount_rate * T)* np.std(pair_averages, ddof=1) / np.sqrt(n_pairs)
        else:
            std_dev_payoffs = np.std(payoffs, ddof=1)
            sem = std_dev_payoffs / np.sqrt(len(payoffs))
        return price, sem
#The standard error is sigma_payoffs/sqrt(N)
    def price_wc(self, steps=252, n_sims=50000):
        paths = self.simulator.generate_paths(...)
        payoffs = self.option.payoff(paths)
        terminal_prices = paths[-1, :] 

        expected_ST = self.simulator.S0 * np.exp(self.simulator.r * self.option.T)

        covariance = np.cov(payoffs, terminal_prices)[0, 1]
        variance = np.var(terminal_prices)
        beta = covariance / variance

        adjusted_payoffs = payoffs - beta * (terminal_prices - expected_ST)
        price = np.exp(-self.simulator.r * self.option.T) * np.mean(adjusted_payoffs)
        return price
