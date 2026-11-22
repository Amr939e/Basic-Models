import numpy as np
class GBMSimulator:
    def __init__(self, S0: float, r: float, sigma: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.q = q

    def generate_paths(self, T, steps, n_sims, antithetic= False):
        dt = T / steps
        drift = (self.r - self.q - 0.5 * self.sigma ** 2) * dt
        Z = np.random.standard_normal((steps, n_sims))
      
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=1)
            n_sims = n_sims * 2  
          
        log_returns = drift + self.sigma * np.sqrt(dt) * Z

        log_price_paths = np.zeros((steps + 1, n_sims))
        log_price_paths[0, :] = np.log(self.S0) 

        for i in range(1, steps + 1):
            log_price_paths[i, :] = log_price_paths[i - 1, :] + log_returns[i - 1, :]

        log_price_paths = np.insert(log_price_paths, 0, np.log(self.S0), axis=0)
        return np.exp(log_price_paths)
      
class Option:
    def __init__(self, strike: float, maturity: float, option_type: str = "call", payoff_style: str = "european",
                 barrier: float = None):
        self.strike = strike
        self.T = maturity
        self.option_type = option_type  # 'call' or 'put'
        self.payoff_style = payoff_style  # 'european', 'asian', 'barrier_up_out'
        self.barrier = barrier

    def payoff(self, price_paths: np.ndarray) -> np.ndarray:
        # (time_steps+1, n_sims)
        if self.payoff_style == "european":
            S_T = price_paths[-1, :]

        elif self.payoff_style == "asian":
            S_T = np.mean(price_paths[1:, :], axis=0)

        elif self.payoff_style == "barrier_up_out":
            max_prices = np.max(price_paths[1:, :], axis=0)
            S_T = price_paths[-1, :]
            S_T = np.where(max_prices > self.barrier, 0, S_T)

        if self.option_type == "call":
            return np.maximum(S_T - self.strike, 0)
        else: # Put
            return np.maximum(self.strike - S_T, 0)
