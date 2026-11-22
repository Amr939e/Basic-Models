from Simulator import GBMSimulator
from Simulator import Option
from scipy.stats import norm
from Pricer import MonteCarloPricer
import numpy as np


def black_scholes(S0, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def garman_kohlhagen(S0, K, T, r_d,r_f, sigma, option_type="call"):
    r = r_d - r_f
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 *np.exp(-r_f*T)*norm.cdf(d1) - K * np.exp(-r_d* T) * norm.cdf(d2)
    else:
        return K * np.exp(-r_d * T) * norm.cdf(-d2) - S0 *np.exp(-r_f*T)* norm.cdf(-d1)


def test_convergence_to_bsm():
    S0, K, T, r, sigma = 154, 151, 3, 0.05, 0.2

    true_price = black_scholes(S0, K, T, r, sigma)


    sim = GBMSimulator(S0, r, sigma)
    opt = Option(K, T, "call", "european")
    pricer = MonteCarloPricer(sim, opt)

    price, sem = pricer.price(n_sims=200000, antithetic=True)
    lower_bound = price - 1.96 * sem
    upper_bound = price + 1.96 * sem

    assert lower_bound < true_price < upper_bound, f"True price {true_price} outside CI [{lower_bound}, {upper_bound}]"

def test_convergence_to_gk():
    S0, K, T, r_d, r_f, sigma = 154, 151, 3, 0.05,0.04, 0.2

    true_price = garman_kohlhagen(S0, K, T, r_d, r_f, sigma)
    drift = r_d - r_f
    sim = GBMSimulator(S0, drift, sigma)
    opt = Option(K, T, "call", "european")
    pricer = MonteCarloPricer(sim, opt)
    price, sem = pricer.price(n_sims=200000, antithetic=True, discount_rate = r_d)

    lower_bound = price - 1.96 * sem
    upper_bound = price + 1.96 * sem

    assert lower_bound < true_price < upper_bound, f"True price {true_price} outside CI [{lower_bound}, {upper_bound}]"
