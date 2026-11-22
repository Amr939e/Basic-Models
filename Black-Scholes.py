from Simulator import GBMSimulator
from Simulator import Option
from Pricer import MonteCarloPricer
from Greeks import calculate_delta, calculate_vega
from visualize import plot_convergence1
from test_bsm_validation import black_scholes
import numpy as np

S0, K, T, r, sigma = 154, 151, 3, 0.05, 0.2

sim = GBMSimulator(S0, r, sigma)
opt = Option(K, T, "call", "european")
pricer = MonteCarloPricer(sim, opt)

paths = sim.generate_paths(T=T, steps=252, n_sims=10, antithetic=True)
print(f"Paths shape: {paths.shape}")

final_prices = paths[-1, :]
print(f"Final prices (first 10): {final_prices[:10]}")
print(f"Mean final price: {np.mean(final_prices)}")
print(f"Expected mean: {S0 * np.exp(r * T)}")

payoffs = opt.payoff(paths)
print(f"Payoffs (first 10): {payoffs[:10]}")
print(f"Number of payoffs: {len(payoffs)}")
print(f"Mean payoff: {np.mean(payoffs)}")

discounted = np.exp(-r * T) * payoffs
price = np.mean(discounted)
print(f"Price: {price}")


if __name__ == "__main__":
    S0, K, T, r, sigma = 154, 151, 3, 0.05, 0.2

    sim = GBMSimulator(S0, r, sigma)
    opt = Option(K, T, "call", "european")
    pricer = MonteCarloPricer(sim, opt)
    price, error = pricer.price(n_sims=150000, antithetic=True)
    true_bsm = black_scholes(S0, K, T, r, sigma)

    print(f"Monte Carlo Price: ${price:.4f} ± ${error:.4f}")
    print(f"Black-Scholes Price: ${true_bsm:.4f}")
    print(f"Difference: {((price - true_bsm) / true_bsm) * 100:.2f}%")

    delta = calculate_delta(pricer)
    vega = calculate_vega(pricer,sigma)
    print(f"Delta: {delta:.4f} | Vega: {vega:.4f}")

    plot_convergence1(pricer, true_bsm)
