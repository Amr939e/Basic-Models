import matplotlib.pyplot as plt

def plot_convergence1(pricer, true_price):
    sim_counts = [1000, 5000, 10000, 25000, 50000, 100000, 250000]
    prices = []
    errors = []

    for n in sim_counts:
        p, e = pricer.price(n_sims=n, antithetic=True)
        prices.append(p)
        errors.append(1.96 * e)  # 95% CI

    plt.figure(figsize=(10, 6))
    plt.errorbar(sim_counts, prices, yerr=errors, fmt='o-', capsize=5, label='Simulated Price')
    plt.axhline(y=true_price, color='r', linestyle='--', label='Black Scholes True Price')
    plt.xscale('log')
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Convergence (BS) with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_plot1.png', dpi=300)
    plt.show()

def plot_convergence2(pricer, true_price, r_d = None):
    sim_counts = [1000, 5000, 10000, 25000, 50000, 100000, 250000]
    prices = []
    errors = []

    for n in sim_counts:
        p, e = pricer.price(n_sims=n, antithetic=True, discount_rate = r_d)
        prices.append(p)
        errors.append(1.96 * e)  

    plt.figure(figsize=(10, 6))
    plt.errorbar(sim_counts, prices, yerr=errors, fmt='o-', capsize=5, label='Simulated Price')
    plt.axhline(y=true_price, color='r', linestyle='--', label='Garmen Kohlhagen True Price')
    plt.xscale('log')
    plt.xlabel('Number of Simulations (log scale)')
    plt.ylabel('Option Price')
    plt.title('Monte Carlo Convergence (GK) with 95% Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('convergence_plot2.png', dpi=300)
    plt.show()

