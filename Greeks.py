def calculate_delta(pricer, bp=0.01):

    original_S0 = pricer.simulator.S0

    pricer.simulator.S0 = original_S0 * (1 + bp)
    price_up, _ = pricer.price()  

    pricer.simulator.S0 = original_S0 * (1 - bp)
    price_down, _ = pricer.price()  

    pricer.simulator.S0 = original_S0

    delta = (price_up - price_down) / (2 * original_S0 * bp)
    return delta
#total change is S0*(1+bump) - S0*(1-bump) = 2*SO*bump
#Vega (bump sigma)
def calculate_vega(pricer, sigma):
    original_S0 = pricer.simulator.S0

    pricer.simulator.S0 = original_S0 * (1 + sigma)
    price_up, _ = pricer.price() 


    pricer.simulator.S0 = original_S0 * (1 - sigma)
    price_down, _ = pricer.price() 

    pricer.simulator.S0 = original_S0

    vega = (price_up - price_down) / (2 * original_S0 * sigma)
    return vega
