import numpy as np
import math


# Monte Carlo method approximation for k order statistics on Normal distribution
# with mean mu and std. deviation sigma for sample of size n
def approx_k_order_stat(mu, sigma, k, n, repeats=200):
    ks = []
    for r in range(0, repeats):
        items = np.random.normal(mu, sigma, n)
        items.sort()
        ks.append(items[k-1])

    return np.average(ks)
