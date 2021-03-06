import numpy as np
import math

# Compute time to transmit for a message in bytes and a given network speed
def computeTTX(msgSize, netSpeed, netSpeedStdDev):
    v = netSpeed / netSpeedStdDev
    ttxsec = msgSize * 8 / netSpeed
    ttx_std_dev = ttxsec / v
    ttxms = ttxsec * 1000
    ttx_std_dev_ms = ttx_std_dev * 1000
    return ttxms, ttx_std_dev_ms

# Monte Carlo method approximation for k order statistics on Normal distribution
# with mean mu and std. deviation sigma for sample of size n
def approx_k_order_stat(mu, sigma, k, n, repeats=200):
    ks = []
    for r in range(0, repeats):
        items = np.random.normal(mu, sigma, n)
        items.sort()
        ks.append(items[k-1])

    return np.average(ks)

def approx_k_order_stat_paxos_wan(mu, sigma, mu_ms_s, sigma_ms_s, mu_md_s, sigma_md_s, ttx_s, ttx_stdev_s, k, n, leader, repeats=200):
    ks = []
    for r in range(0, repeats):
        items = []
        for i in range(0, n):
            if i != leader:
                rtt_with_processing = mu[leader][i] + mu_ms_s + mu_md_s + 2*ttx_s
                sigma_with_processing = math.sqrt(sigma[leader][i]**2 + sigma_ms_s**2 + sigma_md_s**2 + (2*ttx_stdev_s)**2)
                items.append(np.random.normal(rtt_with_processing, sigma_with_processing))

        items.sort()
        ks.append(items[k-1])

    return np.average(ks)

def approx_k_order_stat_wpaxos_zone(mu, sigma,qw, k, n, repeats=200, exclude=-1):
    ks = []
    for r in range(0, repeats):
        items = []
        for i in range(0, n):
            if i != exclude:
                items.append(np.random.normal(mu, sigma) + qw[i])
        items.sort()
        ks.append(items[k-1])

    return np.average(ks)

def marchal_mean_queue_wait_time(num_d, num_s, mu_r_s, sigma_r_s, mu_ms_s, mu_md_s, ttx_s, ttx_stddev_s, n_p, sigma_ms_s, sigma_md_s):
    R = (1 / mu_r_s)
    C_a = (sigma_r_s ** 2) / (mu_r_s ** 2)
    lmda = R  # mean rate of arrival in rounds per second

    # mean rate of service (speed of the pipeline). essentially max throughput:
    queue_time_per_round = num_d * (mu_md_s + ttx_s) + num_s * (mu_ms_s + ttx_s)

    sigma_send = math.sqrt(sigma_ms_s ** 2 + ttx_stddev_s ** 2)
    sigma_recv = math.sqrt(sigma_md_s ** 2 + ttx_stddev_s ** 2)

    mu_sr = n_p / queue_time_per_round  # Do not need this.

    p_queue = (R * queue_time_per_round) / n_p  # average queue load (prob queue is empty) lmda/mu_sr


    mu_st = queue_time_per_round / n_p
    var_st = (num_d ** 2 * (sigma_recv ** 2) + num_s ** 2 * (sigma_send ** 2)) / n_p ** 2
    C_st = var_st / mu_st ** 2
    # sigma_st = math.sqrt(var_st)  #do not need this.

    # Marchal's approximation for G/G/1 queue
    L_q = (p_queue**2*(1+C_st)*(C_a+C_st*p_queue**2))/(2*(1-p_queue)*(1+C_st*p_queue**2))
    wait_queue = L_q / lmda
    return wait_queue
