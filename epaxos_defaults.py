import numpy as np
import math
import model


def load_latencies(mu_net_filename, mu_net_std_dev_filename):
    mu_remote = np.loadtxt(mu_net_filename, delimiter=",")
    sigma_remote = np.loadtxt(mu_net_std_dev_filename, delimiter=",")
    return mu_remote, sigma_remote

def calc_quorums(mu_nodes):
    N = len(mu_nodes)
    F = int(math.floor((N - 1) / 2))
    fast_q = int(F + math.floor((F + 1) / 2))
    slow_q = F + 1
    return N, fast_q, slow_q

def calc_mu_r(R):
    mu_r = []
    sigma_r = []
    mr = 1000.0 / R
    for r in range(0, N):
        mu_r.append(mr)
        sigma_r.append(mr / 0.5)  # give it some good round spread

    return mu_r, sigma_r

# REFERENCE SAMPLE VALUES FOR WPAXOS PARAMETERS '''
mu_net_filename = "params/mu_net_remote.csv"
mu_net_std_dev_filename = "params/sigma_net_remote.csv"

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

msgSize = 110  # 100 bytes
netSpeed = 980e6  # 98 mbits/sec
netSpeedStdDev = 30e5  # 0.3 mbits/sec

ttx, ttx_stddev = model.computeTTX(msgSize=msgSize, netSpeed=netSpeed, netSpeedStdDev=netSpeedStdDev)  # time to transmit in ms

mu_ms = 0.002  # message serialization overhead in ms
sigma_ms = 0.010

mu_md = 0.050  # message deserialization overhead in ms
sigma_md = 0.030

n_p = 1  # number of pipelines

conflict = 0.02

# regions form a graph with edges being communication links, and weights being mean communication latencies
mu_remote = np.loadtxt(mu_net_filename, delimiter=",")
# and weights being std. deviations as well
sigma_remote = np.loadtxt(mu_net_std_dev_filename, delimiter=",")

N, fast_q, slow_q = calc_quorums(mu_remote)

R = 6000  # Throughput in rounds/sec for each region
mu_r, sigma_r = calc_mu_r(R)
