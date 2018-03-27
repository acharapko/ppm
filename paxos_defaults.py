import numpy as np

#  REFERENCE SAMPLE VALUES FOR MULTIPAXOS/FPAXOS PARAMETERS
t = 60  # sim duration in seconds
N = 3  # number of nodes in the cluster

qs = N / 2 + 1  # quorum size. For Paxos it is majority

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

mu_ms = 0.001  # message serialization overhead in ms
sigma_ms = 0.005

mu_md = 0.025  # message deserialization overhead in ms
sigma_md = 0.015

n_p = 1  # number of pipelines

R = 6000  # Throughput in rounds/sec
mu_r = 1000.0 / R
sigma_r = mu_r / 0.5  # give it some good round spread

# nodes form a graph with edges being communication links, and weights being mean communication latencies
# this parameter is used by model_multipaxos_wan when distances between paxos nodes are not identical
mu_nodes = np.loadtxt("params/mu_net_remote.csv", delimiter=",")
sigma_nodes = np.loadtxt("params/sigma_net_remote.csv", delimiter=",")