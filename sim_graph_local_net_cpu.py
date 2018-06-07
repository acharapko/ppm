import model_multipaxos as paxos
import model_epaxos as epaxos
import paxos_defaults as paxos_params
import epaxos_defaults as epaxos_params
import matplotlib.pyplot as plt
import numpy as np
import math

paxos_params.mu_md *= 0.5

paxos_params.mu_ms *= 0.5

from cycler import cycler

start_tp = 500
tp_step = 200

n_p = 1

colors = ['g', 'r', 'b', 'c', 'k']
markers = ['s', '*', 'o', 'X', 'D']
colorID = 0
n = 9

fig, ax = plt.subplots()
plt.ylabel('Max Throughput (rounds/sec)')
plt.xlabel('Network Bandwidth (mbps)')

epaxRmax = []
pmr = []
pmr5 = []
conflict = 0.1
mu_net, sigma_net = epaxos_params.load_latencies('params/mu_net_local_epaxos.csv', 'params/sigma_net_local_epaxos.csv')
for b in range(10, 1000, 20):
    netSpeed = b * 1e6
    ttxsec = paxos_params.msgSize * 8 / netSpeed
    paxos_params.ttx = ttxsec * 1000
    epaxos_params.ttx = ttxsec * 1000

    paxosRmax = paxos.computeRmax(n, n_p, paxos_params.mu_md, paxos_params.mu_ms, paxos_params.ttx)
    pmr.append(paxosRmax)

    paxosRmax5 = paxos.computeRmax(5, n_p, paxos_params.mu_md, paxos_params.mu_ms, paxos_params.ttx)
    pmr5.append(paxosRmax5)

    rm = epaxos.get_max_throughput(len(mu_net), conflict, epaxos_params.mu_ms, epaxos_params.mu_md, epaxos_params.ttx, epaxos_params.n_p, True)
    epaxRmax.append(np.sum(rm))

cl = range(10, 1000, 20)
p2 = ax.plot(cl, epaxRmax, marker=markers[colorID % 5], color=colors[colorID % 5], label="EPaxos (9 nodes)")
colorID += 1
p3 = ax.plot(cl, pmr, marker=markers[colorID % 5], color=colors[colorID % 5], label="MultiPaxos (9 nodes)")
colorID += 1
p3 = ax.plot(cl, pmr5, marker=markers[colorID % 5], color=colors[colorID % 5], label="MultiPaxos (5 nodes)")
colorID += 1
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
