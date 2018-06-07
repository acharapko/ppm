import epaxos_defaults as params
import model_epaxos as model

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

params.mu_md *= 0.5

params.mu_ms *= 0.5

start_tp = 100

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))

Rmax = model.get_max_throughput(len(params.mu_remote), params.conflict, params.mu_ms, params.mu_md, params.ttx, params.n_p, True)
end_tp = Rmax[0] - 10
print "end tp = " + str(end_tp)
lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
colors = ['g', 'r', 'b', 'c', 'k']
markers = ['o', 's', '*', 'X', 'D']
lats = []
tp = []

r = start_tp
while r < end_tp:

    tp_step = (Rmax[0] - r)/10
    if tp_step < 10:
        tp_step = 10

    tp.append(r)
    params.mu_r = 1000.0 / r
    r += tp_step
    params.mu_r, params.sigma_r = params.calc_mu_r(r)

    R, Lr = model.model_random_round_arrival(
        mu_nodes=params.mu_remote,
        sigma_nodes=params.sigma_remote,
        mu_local=params.mu_local,
        qs=params.slow_q,
        fqs=params.fast_q,
        conflict_rate=params.conflict,
        mu_ms=params.mu_ms,
        sigma_ms=params.sigma_ms,
        mu_md=params.mu_md,
        sigma_md=params.sigma_md,
        ttx=params.ttx,
        ttx_stddev=params.ttx_stddev,
        mu_r=params.mu_r,
        sigma_r=params.mu_r,
        n_p=params.n_p,
        sim_clients=True
    )

    #params.conflict = params.conflict + (r / 50000.0)
    print params.conflict

    if len(lats) == 0:
        for i in range(0, len(Lr)):
            lats.append([])

    for i in range(0, len(Lr)):
        lats[i].append(Lr[i] * 1000)


for i in range(0, len(lats)):
    p2 = ax.plot(tp, lats[i], marker=markers[i % 5], color=colors[i % 5], label=lbl[i])

#plt.ylim(0.0, 100.00)
legend = ax.legend(loc='center left', shadow=True)
plt.show()

fig, ax = plt.subplots()
plt.ylabel('Throughput (rounds/sec)')
plt.xlabel('Conflict %')

Rmax = []
pmr = []
for c in range(0, 101, 5):
    rm = model.get_max_throughput(len(params.mu_remote), c/100.0, params.mu_ms, params.mu_md, params.ttx, params.n_p, True)
    Rmax.append(np.sum(rm))
    paxosRmax = (n_p / (len(params.mu_remote) * params.mu_md + 2 * params.mu_ms) * 1000)
    pmr.append(paxosRmax)

print Rmax
cl = range(0, 101, 5)
p2 = ax.plot(cl, Rmax, marker='o', label="EPaxos Throughput")


p3 = ax.plot(cl, pmr, marker='x', label="Paxos Throughput")

legend = ax.legend(loc='center left', shadow=True)
plt.show()