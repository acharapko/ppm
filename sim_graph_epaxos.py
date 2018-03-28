import epaxos_defaults as params
import model_epaxos as model
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

start_tp = 100

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.title('Throughput vs. Latency [EPAXOS 5 NODES; 5 REGIONS]')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))

Rmax = model.get_max_throughput(len(params.mu_remote), params.conflict, params.mu_ms, params.mu_md, params.n_p, True)
end_tp = Rmax[0] - 50
print "end tp = " + str(end_tp)
lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
lats = []
tp = []
#print "Rmax = " + str(Rmax)
r = start_tp
while r < end_tp:

    tp_step = (Rmax[0] - r)/5
    print tp_step
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
        mu_r=params.mu_r,
        sigma_r=params.mu_r,
        n_p=params.n_p,
        sim_clients=True
    )


    if len(lats) == 0:
        for i in range(0, len(Lr)):
            lats.append([])

    for i in range(0, len(Lr)):
        lats[i].append(Lr[i] * 1000)


for i in range(0, len(lats)):
    p2 = ax.plot(tp, lats[i], marker='o', label=lbl[i])


#plt.ylim(0.0, 100.00)
legend = ax.legend(loc='center left', shadow=True)
plt.show()
