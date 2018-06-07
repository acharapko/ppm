import model_multipaxos_wan as model
import model_multipaxos as model_lan
import paxos_defaults as params
import matplotlib.pyplot as plt
import numpy as np

params.mu_md *= 0.5

params.mu_ms *= 0.5

colors = ['g', 'r', 'b', 'c', 'k']
markers = ['s', '*', 'o', 'X', 'D']
colorID = 0

start_tp = 100
tp_step = 500

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
#plt.title('Throughput vs. Latency [Model]')
plt.rc('lines', linewidth=1)

n = 5  # 5 nodes
leader = 3

tp_all = []
lats_all = []

for i in range(0, n):
    lats_all.append([])
    tp_all.append([])

lats_av = []
tp_agg = []


Rmax = model_lan.computeRmax(n, n_p, params.mu_md, params.mu_ms, params.ttx)
end_tp = int(Rmax) + 1
print "end tp = " + str(end_tp)
print "Rmax = " + str(Rmax)
r = start_tp
tp_step = 500
while r < end_tp:

    print "tick: " + str(n) + "," + str(r)
    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5
    numops, simlats = model.model_random_round_arrival(
        mu_nodes=params.mu_nodes,
        sigma_nodes=params.sigma_nodes,
        mu_local=params.mu_local,
        leader_id=leader,
        cmd_distrib=[0.2, 0.2, 0.2, 0.2, 0.2],
        #cmd_distrib=[0.13, 0.25, 0.35, 0.15, 0.12],
        qs=n / 2 + 1,  # majority quorum
        mu_ms=params.mu_ms,
        sigma_ms=params.sigma_ms,
        mu_md=params.mu_md,
        sigma_md=params.sigma_md,
        ttx=params.ttx,
        ttx_stddev=params.ttx_stddev,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        sim_clients=True
    )

    print simlats

    for i in range(0, n):
        lats_all[i].append(simlats[i] * 1000)
        tp_all[i].append(numops[i])



    lats_av.append(np.average(simlats) * 1000)  # convert to ms
    tp_agg.append(r)

    if r + 600 > end_tp:
        tp_step = 50
    r += tp_step


print lats_av
lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
#p1 = ax.plot(tp_agg, lats_av, marker='x', label="Average (" + lbl[leader] + " Leader)")


for zone in range(0, len(lats_all)):
    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")

plt.ylim(40, 600)
legend = ax.legend(loc='upper left', shadow=True)
plt.show()