import model_multipaxos_wan as model
import paxos_defaults as params
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

repeats = 1

start_tp = 100
tp_step = 500

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.title('Throughput vs. Latency [Model]')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))

n = 5  # 5 nodes

for leader in range(0, 5, 1):
    lats = {}
    Rmax = n_p / (n * params.mu_md + 2 * params.mu_ms) * 1000
    end_tp = int(Rmax) + 1
    print "end tp = " + str(end_tp)
    print "Rmax = " + str(Rmax)
    r = start_tp
    tp_step = 500
    tp = []
    while r < end_tp:

        for i in range(0, repeats):
            print "tick: " + str(n) + "," + str(r) + ","+str(i)
            mu_r = 1000.0 / r
            sigma_r = mu_r / 0.5
            numops, simlats = model.model_random_round_arrival(
                mu_nodes=params.mu_nodes,
                sigma_nodes=params.sigma_nodes,
                mu_local=params.mu_local,
                leader_id=leader,
                qs=n / 2 + 1,  # majority quorum
                mu_ms=params.mu_ms,
                sigma_ms=params.sigma_ms,
                mu_md=params.mu_md,
                sigma_md=params.sigma_md,
                n_p=n_p,
                mu_r=mu_r,
                sigma_r=sigma_r,
                sim_clients=True
            )

            l = np.average(simlats) * 1000  # convert to ms
            # print "average = " + str(l) + " ms"
            if r in lats:
                lats[r] += l
            else:
                lats[r] = l
        lats[r] /= repeats
        # print lats
        tp.append(r)
        if r + 600 > end_tp:
            tp_step = 50
        r += tp_step


    lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
    print lat
    lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
    p2 = ax.plot(tp, lat, marker='o', label=lbl[leader] + " Leader")

plt.ylim(40, 200)
legend = ax.legend(loc='upper left', shadow=True)
plt.show()
