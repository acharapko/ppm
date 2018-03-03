import local_multipaxos as sim
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
plt.title('Throughput vs. Latency Simulation')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))


for n in range(3, 11, 2):
    lats = {}
    Rmax = n_p / (n * sim.mu_md + 2 * sim.mu_ms) * 1000
    end_tp = int(Rmax) + 1
    print "end tp = " + str(end_tp)
    print "Rmax = " + str(Rmax)
    for r in range(start_tp, end_tp, tp_step):
        for i in range(0, repeats):
            print "tick: " + str(n) + "," + str(r) + ","+str(i)
            mu_r = 1.0 / r
            sigma_r = mu_r / 0.5
            numops, simlats = sim.model_random_round_arrival(
                N=n,
                qs=n / 2 + 1,  # majority quorum
                mu_local=sim.mu_local,
                sigma_local=sim.sigma_local,
                mu_ms=sim.mu_ms,
                sigma_ms=sim.sigma_ms,
                mu_md=sim.mu_md,
                sigma_md=sim.sigma_md,
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

    tp = range(start_tp, end_tp, tp_step)
    lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
    print lat
    p2 = ax.plot(tp, lat, marker='o', label=str(n) + " Nodes")

plt.ylim(0.0, 2.00)
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
