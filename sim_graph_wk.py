import local_multipaxos as sim
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

repeats = 3

start_tp = 1000
end_tp = 21000
tp_step = 250

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.title('Throughput vs. Latency Simulation')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))


for wk_variance in range(2, 13, 2):
    wkv = wk_variance / 3.0
    lats = {}
    n = 5
    Rmax = n_p / (n * sim.mu_md + 2 * sim.mu_ms) * 1000
    end_tp = int(Rmax) + 1001
    print "end tp = " + str(end_tp)
    print "Rmax = " + str(Rmax)

    for r in range(start_tp, end_tp, tp_step):
        for i in range(0, repeats):
            print "tick: " + str(wk_variance) + "," + str(r) + ","+str(i)
            mu_r = 1.0 / r
            sigma_r = mu_r * wkv
            numops, simlats = sim.sim(
                t=30,
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
    wkvs = "{:5.2f}".format(wkv)
    p2 = ax.plot(tp, lat, marker='o', label=r'$\sigma_r=$' + wkvs + r'$\mu_r$')

plt.ylim(0.0, 3.50)
legend = ax.legend(loc='upper left', shadow=True)
plt.show()