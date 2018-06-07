import model_multipaxos as model
import model as model_std
import paxos_defaults as params
import matplotlib.pyplot as plt
import numpy as np

start_tp = 500
tp_step = 200

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.rc('lines', linewidth=1)
colors = ['g', 'r', 'b', 'c', 'k']
markers = ['o', 's', '*', 'X', 'D']
colorID = 0
n = 5

for m in range(100, 1000, 200):
    lats = {}
    ttx, ttx_stddev = model_std.computeTTX(m, params.netSpeed, params.netSpeedStdDev)
    Rmax = model.computeRmax(n, n_p, params.mu_md, params.mu_ms, ttx)
    tp_step = 200
    end_tp = int(Rmax) - 200
    print "end tp = " + str(end_tp)
    print "Rmax = " + str(Rmax)
    tp = []
    r = start_tp
    #for r in range(start_tp, end_tp, tp_step):
    while r < end_tp:

        tp.append(r)
        r += tp_step

        print "tick: " + str(n) + "," + str(r)
        mu_r = 1000.0 / r
        sigma_r = mu_r / 0.5

        numops, simlats = model.model_random_round_arrival(
            N=n,
            qs=n / 2 + 1,  # majority quorum
            mu_local=params.mu_local,
            sigma_local=params.sigma_local,
            mu_ms=params.mu_ms,
            sigma_ms=params.sigma_ms,
            mu_md=params.mu_md,
            sigma_md=params.sigma_md,
            ttx=ttx,
            ttx_stddev=ttx_stddev,
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
        # print lats

    #tp = range(start_tp, end_tp, tp_step)
    lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
    print lat
    p2 = ax.plot(tp, lat, marker=markers[colorID % 5], color=colors[colorID % 5], label=str(m) + " Bytes")
    colorID += 1

plt.ylim(0, 5)
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
