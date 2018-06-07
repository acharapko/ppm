import model_multipaxos as model
import model as m
import paxos_defaults as params
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler


fig, ax = plt.subplots()
plt.xlabel('Cluster Size (Nodes)')
plt.ylabel('Throughput (transactions/s)')

plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))

tp = []
tp100 = []
tp2020 = []

n1 = []
n100 = []
n2020 = []
btp = []

numBlocks = 6


blockSize = 1024*1024    # bytes
txSize = 495    # bytes
txPerBlock = blockSize / txSize
netSpeed = 100.0  # 100 Mbit/s

for n in range(2000, 10000, 200):
    n2020.append(n)

    t = ((blockSize * n) * 8) / 1000000
    t /= netSpeed
    tp2020.append(1 / t * txPerBlock)

    if t > 0:

        tp.append(t)
        tp100.append(t*100)
        n1.append(n)
        n100.append(n)

        btp_temp = numBlocks * txPerBlock / 3600.0
        #print(btp_temp)
        btp.append(btp_temp)

p1 = ax.plot(n2020, tp2020, marker='o', label="Paxos-2020")

p3 = ax.plot(n100, btp, marker='.', label="Blockchain-2020")

legend = ax.legend(loc='upper right', shadow=True)

plt.show()

'''
for n in range(300, 1300, 200):
    lats = {}
    Rmax = n_p / (n * params.mu_md + 2 * params.mu_ms) * 1000
    tp_step = 4
    end_tp = int(Rmax) - 3
    print "end tp = " + str(end_tp)
    print "Rmax = " + str(Rmax)
    tp = []
    r = start_tp
    #for r in range(start_tp, end_tp, tp_step):
    while r < end_tp:

        tp.append(r)
        r += tp_step
        for i in range(0, repeats):
            print "tick: " + str(n) + "," + str(r) + ","+str(i)
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

    #tp = range(start_tp, end_tp, tp_step)
    lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
    print lat
    p2 = ax.plot(tp, lat, marker='o', label=str(n) + " Nodes")

plt.ylim(0, 500)
legend = ax.legend(loc='lower right', shadow=True)
plt.show()
'''