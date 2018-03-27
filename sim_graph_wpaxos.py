import wpaxos_defaults as params
import model_wpaxos as model
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

start_tp = 100
tp_step = 1000

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.title('Throughput vs. Latency [5 REGIONS]')
plt.rc('lines', linewidth=1)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y', 'k'])))

end_tp = 30800
print "end tp = " + str(end_tp)
lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
lats = []
tp = []
#print "Rmax = " + str(Rmax)
r = start_tp
while r < end_tp:
    if r > 30000:
        tp_step = 100
    tp.append(r)
    mu_r = 1000.0 / r
    r += tp_step
    sigma_r = mu_r / 0.5

    #print params.sigma_ms

    L_local_ops, L_remote_ops, L_average_round = model.model_random_round_arrival(
        rows=params.rows,
        cols=params.columns,
        q2regions=params.q2regions,
        q1nodes_per_column=params.q1nodes_per_column,
        q1s=params.q1s,
        q2nodes_per_column=params.q2nodes_per_column,
        mu_local=params.mu_local,
        sigma_local=params.sigma_local,
        mu_ms=params.mu_ms,
        sigma_ms=params.sigma_ms,
        mu_md=params.mu_md,
        object_ownership=params.object_ownership,
        sigma_md=params.sigma_md,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        mu_remote=params.mu_remote,
        sigma_remote=params.sigma_remote,
        client_contact_probability=params.client_contact_probability,
        locality=params.locality,
        p_remote_steal=params.p_remote_steal
    )
    L_region = np.sum(L_average_round, axis=0)
    L_region /= params.rows
    L_region *= 1000  # convert from sec to ms
    #print "L_remote_ops:"
    #print L_remote_ops
    if len(lats) == 0:
        for i in range(0, len(L_region)):
            lats.append([])

    for i in range(0, len(L_region)):
        lats[i].append(L_region[i])


for i in range(0, len(lats)):
    #print lats[i]
    #print"---"
    p2 = ax.plot(tp, lats[i], marker='o', label=lbl[i])


plt.ylim(0.0, 100.00)
legend = ax.legend(loc='upper left', shadow=True)
plt.show()
