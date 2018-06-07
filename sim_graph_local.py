import model_multipaxos as paxos
import model_epaxos as epaxos
import model_wpaxos as wpaxos
import paxos_defaults as paxos_params
import epaxos_defaults as epaxos_params
import wpaxos_defaults as wpaxos_params
import matplotlib.pyplot as plt
import numpy as np
import math

#double the speed from t2.small to about m5.large
paxos_params.mu_md *= 0.5
epaxos_params.mu_md *= 0.5
wpaxos_params.mu_md *= 0.5

paxos_params.mu_ms *= 0.5
epaxos_params.mu_ms *= 0.5
wpaxos_params.mu_ms *= 0.5

from cycler import cycler

start_tp = 500
tp_step = 200

n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
plt.rc('lines', linewidth=1)
colors = ['g', 'r', 'b', 'c', 'k']
markers = ['s', '*', 'o', 'X', 'D']
colorID = 0
n = 9

# MultiPaxos
lats = {}
Rmax = paxos.computeRmax(n, n_p, paxos_params.mu_md, paxos_params.mu_ms, paxos_params.ttx)
end_tp = int(Rmax) - 200
print "end tp = " + str(end_tp)
print "Rmax = " + str(Rmax)
tp = []
r = start_tp
while r < end_tp:
    tp.append(r)
    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5
    numops, simlats = paxos.model_random_round_arrival(
        N=n,
        qs=n / 2 + 1,  # majority quorum
        mu_local=paxos_params.mu_local,
        sigma_local=paxos_params.sigma_local,
        mu_ms=paxos_params.mu_ms,
        sigma_ms=paxos_params.sigma_ms,
        mu_md=paxos_params.mu_md,
        sigma_md=paxos_params.sigma_md,
        ttx=paxos_params.ttx,
        ttx_stddev=paxos_params.ttx_stddev,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        sim_clients=True
    )
    print "tick: " + str(n) + "," + str(r) + ' - ' + str(simlats) + ' s'
    r += tp_step
    l = np.average(simlats) * 1000  # convert to ms
    # print "average = " + str(l) + " ms"
    if r in lats:
        lats[r] += l
    else:
        lats[r] = l

lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
p2 = ax.plot(tp, lat, marker=markers[colorID % 5], color=colors[colorID % 5], label="MultiPaxos")
colorID += 1


#FPaxos |q2| = 3
lats = {}
Rmax = paxos.computeRmax(n, n_p, paxos_params.mu_md, paxos_params.mu_ms, paxos_params.ttx)
end_tp = int(Rmax) + 1
print "FPaxos end tp = " + str(end_tp)
print "FPaxos Rmax = " + str(Rmax)
tp = []
r = start_tp
while r < end_tp:
    tp.append(r)
    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5
    numops, simlats = paxos.model_random_round_arrival(
        N=n,
        qs=3,
        mu_local=paxos_params.mu_local,
        sigma_local=paxos_params.sigma_local,
        mu_ms=paxos_params.mu_ms,
        sigma_ms=paxos_params.sigma_ms,
        mu_md=paxos_params.mu_md,
        sigma_md=paxos_params.sigma_md,
        ttx=paxos_params.ttx,
        ttx_stddev=paxos_params.ttx_stddev,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        sim_clients=True
    )
    print "tick: " + str(n) + "," + str(r) + ' - ' + str(simlats) + ' s'
    r += tp_step

    l = np.average(simlats) * 1000  # convert to ms
    # print "average = " + str(l) + " ms"
    if r in lats:
        lats[r] += l
    else:
        lats[r] = l

tp = range(start_tp, end_tp, tp_step)
lat = [lats[key] for key in sorted(lats.keys(), reverse=False)]
p2 = ax.plot(tp, lat, marker=markers[colorID % 5], color=colors[colorID % 5], label="FPaxos " + str(n) + " Nodes (|q2|=3)")
colorID += 1


#EPaxos Local
tp_step = 200
lats = []
tp = []
epaxos_params.conflict = 0.75
mu_net, sigma_net = epaxos_params.load_latencies('params/mu_net_local_epaxos.csv', 'params/sigma_net_local_epaxos.csv')
epaxos_params.N, epaxos_params.fast_q, epaxos_params.slow_q = epaxos_params.calc_quorums(mu_net)

Rmax = epaxos.get_max_throughput(len(mu_net), epaxos_params.conflict, epaxos_params.mu_ms, epaxos_params.mu_md, epaxos_params.ttx, epaxos_params.n_p, True)
end_tp = Rmax[0]
r = 25
print "start TP:" + str(start_tp)
print "EPaxos Rmax:" + str(Rmax)
while r < end_tp:

    tp_step = (Rmax[0] - r)/2
    if tp_step < 10:
        tp_step = 10

    if tp_step > 200:
       tp_step = 200

    tp.append(r * 9)
    epaxos_params.mu_r, epaxos_params.sigma_r = epaxos_params.calc_mu_r(r)

    R, Lr = epaxos.model_random_round_arrival(
        mu_nodes=mu_net,
        sigma_nodes=sigma_net,
        mu_local=epaxos_params.mu_local,
        qs=epaxos_params.slow_q,
        fqs=epaxos_params.fast_q,
        conflict_rate=epaxos_params.conflict,
        mu_ms=epaxos_params.mu_ms,
        sigma_ms=epaxos_params.sigma_ms,
        mu_md=epaxos_params.mu_md,
        sigma_md=epaxos_params.sigma_md,
        ttx=epaxos_params.ttx,
        ttx_stddev=epaxos_params.ttx_stddev,
        mu_r=epaxos_params.mu_r,
        sigma_r=epaxos_params.mu_r,
        n_p=epaxos_params.n_p,
        sim_clients=True
    )
    print "tick: " + str(n) + "," + str(r) + ' mu_r = ' + str(epaxos_params.mu_r) + ' ms'
    r += tp_step

    lrsum = np.sum(Lr)
    lrsum = lrsum / 9

    #epaxos_params.conflict = epaxos_params.conflict + (r / 100000.0)
    #print epaxos_params.conflict

    lats.append(lrsum * 1000)

p2 = ax.plot(tp, lats, marker=markers[colorID % 5], color=colors[colorID % 5], label="EPaxos")
colorID += 1

#Dynamo

# Dynamo - we approximate dynamo as a tweak on WPaxos.  out of 9 nodes, 3 are used to replicate.
# We do not model the ring, but number of messages should be correct
# here we assume write quorum W = 1

wpaxos_params.p_remote_steal = 0  # no stealing in dynamo
wpaxos_params.mu_remote, wpaxos_params.sigma_remote = wpaxos_params.load_latencies('params/mu_net_local_wpaxos.csv', 'params/sigma_net_local_wpaxos.csv')
wpaxos_params.locality = wpaxos_params.load_locality('params/zone_locality_uniform.csv')

wpaxos_params.columns = len(wpaxos_params.mu_remote)  # columns is number of regions

wpaxos_params.q2regions = [[[0]], [[1]], [[2]]]
wpaxos_params.q2nodes_per_column = 1

# probability of client reaching out to a node. row-region, column-nodeID
wpaxos_params.client_contact_probability = np.full((wpaxos_params.rows, wpaxos_params.columns), 1.0 / wpaxos_params.rows, dtype=float)

# Object ownership
wpaxos_params.object_ownership = np.full((wpaxos_params.rows, wpaxos_params.columns), 1.0 / (wpaxos_params.rows*wpaxos_params.columns), dtype=float)


end_tp = 69800
tp_step = 3000
lats = []
tp = []
#print "Rmax = " + str(Rmax)
r = start_tp / 3 # in Dynamo r is per region!
while r < end_tp:
    if r > 65700:
        tp_step = 200
    tp.append(r*3)
    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5
    print "mu_r=" + str(mu_r)
    #print params.sigma_ms

    L_average_round = wpaxos.model_dynamo_random_round_arrival(
        rows=wpaxos_params.rows,
        cols=wpaxos_params.columns,
        q2regions=wpaxos_params.q2regions,
        q2nodes_per_column=wpaxos_params.q2nodes_per_column,
        mu_local=wpaxos_params.mu_local,
        sigma_local=wpaxos_params.sigma_local,
        mu_ms=wpaxos_params.mu_ms,
        sigma_ms=wpaxos_params.sigma_ms,
        mu_md=wpaxos_params.mu_md,
        object_ownership=wpaxos_params.object_ownership,
        sigma_md=wpaxos_params.sigma_md,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        ttx=wpaxos_params.ttx,
        ttx_stddev=wpaxos_params.ttx_stddev,
        mu_remote=wpaxos_params.mu_remote,
        sigma_remote=wpaxos_params.sigma_remote,
        client_contact_probability=wpaxos_params.client_contact_probability,
        locality=wpaxos_params.locality,
        p_remote_steal=wpaxos_params.p_remote_steal
    )

    r += tp_step

    L_region = np.sum(L_average_round, axis=0)
    L_region /= wpaxos_params.rows
    L_region *= 1000  # convert from sec to ms

    Lall = np.sum(L_region)
    Lall /= 3
    lats.append(Lall)


p2 = ax.plot(tp, lats, marker=markers[colorID % 5], color=colors[colorID % 5], label="Dynamo")
colorID += 1


#WPaxos
wpaxos_params.p_remote_steal = 0.01
wpaxos_params.mu_ms *= 4
wpaxos_params.sigma_ms = math.sqrt((4 * wpaxos_params.sigma_ms) ** 2)
wpaxos_params.mu_md *= 4
wpaxos_params.sigma_md = math.sqrt((4 * wpaxos_params.sigma_md) ** 2)
wpaxos_params.ttx *= 4
wpaxos_params.ttx_stddev = math.sqrt((4 * wpaxos_params.ttx_stddev) ** 2)

wpaxos_params.mu_remote, wpaxos_params.sigma_remote = wpaxos_params.load_latencies('params/mu_net_local_wpaxos.csv', 'params/sigma_net_local_wpaxos.csv')
wpaxos_params.locality = wpaxos_params.load_locality('params/zone_locality_uniform.csv')


wpaxos_params.columns = len(wpaxos_params.mu_remote)  # columns is number of regions


wpaxos_params.q1s = 1 * wpaxos_params.q1nodes_per_column * wpaxos_params.columns  # q1 size: rows x nodes/per column * columns

wpaxos_params.q2regions = [[[0]], [[1]], [[2]]]
wpaxos_params.q2nodes_per_column = wpaxos_params.q1nodes_per_column + 1

# probability of client reaching out to a node. row-region, column-nodeID
wpaxos_params.client_contact_probability = np.full((wpaxos_params.rows, wpaxos_params.columns), 1.0 / wpaxos_params.rows, dtype=float)

# Object ownership
wpaxos_params.object_ownership = np.full((wpaxos_params.rows, wpaxos_params.columns), 1.0 / (wpaxos_params.rows*wpaxos_params.columns), dtype=float)


end_tp = 13300
tp_step = 1000
lats = []
tp = []
#print "Rmax = " + str(Rmax)
r = start_tp / 3  # in WPaxos r is per region!
while r < end_tp:
    if r > 10200:
        tp_step = 100
    tp.append(r*3)
    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5

    #print params.sigma_ms

    L_local_ops, L_remote_ops, L_average_round = wpaxos.model_random_round_arrival(
        rows=wpaxos_params.rows,
        cols=wpaxos_params.columns,
        q2regions=wpaxos_params.q2regions,
        q1nodes_per_column=wpaxos_params.q1nodes_per_column,
        q1s=wpaxos_params.q1s,
        q2nodes_per_column=wpaxos_params.q2nodes_per_column,
        mu_local=wpaxos_params.mu_local,
        sigma_local=wpaxos_params.sigma_local,
        mu_ms=wpaxos_params.mu_ms,
        sigma_ms=wpaxos_params.sigma_ms,
        mu_md=wpaxos_params.mu_md,
        object_ownership=wpaxos_params.object_ownership,
        sigma_md=wpaxos_params.sigma_md,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        ttx=wpaxos_params.ttx,
        ttx_stddev=wpaxos_params.ttx_stddev,
        mu_remote=wpaxos_params.mu_remote,
        sigma_remote=wpaxos_params.sigma_remote,
        client_contact_probability=wpaxos_params.client_contact_probability,
        locality=wpaxos_params.locality,
        p_remote_steal=wpaxos_params.p_remote_steal
    )
    print "tick: " + str(n) + "," + str(r) + ' mu_r = ' + str(mu_r) + ' ms'
    r += tp_step
    L_region = np.sum(L_average_round, axis=0)
    L_region /= wpaxos_params.rows
    L_region *= 1000  # convert from sec to ms

    Lall = np.sum(L_region)
    Lall /= 3
    lats.append(Lall)


p2 = ax.plot(tp, lats, marker=markers[colorID % 5], color=colors[colorID % 5], label="WPaxos")
colorID += 1



plt.ylim(0.0, 8)
plt.xlim(0.0, 38000)

legend = ax.legend(loc='upper right', shadow=True)
plt.show()
