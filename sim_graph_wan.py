import model_multipaxos_wan as model
import model_epaxos as emodel
import model_wpaxos as wmodel
import model_multipaxos as model_lan
import paxos_defaults as params
import epaxos_defaults as eparams
import wpaxos_defaults as wparams
import matplotlib.pyplot as plt
import numpy as np
import math

params.mu_md *= 0.5
wparams.mu_md *= 0.5
eparams.mu_md *= 0.5

params.mu_ms *= 0.5
wparams.mu_ms *= 0.5
eparams.mu_ms *= 0.5

lbl = ['VA', 'OR', 'CA', 'IR', 'JP']
colors = ['g', 'r', 'b', 'c', 'k']
markers = ['s', '*', 'o', 'X', 'D']
colorID = 0
start_tp = 100
n_p = 1

fig, ax = plt.subplots()
plt.xlabel('Aggregate Throughput (rounds/sec)')
plt.ylabel('Latency (ms)')
#plt.title('Throughput vs. Latency [Model]')
plt.rc('lines', linewidth=1)


n = 5  # 5 nodes
leader = 2

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
tp_step = 2000
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

    rated_avg = 0

    for i in range(0, n):
        lats_all[i].append(simlats[i] * 1000)
        tp_all[i].append(numops[i])
        rated_avg += simlats[i] * numops[i]

    rated_avg /= r

    lats_av.append(rated_avg * 1000)  # convert to ms
    tp_agg.append(r)

    if r + 600 > end_tp:
        tp_step = 50
    r += tp_step


print lats_av

p1 = ax.plot(tp_agg, lats_av, marker=markers[colorID % 5], color=colors[colorID % 5], label="MultiPaxos (" + lbl[leader] + " Leader)")
colorID += 1


#for zone in range(0, len(lats_all)):
#    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")


#FPaxos
tp_step = 2000
n = 5  # 5 nodes

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
print "FPaxos Rmax = " + str(Rmax)
r = start_tp
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
        qs=2,  # |q2| = 2
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

    rated_avg = 0

    for i in range(0, n):
        lats_all[i].append(simlats[i] * 1000)
        tp_all[i].append(numops[i])
        rated_avg += simlats[i] * numops[i]

    rated_avg /= r

    lats_av.append(rated_avg * 1000)  # convert to ms
    tp_agg.append(r)

    if r + 600 > end_tp:
        tp_step = 50
    r += tp_step

p1 = ax.plot(tp_agg, lats_av, marker=markers[colorID % 5], color=colors[colorID % 5], label="FPaxos (" + lbl[leader] + " Leader)")
colorID += 1

#for zone in range(0, len(lats_all)):
#    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")



#EPaxos
tp_step = 300

eparams.conflict = 0.3

Rmax = emodel.get_max_throughput(len(eparams.mu_remote), eparams.conflict, eparams.mu_ms, eparams.mu_md, eparams.ttx, eparams.n_p, True)
end_tp = Rmax[0] - 10

tp_all = []
lats_all = []

for i in range(0, n):
    lats_all.append([])
    tp_all.append([])

lats_av = []
tp_agg = []

r = start_tp
eparams.mu_r, eparams.sigma_r = eparams.calc_mu_r(r)
while r < end_tp:

    print "tick: " + str(len(eparams.mu_remote)) + "," + str(r)

    R, Lr = emodel.model_random_round_arrival(
        mu_nodes=eparams.mu_remote,
        sigma_nodes=eparams.sigma_remote,
        mu_local=eparams.mu_local,
        qs=eparams.slow_q,
        fqs=eparams.fast_q,
        conflict_rate=eparams.conflict,
        mu_ms=eparams.mu_ms,
        sigma_ms=eparams.sigma_ms,
        mu_md=eparams.mu_md,
        sigma_md=eparams.sigma_md,
        ttx=eparams.ttx,
        ttx_stddev=eparams.ttx_stddev,
        mu_r=eparams.mu_r,
        sigma_r=eparams.mu_r,
        n_p=eparams.n_p,
        sim_clients=True
    )

    rated_avg = 0
    tp_agr = np.sum(R);

    for i in range(0, n):
        lats_all[i].append(Lr[i] * 1000)
        tp_all[i].append(R[i])
        rated_avg += Lr[i] * R[i]

    rated_avg /= tp_agr

    lats_av.append(rated_avg * 1000)  # convert to ms
    tp_agg.append(tp_agr)

    print rated_avg

    if r + 600 > end_tp:
        tp_step = 50
    r += tp_step
    eparams.mu_r, eparams.sigma_r = eparams.calc_mu_r(r)

    #params.conflict = params.conflict + (r / 50000.0)
    print eparams.conflict


p1 = ax.plot(tp_agg, lats_av, marker=markers[colorID % 5], color=colors[colorID % 5], label="EPaxos (Conflict=0.3)")
colorID += 1

#for zone in range(0, len(lats_all)):
#    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")

#EPaxos
tp_step = 200

eparams.conflict = 0.02

Rmax = emodel.get_max_throughput(len(eparams.mu_remote), 0.55, eparams.mu_ms, eparams.mu_md, eparams.ttx, eparams.n_p, True)
end_tp = Rmax[0] - 10

tp_all = []
lats_all = []

for i in range(0, n):
    lats_all.append([])
    tp_all.append([])

lats_av = []
tp_agg = []

r = start_tp
eparams.mu_r, eparams.sigma_r = eparams.calc_mu_r(r)


while r < end_tp:

    print "tick: " + str(len(eparams.mu_remote)) + "," + str(r)

    R, Lr = emodel.model_random_round_arrival(
        mu_nodes=eparams.mu_remote,
        sigma_nodes=eparams.sigma_remote,
        mu_local=eparams.mu_local,
        qs=eparams.slow_q,
        fqs=eparams.fast_q,
        conflict_rate=eparams.conflict,
        mu_ms=eparams.mu_ms,
        sigma_ms=eparams.sigma_ms,
        mu_md=eparams.mu_md,
        sigma_md=eparams.sigma_md,
        ttx=eparams.ttx,
        ttx_stddev=eparams.ttx_stddev,
        mu_r=eparams.mu_r,
        sigma_r=eparams.mu_r,
        n_p=eparams.n_p,
        sim_clients=True
    )

    rated_avg = 0
    tp_agr = np.sum(R);

    for i in range(0, n):
        lats_all[i].append(Lr[i] * 1000)
        tp_all[i].append(R[i])
        rated_avg += Lr[i] * R[i]

    rated_avg /= tp_agr

    lats_av.append(rated_avg * 1000)  # convert to ms
    tp_agg.append(tp_agr)

    eparams.conflict = 0.02*math.pow(1.075, tp_agr / 275)

    #eparams.conflict = eparams.conflict + (r / cf)
    print eparams.conflict

    r += tp_step
    eparams.mu_r, eparams.sigma_r = eparams.calc_mu_r(r)



p1 = ax.plot(tp_agg, lats_av, marker=markers[colorID % 5], color=colors[colorID % 5], label="EPaxos (Conflict= [0.02, " + "{0:.2f}".format(eparams.conflict) +"])")
colorID += 1

#for zone in range(0, len(lats_all)):
#    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")


#WPaxos

wparams.mu_ms *= 7
wparams.sigma_ms = math.sqrt((7 * wparams.sigma_md) ** 2)
wparams.mu_md *= 7
wparams.sigma_md = math.sqrt((7 * wparams.sigma_md) ** 2)
wparams.ttx *= 7
wparams.ttx_stddev = math.sqrt((7 * wparams.ttx_stddev) ** 2)

wparams.p_remote_steal = 0.10

tp_step=1000
tp_all = []
lats_all = []

for i in range(0, n):
    lats_all.append([])
    tp_all.append([])

lats_av = []
tp_agg = []

r = start_tp
mu_r = 1000.0 / r
sigma_r = mu_r / 0.5

end_tp = 7550
print "end tp = " + str(end_tp)

while r < end_tp:
    if r > 5700:
        tp_step = 500
    if r > 6500:
        tp_step = 250
    if r > 7100:
        tp_step = 50

    #print params.sigma_ms

    L_local_ops, L_remote_ops, L_average_round = wmodel.model_random_round_arrival(
        rows=wparams.rows,
        cols=wparams.columns,
        q2regions=wparams.q2regions,
        q1nodes_per_column=wparams.q1nodes_per_column,
        q1s=wparams.q1s,
        q2nodes_per_column=wparams.q2nodes_per_column,
        mu_local=wparams.mu_local,
        sigma_local=wparams.sigma_local,
        mu_ms=wparams.mu_ms,
        sigma_ms=wparams.sigma_ms,
        mu_md=wparams.mu_md,
        object_ownership=wparams.object_ownership,
        sigma_md=wparams.sigma_md,
        n_p=n_p,
        mu_r=mu_r,
        sigma_r=sigma_r,
        ttx=wparams.ttx,
        ttx_stddev=wparams.ttx_stddev,
        mu_remote=wparams.mu_remote,
        sigma_remote=wparams.sigma_remote,
        client_contact_probability=wparams.client_contact_probability,
        locality=wparams.locality,
        p_remote_steal=wparams.p_remote_steal
    )

    rated_avg = 0
    tp_agr = r * 5

    L_region = np.sum(L_average_round, axis=0)
    L_region /= wparams.rows
    L_region *= 1000  # convert from sec to ms

    for i in range(0, 5):
        lats_all[i].append(L_region[i])
        tp_all[i].append(r)
        rated_avg += L_region[i]

    rated_avg /= 5

    lats_av.append(rated_avg)  # convert to ms
    tp_agg.append(tp_agr)


    mu_r = 1000.0 / r
    sigma_r = mu_r / 0.5
    r += tp_step



p1 = ax.plot(tp_agg, lats_av, marker=markers[colorID % 5], color=colors[colorID % 5], label="WPaxos (Locality=0.7)")
colorID += 1

#for zone in range(0, len(lats_all)):
#    p2 = ax.plot(tp_all[zone], lats_all[zone], marker=markers[zone % 5], color=colors[zone % 5], label=lbl[zone] + "(" + lbl[leader] + " Leader)")

plt.ylim(0, 200)
legend = ax.legend(loc='upper right', shadow=True)
plt.show()