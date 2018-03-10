import numpy as np
import math
import model
import time

# Modeling WPaxos


# in each round, we have 4 possibilities:
#   (1) run phase 2 locally
#   (2) forward to another local node and run phase 2
#   (3) forward to remote node and run phase 2
#   (4) run phase 1 followed by phase 2


def compute_node_shares(cols, rows, object_ownership, zone_object_ownership, q2regions):
    node_shares = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):
            # we are doing nodeid in region

            node_region = len(q2regions[zone])  # how many regions this node participates in

            regions_ownership = 0  # total object ownership by all node's regions
            for region in q2regions[zone]:
                for region_zone in region:
                    regions_ownership += zone_object_ownership[region_zone] / node_region

            node_shares[nodeid][zone] = object_ownership[nodeid][zone] / regions_ownership
    return node_shares

def compute_probabilities(cols, rows, node_shares, locality, p_remote_steal):
    p_local = np.empty((rows, cols))
    p_local_forward = np.empty((rows, cols))
    p_steal = np.empty((rows, cols))
    p_remote_fwd = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):
            # we are doing nodeid in region

            node_share = node_shares[nodeid][zone]

            #  probability for a request to be forwarded
            p_local[nodeid][zone] = locality[zone][zone] * node_share
            p_local_forward[nodeid][zone] = locality[zone][zone] * (1-node_share)
            p_remote = 1 - locality[zone][zone]
            p_steal[nodeid][zone] = p_remote_steal * p_remote
            p_remote_fwd[nodeid][zone] = p_remote - p_steal[nodeid][zone]

    return p_local, p_local_forward, p_steal, p_remote_fwd


def compute_local_p2_counts(cols, rows, q2regions, node_shares, locality, client_contact_probability, R):
    p2_counts = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):

            node_r = client_contact_probability[nodeid][zone] * R * locality[zone][zone]

            for region in q2regions[zone]:
                for region_zone in region:
                    for rnid in range(0, rows):
                        #print node_shares[rnid][region_zone]
                        p2_counts[rnid][region_zone] += node_r * node_shares[rnid][region_zone]

    #print "p2_counts:"
    #print p2_counts
    return p2_counts


def compute_remote_p2_counts(cols, rows, node_shares, locality, ccp, R):
    p2_counts = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):

            node_r = ccp[nodeid][zone] * R
            # print node_r

            for fwd_zone in range(0, cols):
                if fwd_zone != zone:
                    for fwd_nodeid in range(0, rows):
                        #print node_shares[rnid][region_zone]
                        p2_counts[fwd_nodeid][fwd_zone] += node_r * node_shares[fwd_nodeid][fwd_zone] * locality[zone][fwd_zone]

    # print "remote p2_counts:"
    # print p2_counts
    return p2_counts


def compute_queue_wait(cols, rws, q1s, q2regions, p_local_at_node, p_local_forward, p_remote_fwd, p_steal,
                       mu_md_s, sigma_md_s, mu_ms_s, sigma_ms_s, mu_round, sigma_r_s, n_p):

    wait_queue = np.empty((rws, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rws):

            region_size = len(q2regions[zone][0]) * rws

            p_l = p_local_at_node[nodeid][zone]
            p_l_fwd = p_local_forward[nodeid][zone]
            p_r_fwd = p_remote_fwd[nodeid][zone]
            p_s = p_steal[nodeid][zone]
            '''
            print "p local at node = " + str(p_l)
            print "p local forward = " + str(p_l_fwd)
            print "p remote fwd = " + str(p_r_fwd)
            print "p steal = " + str(p_s)
            '''
            # so, we always have a phase 2 for every operation: region_size deserialization + 2 serializations
            # but in p_local_forward ratio of requests we forward: 1 msg serialization/deserialization
            # same for remote forward
            # and finally, p_steal of the times we run phase 1, which has 1 serialization and rows*cols deserializations
            num_deserializations = region_size + p_l_fwd * 1 + p_r_fwd * 1 + p_s * q1s
            num_serializations = 2 + p_l_fwd * 1 + p_r_fwd * 1 + p_s * 1

            # however, we also receive FWD requests that add to the queue
            # compute how many fwd requests this node can get
            # number of local forwards TO this node
            mu_r_s = mu_round[nodeid][zone]
            wait_queue[nodeid][zone] = model.marchal_mean_queue_wait_time(num_d=num_deserializations, num_s=num_serializations,
                                                            mu_md_s=mu_md_s, sigma_md_s=sigma_md_s, mu_ms_s=mu_ms_s,
                                                            sigma_ms_s=sigma_ms_s, n_p=n_p, mu_r_s=mu_r_s,
                                                            sigma_r_s=sigma_r_s)
            '''
            print "mu_r_s = " + str(mu_r_s)
            print "num_deserializations = " + str(num_deserializations)
            print "num_serializations = " + str(num_serializations)
            print "wait_queue = " + str(wait_queue[nodeid][zone])
            print "----"
            '''

    return wait_queue


def compute_local_phase2(cols, rows, mu_local_s, sigma_local_s, mu_ms_s, sigma_ms_s, mu_md_s, sigma_md_s,
                         mu_remote_s, sigma_remote_s, qw, q2regions, q2nodes_per_column):
    L_local_at_node = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):
            # now lets calculate the latencies for different modes

            # local operation, no forwarding.
            r_q_max = 0
            for region in q2regions[zone]:
                for region_zone in region:
                    if region_zone == zone:
                        rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)
                        qwc = qw[:,region_zone]
                        t = model.approx_k_order_stat_wpaxos_zone(mu_local_s + mu_ms_s + mu_md_s, rtt_sigma_s, qwc, q2nodes_per_column-1, rows, 200, nodeid)
                        if t > r_q_max:
                            r_q_max = t
                    else:
                        rtt_sigma_s = math.sqrt(sigma_remote_s[zone][region_zone] ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)
                        qwc = qw[:,region_zone]
                        t = model.approx_k_order_stat_wpaxos_zone(mu_remote_s[zone][region_zone] + mu_ms_s + mu_md_s, rtt_sigma_s, qwc, q2nodes_per_column, rows, 200)
                        if t > r_q_max:
                            r_q_max = t

            L_local_at_node[nodeid][zone] = mu_ms_s + r_q_max + qw[nodeid][zone] + mu_md_s
            # client communication
            L_local_at_node[nodeid][zone] += mu_local_s + mu_ms_s + mu_ms_s

    return L_local_at_node


def compute_local_fwd_phase2(cols, rows, mu_local_s, mu_remote_s, q2regions, L_local, node_shares):
    L_local_fwd = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):

            sum_share = 0
            for region in q2regions[zone]:
                for region_zone in region:
                    for rnid in range(0, rows):
                        if region_zone != zone or rnid != nodeid:
                            # print node_shares[rnid][region_zone]
                            sum_share += node_shares[rnid][region_zone]

            # print "sum_share:"
            # print sum_share

            for region in q2regions[zone]:
                for region_zone in region:
                    for rnid in range(0, rows):
                        if region_zone != zone or rnid != nodeid:
                            ratio_fwd_to_node = node_shares[rnid][region_zone] / sum_share
                            # print ratio_fwd_to_node
                            if region_zone == zone:
                                rtt = mu_local_s
                            else:
                                rtt = mu_remote_s.item((zone, region_zone))
                            L_local_fwd[nodeid][zone] += ratio_fwd_to_node * (L_local[rnid][region_zone] + rtt)

    return L_local_fwd


def compute_remote_fwd_phase2(cols, rows, mu_remote_s, q2regions, L_local, node_shares):
    L_remote_fwd = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):

            sum_share = 0
            for zn in range(0, cols):
                if zn != zone:
                    for rnid in range(0, rows):
                        sum_share += node_shares[rnid][zn]

            # print sum_share
            for zn in range(0, cols):
                if zn != zone:
                    for rnid in range(0, rows):
                        ratio_fwd_to_node = node_shares[rnid][zn] / sum_share
                        rtt = mu_remote_s.item((zone, zn))
                        # print L_local[rnid][zn] + rtt
                        L_remote_fwd[nodeid][zone] += ratio_fwd_to_node * (L_local[rnid][zn] + rtt)

    return L_remote_fwd


def compute_remote_steal(cols, rows, mu_remote_s, L_local):
    L_steal = np.empty((rows, cols))
    for zone in range(0, cols):
        for nodeid in range(0, rows):

            # find max remote latency
            r_lat_max = 0
            zone_dist = mu_remote_s[zone]
            for zn in range(0, cols):
                 if zone_dist.item(zn) > r_lat_max:
                     r_lat_max = zone_dist.item(zn)


            L_steal[nodeid][zone] = L_local[nodeid][zone] + r_lat_max

    return L_steal

''' Model '''
def model_random_round_arrival(cols, rows, q1nodes_per_column, q1s, q2regions, q2nodes_per_column, mu_local,
                               sigma_local, mu_remote, sigma_remote, mu_ms, sigma_ms, mu_md, object_ownership,
                               sigma_md, n_p, mu_r, sigma_r, client_contact_probability, locality, p_remote_steal):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000
    mu_remote_s = mu_remote / 1000
    sigma_remote_s = sigma_remote / 1000
    mu_r_s = mu_r / 1000
    sigma_r_s = sigma_r / 1000

    R = 1.0 / mu_r_s

    zone_object_ownership = np.sum(object_ownership, axis=0)
    #print zone_object_ownership

    node_shares = compute_node_shares(cols, rows, object_ownership, zone_object_ownership, q2regions)
    #print "node shares:"
    #print node_shares
    p_local_at_node, p_local_forward, p_steal, p_remote_fwd = compute_probabilities(cols, rows, node_shares, locality, p_remote_steal)


    p2_counts_local = compute_local_p2_counts(cols, rows, q2regions, node_shares, locality, client_contact_probability, R)
    p2_counts_remote = compute_remote_p2_counts(cols, rows, node_shares, locality, client_contact_probability, R)

    time.sleep(0.1)


    p2_counts = p2_counts_local + p2_counts_remote


    mu_round = 1.0 / p2_counts
    #print "mu round interval:"
    #print mu_round

    qw = compute_queue_wait(cols, rows, q1s, q2regions, p_local_at_node, p_local_forward, p_remote_fwd, p_steal,
                            mu_md_s, sigma_md_s, mu_ms_s, sigma_ms_s, mu_round, sigma_r_s, n_p)

    #print "queue wait:"
    #print qw

    l_loc = compute_local_phase2(cols, rows, mu_local_s, sigma_local_s, mu_ms_s, sigma_ms_s, mu_md_s,
                                           sigma_md_s, mu_remote_s, sigma_remote_s, qw, q2regions, q2nodes_per_column)

    #print "l_loc:"
    #print l_loc

    L_local_fwd = compute_local_fwd_phase2(cols, rows, mu_local_s, mu_remote_s, q2regions, l_loc, node_shares)

    #print "L_local_fwd:"
    #print L_local_fwd

    L_remote_fwd = compute_remote_fwd_phase2(cols, rows, mu_remote_s, q2regions, l_loc, node_shares)

    #print "L_remote_fwd:"
    #print L_remote_fwd

    L_steal = compute_remote_steal(cols, rows, mu_remote_s, l_loc)

    #print "L_steal:"
    #print L_steal
    print p_local_at_node
    L_local_ops = p_local_at_node * l_loc + p_local_forward * L_local_fwd
    L_remote_ops = p_remote_fwd * L_remote_fwd + p_steal * L_steal
    L_average_round = L_local_ops + L_remote_ops

    #print "L_average_round:"
    #print L_average_round

    return L_local_ops, L_remote_ops, L_average_round

'''
model_random_round_arrival(
    rws=rows,
    cols=columns,
    q2regions=q2regions,
    q1nodes_per_column=q1nodes_per_column,
    q1s=q1s,
    q2nodes_per_column=q2nodes_per_column,
    mu_local=mu_local,
    sigma_local=sigma_local,
    mu_ms=mu_ms,
    sigma_ms=sigma_ms,
    mu_md=mu_md,
    object_ownership=object_ownership,
    sigma_md=sigma_md,
    n_p=n_p,
    mu_r=mu_r,
    sigma_r=mu_r,
    mu_remote=mu_remote,
    sigma_remote=sigma_remote,
    client_contact_probability=client_contact_probability,
    locality=locality,
    p_remote_steal=p_remote_steal
)

tp = numops / float(t)
print "# of operations: " + str(numops)
print "TP: " + str(tp)
av_lat_s = np.average(simlats)
av_lat = av_lat_s * 1000
print "Average latency: " + str(av_lat) + " ms"
'''