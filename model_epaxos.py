import numpy as np
import model

# Modeling EPaxos


def get_msg_stats(N, conflict_rate, mu_ms, mu_md, n_p, R, sim_clients):

    Rmax = []
    num_serialize = []
    num_deserialize = []
    for n in range(0, N):
        rounds_as_follower_per_leader_round = (np.sum(R) - R[n]) / R[n]
        ns = 1 + conflict_rate * 1 + rounds_as_follower_per_leader_round
        nd = (N-1) + conflict_rate * (N-1) + rounds_as_follower_per_leader_round

        if sim_clients:
            ns += 1
            nd += 1

        num_deserialize.append(nd)
        num_serialize.append(ns)
        Rmax.append(n_p / (mu_md * nd + mu_ms * ns) * 1000)  # to req/sec
    return num_serialize, num_deserialize


def get_max_throughput(N, conflict_rate, mu_ms, mu_md, n_p, sim_clients):

    Rmax = []
    for n in range(0, N):
        rounds_as_follower_per_leader_round = N - 1
        ns = 1 + conflict_rate * 1 + rounds_as_follower_per_leader_round
        nd = (N-1) + conflict_rate * (N-1) + rounds_as_follower_per_leader_round

        if sim_clients:
            ns += 1
            nd += 1

        Rmax.append(n_p / (mu_md * nd + mu_ms * ns) * 1000)  # to req/sec
    return Rmax


def model_random_round_arrival(mu_nodes, sigma_nodes, mu_local, qs, fqs, conflict_rate, mu_ms, sigma_ms, mu_md, sigma_md, mu_r, sigma_r, n_p, sim_clients=False):
    # convert everything to seconds
    N = len(mu_nodes)
    mu_local_s = mu_local / 1000
    mu_nodes_s = mu_nodes / 1000
    sigma_nodes_s = sigma_nodes / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000
    mu_r_s = np.array(mu_r) / 1000
    sigma_r_s = np.array(sigma_r) / 1000


    Lr = []

    R = 1.0/mu_r_s  # number of ops in each region

    # on average we have 1-conflict_rate fast quorum rounds and conflict rate full Paxos rounds
    # for fast quorum round we have a message from a round leader to all nodes except itself and N-1 messages
    # coming back to the round leader
    # for full Paxos round we have twice the messages (another one serialization and N-1 deserialization).
    # Since this is a multi-leader protocol, we also account for messages each node process in non-leader role
    # this will be 1 serialization and 1 deserialization for each round
    # or 2 serailiazations and 2 deserializations for full paxos round

    num_serialize, num_deserialize = get_msg_stats(N, conflict_rate, mu_ms, mu_md, n_p, R, sim_clients)

    for n in range(0, N):

        wait_queue = model.marchal_mean_queue_wait_time(num_d=num_deserialize[n], num_s=num_serialize[n], mu_md_s=mu_md_s,
                                                        sigma_md_s=sigma_md_s, mu_ms_s=mu_ms_s, sigma_ms_s=sigma_ms_s,
                                                        n_p=n_p, mu_r_s=mu_r_s[n], sigma_r_s=sigma_r_s[n])

        r_fast_q = model.approx_k_order_stat_paxos_wan(mu_nodes_s, sigma_nodes_s, fqs-1, N, n)

        r_slow_q = model.approx_k_order_stat_paxos_wan(mu_nodes_s, sigma_nodes_s, qs-1, N, n)

        L = mu_ms_s + r_fast_q + (conflict_rate * r_slow_q) + wait_queue + mu_md_s
        if sim_clients:
            L += mu_local_s + mu_ms_s + mu_ms_s

        Lr.append(L)

    return R, Lr

