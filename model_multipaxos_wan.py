import numpy as np
import model
import math

# Third version of the model

''' MODEL '''
# this function uses queuing theory model to approximate Paxos execution time. It uses Monte Carlo method to get
# expecting networking delays for different quorums. Network delay is the k-order statistic for Normal sample of N-1
# message RTT times. In classical quorums, this is very close to mean RTT, however for some flexible quorums
# network variability may have larger effects.


def model_random_round_arrival(mu_nodes, sigma_nodes, mu_local, leader_id, qs, mu_ms, sigma_ms, mu_md, sigma_md, mu_r, n_p, sigma_r, sim_clients=False):
    # convert everything to seconds
    N = len(mu_nodes)
    mu_local_s = mu_local / 1000
    mu_nodes_s = mu_nodes / 1000
    sigma_nodes_s = sigma_nodes / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000
    mu_r_s = mu_r / 1000
    sigma_r_s = sigma_r / 1000

    R = 1000.0/mu_r

    #rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)
    num_serialize = N
    num_deserialize = 2
    if not sim_clients:
        num_serialize -= 1
        num_deserialize -= 1

    wait_queue = model.marchal_mean_queue_wait_time(num_d=num_serialize, num_s=num_deserialize, mu_md_s=mu_md_s,
                                                    sigma_md_s=sigma_md_s, mu_ms_s=mu_ms_s, sigma_ms_s=sigma_ms_s,
                                                    n_p=n_p, mu_r_s=mu_r_s, sigma_r_s=sigma_r_s)

    r_q1 = model.approx_k_order_stat_paxos_wan(mu_nodes_s, sigma_nodes_s, qs-1, N, leader_id)

    Lr = mu_ms_s + r_q1 + wait_queue + mu_md_s  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
    if sim_clients:
        Lr += mu_local_s + mu_ms_s + mu_ms_s

    return R, Lr
