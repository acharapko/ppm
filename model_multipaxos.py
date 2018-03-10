import numpy as np
import model
import math

# Third version of the model

''' SIMULATION '''
# This function simulates multi-paxos every round and the pipeline at the round leader.
# It only accounts for phase-2 repeats


def sim(t, N, qs, mu_local, sigma_local, mu_ms, sigma_ms, mu_md, sigma_md, n_p, mu_r, sigma_r, sim_clients=False):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000
    mu_r_s = mu_r / 1000
    sigma_r_s = sigma_r / 1000

    Rmax = n_p / (N * mu_md + 2 * mu_ms) * 1000

    rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)

    # first fill the "pipeline"
    t_round = 0
    ops = 0
    pipeline = []
    while t_round < t:
        #print t_round
        delta_t_round = np.random.normal(mu_r_s, sigma_r_s)
        rtts = np.random.normal(mu_local_s + mu_ms_s + mu_md_s, rtt_sigma_s, N - 1)
        rtts.sort()
        # this simulates the leader pipeline. it consists of a single serialization messages
        # and N-1 deserialization messages

        if sim_clients:
            pipeline.append([ops, rtts[0] + t_round - mu_ms - mu_md, -2])  # msg from client
            pipeline.append([ops, rtts[qs-1] + t_round + mu_md, -2])  # reply to client must come after l_{q-1]

        # serialization message:
        pipeline.append([ops, rtts[0] + t_round - mu_ms, -1])
        # deserialization messages:
        for i in range(0, len(rtts)):
            pipeline.append([ops, rtts[i] + t_round, rtts[i]])  # [round#, msgTime, msgRTT]
        ops += 1

        t_round += delta_t_round
    pipeline.sort(key=lambda x: x[1])
    # print pipeline

    # now compute round latencies
    lats = []
    msgs_processed = {}
    c_msg = 0
    for i in range(0, len(pipeline)):
        # if not seralization messages, then these are replies, so count the quorum
        # we hav serialization messages in the pipeline only so they can impact c_msg of l_{qs-1} message
        # in other words, serialization msg adds some more load to the pipeline.
        msg_serialization = pipeline[i][2] == -1
        msg_from_client = pipeline[i][2] == -2
        if not msg_serialization and not msg_from_client:
            if pipeline[i][0] in msgs_processed:
                msgs_processed[pipeline[i][0]] += 1
            else:
                msgs_processed[pipeline[i][0]] = 1

        if i > 0:
            if msg_serialization:
                # impact of message serialization is different from deserialization
                c_msg = max(c_msg + mu_ms_s/n_p + pipeline[i-1][1] - pipeline[i][1], 0)
            else:
                c_msg = max(c_msg + mu_md_s/n_p + pipeline[i-1][1] - pipeline[i][1], 0)

        if not msg_serialization and not msg_from_client and msgs_processed[pipeline[i][0]] == qs - 1:
            # we have enough messages from round pipeline[i][0] to proceed
            Lr = mu_ms_s + pipeline[i][2] + c_msg + mu_md_s  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
            if sim_clients:
                # if we sim client, then add client communication latency.
                # network round trip time to receive msg from client and reply back
                Lr += mu_local_s + mu_ms_s + mu_ms_s
            lats.append(Lr)

    return ops, lats


''' MODEL '''
# this function uses queuing theory model to approximate Paxos execution time. It uses Monte Carlo method to get
# expecting networking delays for different quorums. Network delay is the k-order statistic for Normal sample of N-1
# message RTT times. In classical quorums, this is very close to mean RTT, however for some flexible quorums
# network variability may have larger effects.


def model_random_round_arrival(N, qs, mu_local, sigma_local, mu_ms, sigma_ms, mu_md, sigma_md, mu_r, n_p, sigma_r, sim_clients=False):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000
    mu_r_s = mu_r / 1000
    sigma_r_s = sigma_r / 1000

    R = 1000.0/mu_r

    rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)
    num_serialize = N
    num_deserialize = 2
    if not sim_clients:
        num_serialize -= 1
        num_deserialize -= 1

    wait_queue = model.marchal_mean_queue_wait_time(num_d=num_serialize, num_s=num_deserialize, mu_md_s=mu_md_s,
                                                    sigma_md_s=sigma_md_s, mu_ms_s=mu_ms_s, sigma_ms_s=sigma_ms_s,
                                                    n_p=n_p, mu_r_s=mu_r_s, sigma_r_s=sigma_r_s)

    r_q1 = model.approx_k_order_stat(mu_local_s + mu_ms_s + mu_md_s, rtt_sigma_s, qs - 1, N - 1)

    Lr = mu_ms_s + r_q1 + wait_queue + mu_md_s  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
    if sim_clients:
        Lr += mu_local_s + mu_ms_s + mu_ms_s

    return R, Lr
