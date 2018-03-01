import numpy as np
import math

# Third version of the model

''' REFERENCE SAMPLE VALUES FOR SIMULATION PARAMETERS '''

t = 60  # sim duration in seconds
N = 3  # number of nodes in the cluster

qs = N / 2 + 1  # quorum size. For Paxos it is majority

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

mu_ms = 0.001  # message serialization overhead in ms
sigma_ms = 0.005

mu_md = 0.025  # message deserialization overhead in ms
sigma_md = 0.015

n_p = 1  # number of pipelines

R = 6000  # Throughput in rounds/sec
mu_r = 1.0 / R
sigma_r = mu_r / 0.5  # give it some good round spread

''' SIMULATION '''
def sim(t, N, qs, mu_local, sigma_local, mu_ms, sigma_ms, mu_md, sigma_md, n_p, mu_r, sigma_r, sim_clients=False):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000

    Rmax = n_p / (N * mu_md + 2 * mu_ms) * 1000

    rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)

    # first fill the "pipeline"
    t_round = 0
    ops = 0
    pipeline = []
    while t_round < t:
        #print t_round
        delta_t_round = np.random.normal(mu_r, sigma_r)
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
                Lr = Lr + mu_local_s
            lats.append(Lr)

    return ops, lats

def model_random_round_arrival(t, N, qs, mu_local, sigma_local, mu_ms, sigma_ms, mu_md, sigma_md, mu_r, sigma_r, sim_clients=False):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000

    rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)

    t_round = 0
    ops = 0

    R = (1 / mu_r)
    C_a = (sigma_r ** 2) / (mu_r ** 2)
    lmda = R  # mean rate of arrival in rounds per second
    mu_sr = 1 / (N * mu_md_s + 2 * mu_ms_s)  # mean rate of service (speed of the pipeline). essentially max throughput
    p_queue = R * (N * mu_md_s + 2 * mu_ms_s)  # average queue load (prob queue is empty) lmda/mu_sr

    mu_st = (N * mu_md_s + 2 * mu_ms_s)
    var_st = (N * (sigma_md_s ** 2) + 2 * (sigma_ms_s ** 2))
    C_st = var_st / mu_st ** 2
    sigma_st = math.sqrt(var_st)

    # Marchal's approximation for G/G/1 queue
    L_q = (p_queue**2*(1+C_st)*(C_a+C_st*p_queue**2))/(2*(1-p_queue)*(1+C_st*p_queue**2))
    wait_queue = L_q / lmda

    print wait_queue

    lats = []
    while t_round < t:
        delta_t_round = np.random.normal(mu_r, sigma_r)
        rtts = np.random.normal(mu_local_s + mu_ms_s + mu_md_s, rtt_sigma_s, N - 1)
        rtts.sort()

        Lr = mu_ms_s + rtts[qs-2] + wait_queue + mu_md_s  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
        if sim_clients:
            Lr = Lr + mu_local_s
        lats.append(Lr)

        ops += 1
        t_round += delta_t_round

    return ops, lats

def model_poisson_round_arrival(t, N, qs, mu_local, sigma_local, mu_ms, sigma_ms, mu_md, sigma_md, mu_r, sigma_r, sim_clients=False):
    # convert everything to seconds
    mu_local_s = mu_local / 1000
    sigma_local_s = sigma_local / 1000
    mu_ms_s = mu_ms / 1000
    mu_md_s = mu_md / 1000
    sigma_ms_s = sigma_ms / 1000
    sigma_md_s = sigma_md / 1000

    rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_md_s ** 2)

    t_round = 0
    ops = 0

    R = (1 / mu_r)

    lmda = R  # mean rate of arrival in rounds per second
    mu_sr = 1 / (N * mu_md_s + 2 * mu_ms_s)  # mean rate of service (speed of the pipeline). essentially max throughput

    p_queue = R * (N * mu_md_s + 2 * mu_ms_s)  # average queue load (prob queue is empty) lmda/mu_sr
    # mu_st = (N * mu_md_s + 2 * mu_ms_s) / (N + 2)
    var_st = (N * (sigma_md_s ** 2) + 2 * (sigma_ms_s ** 2))
    sigma_st = math.sqrt(var_st)

    L_q = (lmda ** 2 * sigma_st ** 2 + p_queue**2) / (2*(1 - p_queue))  # from Pollaczek-Khinchine formula
    wait_queue = L_q / lmda

    lats = []
    while t_round < t:
        delta_t_round = np.random.normal(mu_r, sigma_r)
        rtts = np.random.normal(mu_local_s + mu_ms_s + mu_md_s, rtt_sigma_s, N - 1)
        rtts.sort()

        Lr = mu_ms_s + rtts[qs-2] + wait_queue + mu_md_s  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
        if sim_clients:
            Lr = Lr + mu_local_s
        lats.append(Lr)

        ops += 1
        t_round += delta_t_round

    return ops, lats

'''
numops, simlats = sim(
    t=t,
    N=N,
    qs=qs,
    mu_local=mu_local,
    sigma_local=sigma_local,
    mu_ms=mu_ms,
    sigma_ms=sigma_ms,
    mu_md=mu_md,
    sigma_md=sigma_md,
    n_p=n_p,
    mu_r=mu_r,
    sigma_r=mu_r,
    sim_clients=True
)

tp = numops / float(t)
print "# of operations: " + str(numops)
print "TP: " + str(tp)
av_lat_s = np.average(simlats)
av_lat = av_lat_s * 1000
print "Average latency: " + str(av_lat) + " ms"
'''