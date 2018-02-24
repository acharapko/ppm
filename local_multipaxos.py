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
        ms = np.random.normal(mu_ms_s, sigma_ms_s)  # time to serialize ith message
        md = np.random.normal(mu_md_s, sigma_md_s)  # time to deserialize ith message

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
                c_msg = max(c_msg + ms/n_p + pipeline[i-1][1] - pipeline[i][1], 0)
            else:
                c_msg = max(c_msg + md/n_p + pipeline[i-1][1] - pipeline[i][1], 0)

        if not msg_serialization and not msg_from_client and msgs_processed[pipeline[i][0]] == qs - 1:
            # we have enough messages from round pipeline[i][0] to proceed
            Lr = ms + pipeline[i][2] + c_msg + md  # T_r = m_s + r_{lq-1} + c_{lq-1} + m_d
            if sim_clients:
                # if we sim client, then add client communication latency.
                # network round trip time to receive msg from client and reply back
                client_rtt = np.random.normal(mu_local_s, sigma_local_s)
                Lr = Lr + client_rtt
            lats.append(Lr)

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