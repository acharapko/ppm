import numpy as np
import math

# Second version of the model: http://charap.co/do-not-blame-only-network-for-your-paxos-scalability/

t = 60  # sim duration in seconds
N = 11  # number of nodes in the cluster

qs = N / 2 + 1  # quorum size. For Paxos it is majority

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

mu_ms = 0.01  # message serialization/deserialization overhead in ms
sigma_ms = 0.002
# convert everything to seconds
mu_local_s = mu_local / 1000
sigma_local_s = sigma_local / 1000
mu_ms_s = mu_ms / 1000
sigma_ms_s = sigma_ms / 1000

rtt_sigma_s = math.sqrt(sigma_local_s ** 2 + sigma_ms_s ** 2 + sigma_ms_s ** 2)

elapsed_t = 0
ops = 0
lats = []
# simulate rounds
while elapsed_t < t:
    #print elapsed_t
    rtts = np.random.normal(mu_local_s + 2 * mu_ms_s, rtt_sigma_s, N - 1)
    rtts.sort()
    #print rtts
    # qs is majority, but we assume self reply from leader so we wait for qs-1 messages
    ms = np.random.normal(mu_ms_s, sigma_ms_s, 2)
    R = rtts[qs-2] + ms[0] + ms[1]
    #print R
    elapsed_t += R
    ops += 1
    lats.append(R)


tp = ops / float(t)
print "# of operations: " + str(ops)
print "TP: " + str(tp)
av_lat_s = np.average(lats)
av_lat = av_lat_s * 1000
print "Average latency: " + str(av_lat) + " ms"
