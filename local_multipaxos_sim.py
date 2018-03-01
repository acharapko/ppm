import numpy as np

# First version of the model: http://charap.co/do-not-blame-only-network-for-your-paxos-scalability/

t = 60  # sim duration in seconds
N = 4  # number of nodes in the cluster

qs = N / 2 + 1  # quorum size. For Paxos it is majority

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

ms = 0.01  # message serialization/deserialization overhead in ms

# convert everything to seconds
mu_local_s = mu_local / 1000
sigma_local_s = sigma_local / 1000
ms_s = ms / 1000


elapsed_t = 0
ops = 0
lats = []
# simulate rounds
while elapsed_t < t:
    rtts = np.random.normal(mu_local_s+2*ms_s, sigma_local_s, N-1)
    rtts.sort()
    #print rtts
    # qs is majority, but we assume self reply from leader so we wait for qs-1 messages
    R = rtts[qs-2] + 2 * ms_s
    elapsed_t += R
    ops += 1
    lats.append(R)


tp = ops / float(t)
print "# of operations: " + str(ops)
print "TP: " + str(tp)
av_lat_s = np.average(lats)
av_lat = av_lat_s * 1000
print "Average latency: " + str(av_lat) + " ms"
