import numpy as np

# REFERENCE SAMPLE VALUES FOR WPAXOS PARAMETERS '''

locality_filename = "params/zone_locality.csv"
mu_net_filename = "params/mu_net_remote.csv"
mu_net_std_dev_filename = "params/sigma_net_remote.csv"

# set the grid
rows = 3

p_remote_steal = 0.01

mu_local = 0.427  # network RTT mean in ms
sigma_local = 0.0476  # network RTT sigma in ms

mu_ms = 0.001  # message serialization overhead in ms
sigma_ms = 0.005

mu_md = 0.025  # message deserialization overhead in ms
sigma_md = 0.015

n_p = 1  # number of pipelines

R = 16000  # Throughput in rounds/sec for each region
mu_r = 1000.0 / R
sigma_r = mu_r / 0.5  # give it some good round spread

# regions form a graph with edges being communication links, and weights being mean communication latencies
mu_remote = np.loadtxt(mu_net_filename, delimiter=",")
print mu_remote
# and weights being std. deviations as well
sigma_remote = np.loadtxt(mu_net_std_dev_filename, delimiter=",")

locality = np.loadtxt(locality_filename, delimiter=",").tolist()

columns = len(mu_remote)  # columns is number of regions

# quorum sizes
q1nodes_per_column = 2
q1s = 1 * q1nodes_per_column * columns  # q1 size: rows x nodes/per column * columns

q2regions = [[[0]], [[1]], [[2]], [[3]], [[4]]]
q2nodes_per_column = q1nodes_per_column + 1

# probability of client reaching out to a node. row-region, column-nodeID
client_contact_probability = np.full((rows, columns), 1.0 / rows, dtype=float)

# Object ownership
object_ownership = np.full((rows, columns), 1.0 / (rows*columns), dtype=float)
