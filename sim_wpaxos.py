import wpaxos_defaults as params
import model_wpaxos as model
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='WPaxos Performance Model')
# the grid
parser.add_argument('-r', dest="rows", action="store", default=params.rows, type=int)

parser.add_argument('-m', dest="mu_net_local", action="store", type=float, default=params.mu_local)
parser.add_argument('-M', dest="sigma_net_local", action="store", type=float, default=params.sigma_local)

parser.add_argument('-c', dest="ccp_filename", action="store", type=str, default="")
parser.add_argument('-o', dest="oo_filename", action="store", type=str, default="")
parser.add_argument('-l', dest="locality_filename", action="store", type=str, default="")
parser.add_argument('-n', dest="mu_net_filename", action="store", type=str, default="")
parser.add_argument('-v', dest="mu_net_std_dev_filename", action="store", type=str, default="")

parser.add_argument('-s', dest="mu_ms", action="store", type=float, default=params.mu_ms)
parser.add_argument('-S', dest="sigma_ms", action="store", type=float, default=params.sigma_ms)

parser.add_argument('-d', dest="mu_md", action="store", type=float, default=params.mu_md)
parser.add_argument('-D', dest="sigma_md", action="store", type=float, default=params.sigma_md)

parser.add_argument('-p', action="store", type=int, default=1)

parser.add_argument('-z', dest="mu_r", action="store", type=float, default=params.mu_r)
parser.add_argument('-Z', dest="sigma_r", action="store", type=float, default=params.sigma_r)

args = parser.parse_args()

# regions form a graph with edges being communication links, and weights being mean communication latencies
if args.mu_net_filename != "":
    params.mu_remote = np.loadtxt(args.mu_net_filename, delimiter=",")
# and weights being std. deviations as well
if args.mu_net_std_dev_filename != "":
    params.sigma_remote = np.loadtxt(args.mu_net_std_dev_filename, delimiter=",")

if args.locality_filename != "":
    params.locality = np.loadtxt(args.locality_filename, delimiter=",").tolist()

columns = len(params.mu_remote)  # columns is number of regions

# probability of client reaching out to a node. row-region, column-nodeID
if args.ccp_filename != "":
    params.client_contact_probability = np.loadtxt(args.ccp_filename, delimiter=",")

# Object ownership
if args.oo_filename != "":
    object_ownership = np.loadtxt(args.oo_filename, delimiter=",")

print '{0: >48}'.format('Model Parameters')
print '{0:-<80s}'.format('')

print '{0:<40s}{1:<3d}x{2:>3d} nodes'.format('Cluster size:', columns, params.rows)
#print '{0:<40s}{1:10d} nodes'.format('Q2 size:', args.q)
print '{0:<40s}{1:10d} workers'.format('Number of pipeline workers:', args.p)
print '{0:<40s}{1:10.3f} ms'.format('Round arrival time:', args.mu_r)
print '{0:<40s}{1:10.3f} ms'.format('Round arrival std. deviation:', args.sigma_r)
print '{0:<40s}{1:10.3f} ms'.format('Mean local network RTT:', args.mu_net_local)
print '{0:<40s}{1:10.3f} ms'.format('local network RTT std. deviation:', args.sigma_net_local)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg deserialization time:', args.mu_md)
print '{0:<40s}{1:10.3f} ms'.format('Msg deserialization std. deviation:', args.sigma_md)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg serialization time:', args.mu_ms)
print '{0:<40s}{1:10.3f} ms'.format('Msg serialization std. deviation:', args.sigma_ms)
print '{0:-<80s}'.format('')

L_local_ops, L_remote_ops, L_average_round = model.model_random_round_arrival(
    rows=args.rows,
    cols=columns,
    q2regions=params.q2regions,
    q1nodes_per_column=params.q1nodes_per_column,
    q1s=params.q1s,
    q2nodes_per_column=params.q2nodes_per_column,
    mu_local=args.mu_net_local,
    sigma_local=args.sigma_net_local,
    mu_ms=args.mu_ms,
    sigma_ms=args.sigma_ms,
    mu_md=args.mu_md,
    object_ownership=params.object_ownership,
    sigma_md=args.sigma_md,
    n_p=args.p,
    mu_r=args.mu_r,
    sigma_r=args.mu_r,
    mu_remote=params.mu_remote,
    sigma_remote=params.sigma_remote,
    client_contact_probability=params.client_contact_probability,
    locality=params.locality,
    p_remote_steal=params.p_remote_steal
)


L_region = np.sum(L_average_round, axis=0)
L_region /= args.rows


print L_region