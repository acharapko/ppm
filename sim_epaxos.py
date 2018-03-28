import epaxos_defaults as params
import model_epaxos as model
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='EPaxos Performance Model')

parser.add_argument('-m', dest="mu_net_local", action="store", type=float, default=params.mu_local)

parser.add_argument('-c', dest="conflict", action="store", type=str, default=params.conflict)

parser.add_argument('-n', dest="mu_net_filename", action="store", type=str, default="")
parser.add_argument('-v', dest="mu_net_std_dev_filename", action="store", type=str, default="")

parser.add_argument('-s', dest="mu_ms", action="store", type=float, default=params.mu_ms)
parser.add_argument('-S', dest="sigma_ms", action="store", type=float, default=params.sigma_ms)

parser.add_argument('-d', dest="mu_md", action="store", type=float, default=params.mu_md)
parser.add_argument('-D', dest="sigma_md", action="store", type=float, default=params.sigma_md)

parser.add_argument('-p', action="store", type=int, default=1)

parser.add_argument('-z', dest="mu_r", action="store", type=float, default=-1)
parser.add_argument('-Z', dest="sigma_r", action="store", type=float, default=-1)

parser.add_argument('-r', dest="mu_r_filename", action="store", type=float, default=params.mu_r)
parser.add_argument('-R', dest="sigma_r_filename", action="store", type=float, default=params.sigma_r)

args = parser.parse_args()

# regions form a graph with edges being communication links, and weights being mean communication latencies
if args.mu_net_filename != "":
    params.mu_remote = np.loadtxt(args.mu_net_filename, delimiter=",")
    params.N, params.fast_q, params.slow_q = params.calc_quorums(params.mu_remote)

# and weights being std. deviations as well
if args.mu_net_std_dev_filename != "":
    params.sigma_remote = np.loadtxt(args.mu_net_std_dev_filename, delimiter=",")

if args.mu_r != -1 and args.sigma_r != 1:
    params.mu_r = []
    params.sigma_r = []
    for r in range(0, params.N):
        params.mu_r.append(args.mu_r)
        params.sigma_r.append(args.sigma_r)  # give it some good round spread

print '{0: >48}'.format('Model Parameters')
print '{0:-<80s}'.format('')

print '{0:<40s}{1:<3d} nodes'.format('Cluster size:', params.N)
print '{0:<40s}{1:<3d} nodes'.format('Fast quorum size:', params.fast_q)
print '{0:<40s}{1:<3d} nodes'.format('Slow quorum size:', params.slow_q)
print '{0:<40s}{1:10d} workers'.format('Number of pipeline workers:', args.p)
print '{0:<40s}{1:10.3f} ms'.format('Mean local network RTT:', args.mu_net_local)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg deserialization time:', args.mu_md)
print '{0:<40s}{1:10.3f} ms'.format('Msg deserialization std. deviation:', args.sigma_md)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg serialization time:', args.mu_ms)
print '{0:<40s}{1:10.3f} ms'.format('Msg serialization std. deviation:', args.sigma_ms)
print '{0:-<80s}'.format('')

R, Lr = model.model_random_round_arrival(
    mu_nodes=params.mu_remote,
    sigma_nodes=params.sigma_remote,
    mu_local=args.mu_net_local,
    qs=params.slow_q,
    fqs=params.fast_q,
    conflict_rate=args.conflict,
    mu_ms=args.mu_ms,
    sigma_ms=args.sigma_ms,
    mu_md=args.mu_md,
    sigma_md=args.sigma_md,
    mu_r=params.mu_r,
    sigma_r=params.mu_r,
    n_p=args.p,
)

print '{0: >46}'.format('Model Results')
print '{0:-<80s}'.format('')
for n in range(0, params.N):
   print '{0: >40} {1:3d}'.format('Region', n)
   print '{0:<40s}{1:10.3f} round/s'.format('Target Throughput:', R[n])
   print '{0:<40s}{1:10.3f} ms'.format('Average round latency:', (Lr[n]*1000))
