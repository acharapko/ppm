import model_multipaxos as sim
import paxos_defaults as params
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='MultiPaxos Performance Sim/Model')

parser.add_argument('-m', dest="model", action="store_true")
parser.add_argument('-c', dest="clients", action="store_true")
parser.add_argument('-t', action="store", default=params.t, type=int)
parser.add_argument('-N', action="store", default=params.N, type=int)
parser.add_argument('-q', action="store", type=int, default=params.qs)
parser.add_argument('-r', dest="mu_net", action="store", type=float, default=params.mu_local)
parser.add_argument('-R', dest="sigma_net", action="store", type=float, default=params.sigma_local)

parser.add_argument('-s', dest="mu_ms", action="store", type=float, default=params.mu_ms)
parser.add_argument('-S', dest="sigma_ms", action="store", type=float, default=params.sigma_ms)

parser.add_argument('-d', dest="mu_md", action="store", type=float, default=params.mu_md)
parser.add_argument('-D', dest="sigma_md", action="store", type=float, default=params.sigma_md)

parser.add_argument('-p', action="store", type=int, default=1)

parser.add_argument('-z', dest="mu_r", action="store", type=float, default=params.mu_r)
parser.add_argument('-Z', dest="sigma_r", action="store", type=float, default=params.sigma_r)

args = parser.parse_args()

print '{0:-<80s}'.format('')
if not args.model:
    print '{0: >50}'.format('Simulation Parameters')
    print '{0:-<80s}'.format('')
    print "Simulation time = " + str(args.t)
else:
    print '{0: >48}'.format('Model Parameters')
    print '{0:-<80s}'.format('')

print '{0:<40s}{1:10}'.format('Client communication:', args.clients)
print '{0:<40s}{1:10d} nodes'.format('Cluster size:', args.N)
print '{0:<40s}{1:10d} nodes'.format('Quorum size:', args.q)
print '{0:<40s}{1:10d} workers'.format('Number of pipeline workers:', args.p)
print '{0:<40s}{1:10.3f} ms'.format('Round arrival time:', args.mu_r)
print '{0:<40s}{1:10.3f} ms'.format('Round arrival std. deviation:', args.sigma_r)
print '{0:<40s}{1:10.3f} ms'.format('Mean network RTT:', args.mu_net)
print '{0:<40s}{1:10.3f} ms'.format('network RTT std. deviation:', args.sigma_net)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg deserialization time:', args.mu_md)
print '{0:<40s}{1:10.3f} ms'.format('Msg deserialization std. deviation:', args.sigma_md)
print '{0:<40s}{1:10.3f} ms'.format('Mean msg serialization time:', args.mu_ms)
print '{0:<40s}{1:10.3f} ms'.format('Msg serialization std. deviation:', args.sigma_ms)
print '{0:-<80s}'.format('')


if args.model:
    numops, simlats = sim.model_random_round_arrival(
        N=args.N,
        qs=args.q,
        mu_local=args.mu_net,
        sigma_local=args.sigma_net,
        mu_ms=args.mu_ms,
        sigma_ms=args.sigma_ms,
        mu_md=args.mu_md,
        sigma_md=args.sigma_md,
        ttx=params.ttx,
        ttx_stddev=params.ttx_stddev,
        n_p=args.p,
        mu_r=args.mu_r,
        sigma_r=args.sigma_r,
        sim_clients=args.clients)
else:
    numops, simlats = sim.sim(
        t=args.t,
        N=args.N,
        qs=args.q,
        mu_local=args.mu_net,
        sigma_local=args.sigma_net,
        mu_ms=args.mu_ms,
        sigma_ms=args.sigma_ms,
        mu_md=args.mu_md,
        sigma_md=args.sigma_md,
        n_p=args.p,
        mu_r=args.mu_r,
        sigma_r=args.sigma_r,
        sim_clients=args.clients
    )

tp = numops / float(args.t)

if args.model:
    print '{0: >46}'.format('Model Results')
    print '{0:-<80s}'.format('')
    print '{0:<40s}{1:10.3f} round/s'.format('Target Throughput:', numops)
    print '{0:<40s}{1:10.3f} ms'.format('Average round latency:', simlats*1000)
else:
    print '{0: >50}'.format('Simulation Results')
    print '{0:-<80s}'.format('')
    av_lat_s = np.average(simlats)
    av_lat = av_lat_s * 1000
    print '{0:<40s}{1:10.3f} rounds'.format('Rounds Simulated:', numops)
    print '{0:<40s}{1:10.3f} round/s'.format('Throughput:', tp)
    print '{0:<40s}{1:10.3f} ms'.format('Average round latency:', av_lat)
