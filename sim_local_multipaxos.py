import local_multipaxos as sim
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='MultiPaxos Performance Model')

parser.add_argument('-t', action="store", default=sim.t, type=int)
parser.add_argument('-N', action="store", default=sim.N, type=int)
parser.add_argument('-q', action="store", type=int, default=sim.qs)
parser.add_argument('-r', dest="mu_net", action="store", type=float, default=sim.mu_local)
parser.add_argument('-R', dest="sigma_net", action="store", type=float, default=sim.sigma_local)

parser.add_argument('-s', dest="mu_ms", action="store", type=float, default=sim.mu_ms)
parser.add_argument('-S', dest="sigma_ms", action="store", type=float, default=sim.sigma_ms)

parser.add_argument('-d', dest="mu_md", action="store", type=float, default=sim.mu_md)
parser.add_argument('-D', dest="sigma_md", action="store", type=float, default=sim.sigma_md)

parser.add_argument('-p', action="store", type=int, default=1)

parser.add_argument('-z', dest="mu_r", action="store", type=float, default=sim.mu_r)
parser.add_argument('-Z', dest="sigma_r", action="store", type=float, default=sim.sigma_r)

args = parser.parse_args()

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
    sigma_r=args.mu_r,
    sim_clients=True
)

tp = numops / float(args.t)
print "# of operations: " + str(numops)
print "TP: " + str(tp)
av_lat_s = np.average(simlats)
av_lat = av_lat_s * 1000
print "Average latency: " + str(av_lat) + " ms"
