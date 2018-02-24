# PPM - Paxos Performance Model

PPM implements paxos modeling as described in my blog: http://charap.co/do-not-blame-only-network-for-your-paxos-scalability/

Some important files:
 
* local_multipaxos.py - multipaxos model. Contains default model parameters.
* sim_local_multipaxos.py - runs the multipaxos model. 
* sim_graph.py - plots the throughput vs latency graph for different cluster sizes.

------

## Model Parameters ##
sim_local_multipaxos.py can take various parameters to override the default settings:
* -t - time to simulate. By default, 60 seconds of paxos runtime is modeled
* -N - number of nodes in the cluster. Default: 3
* -q - Quorum size. By default majority quorum for given cluster size N. Always keep at majority, unless modeling flexible quorums
* -r - mean Network RTT time in ms. Default: 0.427 ms
* -R - standard deviation of netwrok RTT time. Default: 0.0476 ms
* -s - mean of message serialization overhead
* -S - standard deviation of message serialization overhead
* -d - mean of message deserialization/processing overhead
* -D - standard deviation of message deserialization overhead
* -p - number of processing pipelines. This is roughly the number of cores at the node. Default: 1
* -z - mean separation between rounds. This controls the target throughput. Default: 1/6000
* -Z - standard deviation of separation between rounds. Default: 1/3000
