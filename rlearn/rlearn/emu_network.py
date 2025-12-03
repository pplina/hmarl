import os
import sys

from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
import network as network
import networkx as nx

import network as network

net = None


def create(topology):

    net = Mininet(topo=None, build=False)
    hostcounter = 0
    #c0 = net.addController(name="c0", controller=Controller, protocol="tcp", port=6633, ip="127.0.255.255")
    for key in topology:
        splitkey = key.split("_")
        if "s" in key:
            net.addSwitch(key, ip="127.0.%d.%d" % (int(splitkey[0][1:]), int(splitkey[1])))
            continue
        net.addHost(key, ip="127.0.%d.%d" % (int(splitkey[0][1:]), 100 + hostcounter))
        hostcounter += 1
    # Link generated nodes according to the topology
    for key in topology:
        for value in topology[key]:
           if value not in ("healthy", "infected"):
               node1 = net.get(key)
               node2 = net.get(value)
               if not node1.connectionsTo(node2):
                   # print(node1)
                   # print(node2)
                   net.addLink(node1, node2)
    return net


if __name__ == '__main__':

    infected_nodes = ["s3_10", "s1_7", "s2_9"]
    #path2topo = os.getcwd() + "/rlearn/graphs/topo_generic.csv"
    path2topo = "../graphs/topo_generic.csv"
    path2pos = "../graphs/pos_generic.csv"
    topology=network.getTopologyFromCsv2(path2topo, infected_nodes)
    pos = network.getPosFromCsv(path2pos)

    net = create(topology)
    net.start()
    # Make Switch act like a normal switch
    #net['s1'].cmd('ovs-ofctl add-flow s1 action=normal')
    # Make Switch act like a hub
    #net['s1'].cmd('ovs-ofctl add-flow s1 action=flood')
    CLI( net )
    net.stop()
