"""
Implementation of the Defender
"""
import networkx as nx


def isoNpatchNode(net, netgraph, critserver, optserver, node, nwstate, btraffic):
    ##print("Function: Isolate and patch node %s" % node)
    
    #print("nwstate 1 %s" % str(nwstate))
    inactiveLinks = 0
    pFlag = 0
    # Set pFlag (penaltyFlag) = 4 if infected node patched
    if [element for element in nwstate if [1, node] in nwstate]:
        #pFlag = 4
        # Get neighbourhood
        adjacency = [
            element
            for element in nwstate
            if len(element) > 2 and (element[1] == node or element[2] == node)
        ]
        #print("------> Infected node %s with neighbourhood size %d " % (node, len(adjacency)))
        # Get nodes connected to to the server
        reachable = [
            c
            for c in sorted(nx.connected_components(netgraph), key=len, reverse=True)
            if critserver in c
        ]
        if node in reachable[0]:
            shortestpath = nx.shortest_path_length(netgraph, node, critserver)  # Get shortestpath to the Critserver
            ##print(">>>> Select infected node %s with hops %d to critserver" % (node, shortestpath))
            pFlag = shortestpath
        else:
            # set pFlag = 1 (Action Invalid) if node with no connection to critserver is patched
            ##print(">>>> Select infected node %s with no connection to critserver" % node)
            pFlag = -1

    if [0, node] in nwstate:
        ##print(">>>> Select healty node %s" % node)
        pFlag = -1

    for value in nwstate:
        index = nwstate.index(value)
        if len(value) > 2 and value[0] == 1 and (value[1] == node or value[2] == node):
            # remove edge in netgraph
            netgraph.remove_edge(value[1], value[2])
            nwstate[index][0] = 0
            inactiveLinks = inactiveLinks + 1
        if len(value) == 2 and value[1] == node:
            # set node state as healthy
            #print("set node state as healthy %s" % node)
            nwstate[index][0] = 0

    # Set pFlag = 1 (Action Invalid) if no links were shutdown
#a    if inactiveLinks == 0:
#a        pFlag = 1
 
    # Set pFlag = 6 if iso and patch application computer 
#a    if "ac_" in node: 
#a        #print("Function: Isolate and patch ac node %s" % node)
#a        pFlag = 6
    #print("nwstate 2 %s" % str(nwstate))
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic

"""
def isoNode(net, netgraph, critserver, optserver, node, nwstate, btraffic):
    print("Function: Isolate node %s" % node)
    
    #print("nwstate 1 %s" % str(nwstate))
    inactiveLinks = 0
    pFlag = 0
    # Set pFlag (penaltyFlag) = 4 if infected node patched
    if [element for element in nwstate if [1, node] in nwstate]:
        pFlag = 4
        # Get neighbourhood
        adjacency = [
            element
            for element in nwstate
            if len(element) > 2 and (element[1] == node or element[2] == node)
        ]
        #print("------> Infected node %s with neighbourhood size %d " % (node, len(adjacency)))
        # Get nodes connected to to the server
        reachable = [
            c
            for c in sorted(nx.connected_components(netgraph), key=len, reverse=True)
            if critserver in c
        ]
        if node in reachable[0]:
            shortestpath = nx.shortest_path_length(
                netgraph, node, critserver
            )  # Get shortestpath to the Critserver
            #print("------> Infected node %s with hops %d to critserver" % (node, shortestpath))
            pFlag = pFlag + ( 1 / shortestpath * 0.2 )  # workaround to increase the bonusreward for patching infected nodes near the critical server
        else:
            #print("------> Infected node %s with no connection to critserver" % node)
            pFlag = 1

    # Set pFlag = 3 if healthy node is patched
    if [element for element in nwstate if [0, node] in nwstate]:
        pFlag = 3

    # Set pFlag = 1 (Action Invalid) if Node is in Critserver/Optserver is patched
    if node == critserver or node in optserver:
        pFlag = 1
    #    return nwstate, netgraph, pFlag, critserver, optserver

    for value in nwstate:
        index = nwstate.index(value)
        if len(value) > 2 and value[0] == 1 and (value[1] == node or value[2] == node):
            # remove edge in netgraph
            if ( netgraph.has_edge(value[1], value[2]) == True):
                netgraph.remove_edge(value[1], value[2])
            nwstate[index][0] = 0
            inactiveLinks = inactiveLinks + 1
        if len(value) == 2 and value[1] == node:
            # set link inactive in NWSTATE
            xx = 0
            #print(nwstate[index][0])
            #nwstate[index][0] = 0

    # Set pFlag = 1 (Action Invalid) if no links were shutdown
    if inactiveLinks == 0:
        pFlag = 1
        ####return nwstate, netgraph, pFlag, critserver, optserver

    # Set pFlag = 6 if iso and patch application computer 
    if "ac_" in node: 
        #print("Function: Isolate and patch ac node %s" % node)
        pFlag = 6
    
    print("nwstate 2 %s" % str(nwstate))
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic
"""


def reconnNode(net, netgraph, critserver, optserver, node, nwstate, btraffic):
    ##print("Function: Reconnect node %s" % node)
    activeLinks = 0
    for value in nwstate:
        index = nwstate.index(value)
        if len(value) > 2 and value[0] == 0 and (value[1] == node or value[2] == node):
            # add edge in netgraph
            netgraph.add_edge(value[1], value[2])
            nwstate[index][0] = 1
            activeLinks = activeLinks + 1
    # Set pFlag = 1 (Action Invalid) if no Links were activated
    if activeLinks == 0:
        pFlag = 1
        return nwstate, netgraph, pFlag, critserver, optserver
    # Return pFlag = 1 (Action Invalid) if Node is in Critserver/Optserver
    if node == critserver or node in optserver:
        pFlag = 1
        return nwstate, netgraph, pFlag, critserver, optserver, btraffic
    pFlag = 0
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic



def migrateServer(net, netgraph, critserver, optserver, node, nwstate, btraffic):
    ##print("Function: Migrate server node %s" % node)
    # If the specific node is not Infected already, migrate criticalserver to this specific node
    if node in optserver and [0, node] in nwstate:
        optserver = optserver + [critserver]
        optserver.remove(node)
        critserver = node
        # Set pFlag = 2 if migrate server 
        pFlag = 2
        return nwstate, netgraph, pFlag, critserver, optserver, btraffic
    # Set pFlag = 1 (Action Invalid) if migrate to infected node
    pFlag = 1
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic


# Function that blocks traffic
def blockTraffic(net, netgraph, critserver, optserver, nwstate, btraffic):
    ##print("Function: Block traffic")
    # Set pFlag = 1 (Action Invalid) if block traffic
    pFlag = 1
    btraffic = 1
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic


# Function that does Nothing
def doNothing(net, netgraph, critserver, optserver, nwstate, btraffic):
    ##print("Function: Do nothing")
    # Set pFlag = 1 (Action Invalid) if do nothing
    pFlag = 1
    return nwstate, netgraph, pFlag, critserver, optserver, btraffic



# Return list of Nodenames that can be used for each Action, resulting in a list of all possible Actions
def getactionSpace(nwstate, critserver, optserver, topology):
    actionSpace = []
    for element in topology:
        actionSpace.append(element)
    #    for element in topology:
    #        actionSpace.append(element)
    actionSpace.append(critserver)
    for element in optserver:
        actionSpace.append(element)
    actionSpace.append("btraffic")
    actionSpace.append("else")
    # print("Actionspace %s" % actionSpace)
    # print("Len toplog %d" %  len(topology))
    return actionSpace


# Given an Integer <= len(actionspace), return specified action.
def getAction(net, netgraph, critserver, optserver, action, actionSpace, topology, nwstate, btraffic):
    #print("Actionspace %s" % actionSpace)
    #print(action.item())
    #print(actionSpace[action])
    ##print("------> Get action %d, Crit Serv %s, Opt Serv %s, len topo %d" % (action, critserver,optserver, len(topology)))
    if action < len(topology):
        return isoNpatchNode(net, netgraph, critserver, optserver, actionSpace[action], nwstate, btraffic)
        #return isoNode( net, netgraph, critserver, optserver, actionSpace[action], nwstate, btaffic)
    elif action >= len(topology) and action < len(topology) + 3:
        return migrateServer(net, netgraph, critserver, optserver, actionSpace[action], nwstate, btraffic)
    elif action >= len(topology) + 3 and action < len(topology) + 4:
        return blockTraffic(net, netgraph, critserver, optserver, nwstate, btraffic)
    else:
        if action >= len(actionSpace):
            print("Action outside action space")
            exit(1)
        return doNothing(net, netgraph, critserver, optserver, nwstate, btraffic)


### reconnNode does not fit to the reward function
# Given an Integer <= len(actionspace), return specified action.
# def getAction(net,netgraph, critserver,optserver,action,actionSpace, topology , nwstate, mode):
#    if action < len(topology):
#        return isoNpatchNode(net,netgraph, critserver,optserver,actionSpace[action], nwstate, mode)
#    elif action >= len(topology) and action < (len(topology) * 2):
#        return reconnNode(net,netgraph, critserver,optserver,actionSpace[action], nwstate, mode)
#    elif action >= len(topology) * 2 and action < len(topology) * 2 + 3:
#        return migrateServer(net,netgraph, critserver,optserver,actionSpace[action], nwstate, mode)
#    else:
#        return doNothing(net,netgraph, critserver,optserver,actionSpace[action], nwstate, mode)
