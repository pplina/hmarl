"""
Initialiye custom Topology out of datapath

"""
#from mininet.node import Controller

import os, csv

import networkx as nx
import matplotlib.pyplot as plt

rw_steps = 0
data_ex = 0


_CERERE_VERBOSE = os.getenv("CERERE_VERBOSE", "0").lower() not in ("0", "false", "no", "")


def _vprint(*args, **kwargs):
    if _CERERE_VERBOSE:
        print(*args, **kwargs)



def getTopologyFromCsv(path):
    # Create Dictionary containing the Network Topology as adjacency list including the health of each node
    topology = {}
    with open(path, "r") as stream:
        reader = csv.reader(stream, delimiter=",")
        for row in reader:
            topology[row[0]] = [cell for cell in row[1:] if cell]  # Ignore Blank Cells
    for key in topology:  # Check wether the Node is Healthy or not
        if "a" not in key:
            topology[key].insert(0, "healthy")  # Everyone is healthy, exept the attackers
            continue
        topology[key].insert(0, "infected")  # Infect the attacking nodes
        print(key)
    return topology


def getTopologyFromCsv2(path, in_infected_notes):
    # Create Dictionary containing the Network Topology as adjacency list including the health of each node
    topology = {}
    with open(path, "r") as stream:
        reader = csv.reader(stream, delimiter=",")
        for row in reader:
            topology[row[0]] = [cell for cell in row[1:] if cell]  # Ignore Blank Cells
    for key in topology:  # Check wether the Node is Healthy or not
        if key not in in_infected_notes:
            topology[key].insert(0, "healthy")  # Everyone is healthy, expect the attackers
            continue
        topology[key].insert(0, "infected")  # Infect the attacking nodes
        # print(key)
    return topology


# generate NWSTATE(LIST) out of Topology(Dictionary)
def getStateFromTopology(topology):
    nwstate = []
    for element in topology:
        if topology[element][0] == "healthy":
            nwstate.append([0, element])
        else:
            nwstate.append([1, element])
    for element in topology:
        for value in topology[element][1:]:
            if [1, value, element] not in nwstate:
                nwstate.append([1, element, value])
    return nwstate


# Comprimise NWSTATE so that DQN-Network can handle it( 1D-Vector of 0 and 1's)
def getVectorFromState(nwstate):
    flatState = []
    for value in nwstate:
        flatState.append(value[0])
    return flatState


def getVectorFromState2(nwstate, critserver, netgraph):
    flatState = []

    # Compute reachability and distances once
    try:
        reachable = nx.node_connected_component(netgraph, critserver)
    except Exception:
        reachable = {critserver}
    try:
        dist_from_crit = nx.single_source_shortest_path_length(netgraph, critserver)
    except Exception:
        dist_from_crit = {critserver: 0}

    for value in nwstate:
        flatState.append(value[0])
        if len(value) != 2:
            continue

        node = value[1]
        if node == critserver:
            flatState.append(0)
            continue

        if node not in reachable:
            flatState.append(1)
            continue

        d = dist_from_crit.get(node)
        if d is None:
            flatState.append(1)
        else:
            flatState.append(d / 8)

    return flatState


def createNetwork(net, netgraph, topology, mode):
    # Add hosts and switches extracted from the CSV, switches/AccesPoints should be beginning with s
    hostcounter = 0
    for key in topology:
        for value in topology[key]:
            if value not in ("healthy", "infected"):
                netgraph.add_edge(key, value)
    return netgraph


# Function to reset the Network to state of the Topology
def resetNetwork(topology, net, netgraph, mode):
    for key in topology:
        for value in topology[key]:
            if value not in ("healthy", "infected"):
                netgraph.add_edge(key, value)
    nwstate = getStateFromTopology(topology)
    rw_steps = 0
    data_ex = 0
    return nwstate, netgraph


# Function to get the Positions of each node
def getPosFromCsv(path):
    pos = {}
    with open(path, "r") as stream:
        reader = csv.reader(stream, delimiter=",")
        for row in reader:
            pos[row[0]] = tuple([float(cell) for cell in row[1:] if cell])  # Ignoe Blank Cells
    return pos


# Plotting routine for the Network-Topology
def drawGraph(netgraph, pos, nwstate, path, action, topology, actionSpace):
    #print("Action %d" % action)
    #print(actionSpace)
    if action < len(topology) and action >= 0:
        title = "Isolate and patch node " + actionSpace[action]
        ###title = "Isolate node " + actionSpace[action]
        plt.title(title)
    elif action >= len(topology) and action < len(topology) + 3 and action > 0:
        title = "Migrate crtitical server to node " + actionSpace[action]
        plt.title(title)
    elif action >= len(topology) + 3 and action < len(topology) + 4 and action > 0:
        title = "Block traffic " + actionSpace[action]
        plt.title(title)
    elif action == -1:
        title = "Initial state"
        plt.title(title)
    elif action == -2:
        title = "Attack"
        plt.title(title)
    else:
        title = "Do nothing"
        plt.title(title)
    # print(title)
    color_map = []
    for node in netgraph:
        if [1, node] in nwstate:
            color_map.append("red")
            # if node == 's1_2':
            #    print(nwstate)
        else:
            color_map.append("green")
    nx.draw(
        netgraph, pos=pos, node_color=color_map, font_size=14, font_weight="bold", with_labels=True
    )
    plt.savefig(path)
    plt.clf()
    plt.cla()
    plt.close()
    cmd = "chown ubuntu:ubuntu " + path
    os.popen(cmd)



# Node statistic function
def getNodeStatistic(critserver, optserver, topology, nwstate, netgraph, in_block_traffic):
    global data_ex
    reachable_healthy_nodes = 0
    reachable_infected_nodes = 0
    healthyNodes = len([node for node in nwstate if [0, node[1]] in nwstate and len(node) == 2])
    infectedNodes = len(topology) - healthyNodes

    # Get component of all nodes connected to the critical Server (faster than
    # computing all connected components + sorting)
    try:
        reachable = nx.node_connected_component(netgraph, critserver)
    except Exception:
        reachable = {critserver}
    
    # Calculate amount of healthy nodes with no infection on path
    #p = nx.shortest_path(netgraph, critserver, 'tr_m_1_5')
    p = nx.shortest_path(netgraph, source=critserver) 
    healthy_nodes_no_infected_subg = 0
    all_reachable_nodes = 0
    for node in p:
        #print("Path of node %s:%s" % (node, p[node]))
        path_with_infected = False
        for path_node in p[node]:
            #print("Entry %s" % path_node)
            if [1, path_node] in nwstate:                
                #print("Path with infected")
                path_with_infected = True
                break
        if path_with_infected == False:
            healthy_nodes_no_infected_subg += 1 
        all_reachable_nodes += 1  
                
    # Calculate amount of infected healthy nodes connected to the critical server
    for node in reachable:
        if [1, node] in nwstate:
            #print("Reachable infected %s" % node)
            reachable_infected_nodes += 1
        if [0, node] in nwstate:
            #print("Reachable healthy %s" % node)
            reachable_healthy_nodes += 1
  
    if in_block_traffic == 0:
        data_ex = reachable_infected_nodes + data_ex

    ##print("Statistic: Num Nodes %d, Reachable nodes: All = %d, Healthy = %d, Infected = %d, Healthy (no infection on path) = %d, Data Ex %d" % (len(topology), all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex))
    #print(nwstate)
    #return reachable_healthy_nodes, reachable_infected_nodes
    return all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex



# reward function 2
def getReward2(critserver, optserver, topology, nwstate, pFlag, netgraph, in_action, in_actionSpace, in_block_traffic):
    global rw_steps
    rw_steps = rw_steps + 1 
    reachable_healthy_nodes = 0
    reachable_infected_nodes = 0
    healthyNodes = len([node for node in nwstate if [0, node[1]] in nwstate and len(node) == 2])
    infectedNodes = len(topology) - healthyNodes
    terminate = 0
    reward = 0

    # Check if Critical Server is infected, return reward = -1 and terminated = 1
    if [1, critserver] in nwstate:
        reward = -1
        terminate = 1
        ##print("Terminate (infected critical server), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex


    # Component of nodes connected to critical server
    try:
        reachable = nx.node_connected_component(netgraph, critserver)
    except Exception:
        reachable = {critserver}
    
    all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex_tmp = getNodeStatistic(critserver, optserver, topology, nwstate, netgraph, in_block_traffic)
    #print("Num Nodes %d, Reachable nodes: All = %d, Healthy = %d, Infected = %d, Healthy (no infection on path) = %d, Data Ex %d" % (len(topology), all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex))
    #print(nwstate)
	
    # Check wether subgraph is in size smaller as 0.25 of the original Topology, return reward = -1 and terminated = 1
    if len(reachable) - reachable_infected_nodes < 0.25 * len(topology):
        reward = -1
        terminate = 1
        ##print("Terminate (small leftover topology), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex

   # set reward, if no infected nodes in subgraph
    if reachable_infected_nodes == 0:
        reward = healthy_nodes_no_infected_subg / len(topology)
        terminate = 1
        _vprint("Terminate (no infected nodes in subgraph), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex

    if in_action < len(topology):
        # isopatchNode
        node_to_patch = in_actionSpace[in_action]
        #print("Selected node for isoPatch %s " % node_to_patch)
        if node_to_patch == critserver or node_to_patch in optserver:
            reward = -1
            terminate = 1
            ##print("Terminate (crit/opt server patched), Reward = %f" % reward)
            return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex
        else: 
            reward = ( (healthy_nodes_no_infected_subg / len(topology)) - (reachable_infected_nodes / len(topology)) )/5
            #reward = 0
            if pFlag == -1:
                 reward = 0
    elif in_action >= len(topology) and in_action < len(topology) + 3:
        # migrateServer
        reward = 0
    elif in_action >= len(topology) + 3 and in_action < len(topology) + 4:
        # blockTraffic
        reward = 0
        #reward = 1
        #terminate = 1
        ##print("Terminate (block traffic), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex
    else:
        # doNothing
        reward = 0
        if in_action >= len(in_actionSpace):
            _vprint("Exit, action outside action space")
            exit(1)
    
    #print("Rw steps %d, Reward = %f" % (rw_steps, reward))
    return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex


# reward function 3
def getReward3(critserver, optserver, topology, nwstate, pFlag, netgraph, in_action, in_actionSpace, in_block_traffic):
    global rw_steps
    rw_steps = rw_steps + 1 
    reachable_healthy_nodes = 0
    reachable_infected_nodes = 0
    healthyNodes = len([node for node in nwstate if [0, node[1]] in nwstate and len(node) == 2])
    infectedNodes = len(topology) - healthyNodes
    terminate = 0
    reward = 0

    # Check if Critical Server is infected, return reward = -1 and terminated = 1
    if [1, critserver] in nwstate:
        reward = -1
        terminate = 1
        _vprint("Terminate (infected critical server), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex


    # Component of nodes connected to critical server
    try:
        reachable = nx.node_connected_component(netgraph, critserver)
    except Exception:
        reachable = {critserver}
    
    all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex_tmp = getNodeStatistic(critserver, optserver, topology, nwstate, netgraph, in_block_traffic)
    #print("Num Nodes %d, Reachable nodes: All = %d, Healthy = %d, Infected = %d, Healthy (no infection on path) = %d, Data Ex %d" % (len(topology), all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, data_ex))
    #print(nwstate)
	
    # Check wether subgraph is in size smaller as 0.25 of the original Topology, return reward = -1 and terminated = 1
    if len(reachable) - reachable_infected_nodes < 0.25 * len(topology):
        reward = -1
        terminate = 1
        _vprint("Terminate (small leftover topology), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex


   # set reward, if no infected nodes in subgraph
    if reachable_infected_nodes == 0:
        reward = healthy_nodes_no_infected_subg / len(topology)
        terminate = 1
        _vprint("Terminate (no infected nodes in subgraph), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex


    if in_action < len(topology):
        # isopatchNode
        node_to_patch = in_actionSpace[in_action]
        #print("Selected node for isoPatch %s " % node_to_patch)
        if node_to_patch == critserver or node_to_patch in optserver:
            reward = -1
            terminate = 1
            _vprint("Terminate (crit/opt server patched), Reward = %f" % reward)
            return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex
        else: 
            reward = ( (healthy_nodes_no_infected_subg / len(topology)) - (reachable_infected_nodes / len(topology)) )/5
            #reward = 0
            if pFlag == -1:
                 reward = 0
    elif in_action >= len(topology) and in_action < len(topology) + 3:
        # migrateServer
        reward = 0
    elif in_action >= len(topology) + 3 and in_action < len(topology) + 4:
        # blockTraffic
        #reward = 0
        reward = 1
        terminate = 1
        _vprint("Terminate (block traffic), Reward = %f" % reward)
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex
    else:
        # doNothing
        reward = 0
        if in_action >= len(in_actionSpace):
            _vprint("Exit, action outside action space")
            exit(1)
    
    _vprint("Rw steps %d, Reward = %f" % (rw_steps, reward))
    return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes, data_ex



"""
# reward function
def getReward(critserver, optserver, topology, nwstate, pFlag, netgraph):
    global rw_steps
    rw_steps = rw_steps +1 
    reachable_healthy_nodes = 0
    reachable_infected_nodes = 0
    healthyNodes = len([node for node in nwstate if [0, node[1]] in nwstate and len(node) == 2])
    infectedNodes = len(topology) - healthyNodes
    terminated = 0

    # Check if Critical Server is infected, return reward = -1 and terminated = 1
    if [1, critserver] in nwstate:
        reward = -1
        terminate = 1
        rw_steps = 0
        reachable_healthy_nodes = 0
        reachable_infected_nodes = 0
        print("Terminate, infected critical server")
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes
        #return reward, 1, reachable_nodes

    # Get Subgraph of all nodes connected to the critical Server
    reachable = [
        c
        for c in sorted(nx.connected_components(netgraph), key=len, reverse=True)
        if critserver in c
    ]
    #print(reachable[0])
    
    # Calculate amount of healthy nodes with no infection on path
    #p = nx.shortest_path(netgraph, critserver, 'tr_m_1_5')
    p = nx.shortest_path(netgraph, source=critserver) 
    my_no_infected_subg = 0
    all_reachable_nodes = 0
    for node in p:
        #print("Path of node %s:%s" % (node, p[node]))
        path_with_infected = False
        for path_node in p[node]:
            #print("Entry %s" % path_node)
            if [1, path_node] in nwstate:                
                #print("Path with infected")
                path_with_infected = True
                break
        if path_with_infected == False:
            my_no_infected_subg += 1 
        all_reachable_nodes += 1  
                
    # Calculate amount of infected healthy nodes connected to the critical server
    for node in reachable[0]:
        if [1, node] in nwstate:
            #print("Reachable infected %s" % node)
            reachable_infected_nodes += 1
        if [0, node] in nwstate:
            #print("Reachable healthy %s" % node)
            reachable_healthy_nodes += 1
  
    print("Num Nodes %d, Reachable nodes: All = %d, Healthy = %d, Infected = %d, Healthy (no infection on path) = %d" % (len(topology), all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, my_no_infected_subg))
    #print(nwstate)

    # Check wether subgraph is in size smaller as 0.25 of the original Topology, return reward = -1 and terminated = 1
    if len(reachable[0]) - reachable_infected_nodes < 0.25 * len(topology):
        reward = -1
        terminate = 1
        rw_steps = 0
        #reachable_healthy_nodes = 0
        # print("Terminate because of leftover topology < %f" % (0.25*len(topology)))
        print("Terminate, small leftover topology")
        return reward, terminate, reachable_healthy_nodes, reachable_infected_nodes


    # Calculate Reward
    reward = (
        0.5 * (len(reachable[0]) / len(topology))  # portion of reachable nodes
        + 0.4 * ((len(reachable[0]) - reachable_infected_nodes) / len(reachable[0]))  # portion of healthy nodes that are reachable
        + 0.1 * (healthyNodes / len(topology))  # portion of healthy nodes inside the whole topology
    )

    #reward = reward - (rw_steps * 0.05)     

	# Set Multiplier
    multiplier = 0.2
    reward = reward * multiplier
  
    if len(nwstate) > 50: #simple differentiation betweenn the scenarios (enterprise)
        # Apply Penalty for invalid action (Critserver/Optserver or nodes with no connection to is Critserver patched)
        if pFlag == 1:
            reward = 0
            #reward = reward - 0.4 * multiplier
        # Apply Penalty for migrating the Critical Server
        if pFlag == 2:
            reward = 0
            #reward = reward - 0.2 * multiplier
        # Apply Penalty for patching a healty node
        if pFlag == 3:
            reward = 0
            #reward = reward - 0.2 * multiplier


    if len(nwstate) < 50: #simple differentiation betweenn the scenarios (miltitary)
        reward = reward - (rw_steps * 0.05)
        if pFlag == 1:
            reward = reward / 4
        if pFlag == 2:
            reward = reward / 4
        if pFlag == 3:
           reward = reward / 4
        if pFlag == 6:
           reward = reward / 4
  

    print("Reward %f, pflag %f, steps %d" % (reward, pFlag, rw_steps))
    # set reward, if no infected nodes in subgraph
    if reachable_infected_nodes == 0:
        terminated = 1
        rw_steps = 0
        print("Terminate, no infected nodes in subgraph")
#        if len(nwstate) > 50: #simple differentiation betweenn the scenarios
#            reward = len(reachable[0]) / (
#                len(topology) - 3
#            )  # decrease topology by 3 to ignore starting attacking nodes
        #reward = 1
        print("Terminate reward %f" % (reward))
    #print("Reachable nodes %s" % reachable_healthy_nodes)
    return reward, terminated, reachable_healthy_nodes, reachable_infected_nodes
"""

def reachable_node_list2(critserver, node, netgraph):
    reachable_node_list = [c for c in sorted(nx.connected_components(netgraph), key=len, reverse=True) if critserver in c]
    for i in reachable_node_list:
        if node in i:
            reachable_node_list
            #print("Critical server %s with reachable node list %s" % (critserver, str(reachable_node_list)))
        else:
           reachable_node_list = []
           #print("Critical server %s with reachable node list %s" % (critserver, str(reachable_node_list)))
    return reachable_node_list
