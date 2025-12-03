"""
Implementation of the Attacker

"""
import networkx as nx
import random


def attack(net, netgraph, nwstate, critserver, mode, attmode):
    # Collect all infected Nodes
    attnodes = [node[1] for node in nwstate if [1, node[1]] in nwstate and len(node) == 2]
    # All infected nodes can only infect 1 node in neighbourhood
    if attmode == 0:
        ##print("All attnodes %s (all can infect one)" % attnodes)
        for attacker in attnodes:
            bestVictim = 0
            try:
                shortestpath = nx.shortest_path(netgraph, attacker, critserver)[
                    1
                ]  # Get next node of Shortestpath to the Critserver
            except nx.NetworkXNoPath:
                shortestpath = 0
            # Get Neighbourhood
            adjacency = [
                node
                for node in nwstate
                if len(node) > 2 and (node[1] == attacker or node[2] == attacker)
            ]
            #print("Attacker %s neighbourhood %d //// %s" %  (attacker, len(adjacency), adjacency ))
            for linkpair in adjacency:
                # Detect wich node of each Linkpair is the attacking Node or the possible Victim
                if linkpair[1] == attacker:
                    possVictim = linkpair[2]
                elif linkpair[2] == attacker:
                    possVictim = linkpair[1]
                # Check if possible Victim is already infected
                if [1, possVictim] in nwstate:
                    continue
                if linkpair[0] == 1:
                    # Check if possible Victim is on the shortest path and Link is active
                    if possVictim == shortestpath:
                        bestVictim = shortestpath
                        break
                    # If no best Victim to be found, set it to arbitary first Victim
                    if bestVictim == 0:
                        bestVictim = possVictim
            if bestVictim != 0:
                # Infest best Victim, if any
                index = nwstate.index([0, bestVictim])
                nwstate[index] = [1, bestVictim]
                ##print("Attacked node %s " % bestVictim)
    elif attmode == 1:  # Infect nodes infects all nodes neighbourhood
        # selected_attnode = random.choice(attnodes)
        print("All attnodes %s (all can infect all)" % attnodes)
        for attacker in attnodes:
            adjacency = [
                node
                for node in nwstate
                if len(node) > 2 and (node[1] == attacker or node[2] == attacker)
            ]  # Get Neighbourhood
            for linkpair in adjacency:
                # Detect wich node of each Linkpair is the attacking Node or the possible Victim
                if linkpair[1] == attacker:
                    possVictim = linkpair[2]
                elif linkpair[2] == attacker:
                    possVictim = linkpair[1]
                # Check if possible Victim is already infected
                if [1, possVictim] in nwstate:
                    continue
                # Infest all possible Victims
                if linkpair[0] == 1:
                    index = nwstate.index([0, possVictim])
                    nwstate[index] = [1, possVictim]
    # One infected node can only infect one node in neighbourhood
    elif attmode == 2:
        print("All attnodes %s (one can infect one)" % attnodes)
        mybestVictimList = []
        for attacker in attnodes:
            bestVictim = 0
            try:
                # Get next node of Shortestpath to the Critserver
                shortestpath = nx.shortest_path(netgraph, attacker, critserver)[1]
            except nx.NetworkXNoPath:
                shortestpath = 0
            # Get Neighbourhood
            adjacency = [
                node
                for node in nwstate
                if len(node) > 2 and (node[1] == attacker or node[2] == attacker)
            ]
            # print("Attacker %s adjacency %s" % (attacker, adjacency))
            for linkpair in adjacency:
                # Detect wich node of each Linkpair is the attacking Node or the possible Victim
                if linkpair[1] == attacker:
                    possVictim = linkpair[2]
                elif linkpair[2] == attacker:
                    possVictim = linkpair[1]
                # Check if possible Victim is already infected
                if [1, possVictim] in nwstate:
                    continue
                if linkpair[0] == 1:
                    # Check if possible Victim is on the shortest path and Link is active
                    if possVictim == shortestpath:
                        bestVictim = shortestpath
                        break
                    # If no best Victim to be found, set it to arbitary first Victim
                    if bestVictim == 0:
                        bestVictim = possVictim
            if bestVictim != 0:
                # Infest best Victim, if any
                mybestVictimList.append(bestVictim)
                # index = nwstate.index([0,bestVictim])
                # nwstate[index] = [1, bestVictim]
        mybestVictim = random.choice(mybestVictimList)
        print("Selcted new node to attack %s from list %s" % (mybestVictim, mybestVictimList))
        index = nwstate.index([0, mybestVictim])
        nwstate[index] = [1, mybestVictim]
    else:
        print("Unknown attmode")
        exit(1)
    # Return new NWSTATE
    return nwstate
