import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
from typing import Optional

#sys.path.append("../../../rlearn/rlearn/")
sys.path.append("rlearn/rlearn/")

import attacker as attacker
import defender as defender
import network as network

NOTYPE_ACTION = 0
ATTACK_ACTION = 1
DEFENSE_ACTION = 2


#nodes = ["HQ", "M-1-1", "M-1-2", "M-1-3", "M-1-4", "M-1-5", "M-2-1", "M-2-2", "M-2-3",
#         "M-2-4", "M-2-5"]
#edges = [["M-1-1", "M-2-1"], ['HQ', "M-1-1"], ["M-1-1", "M-1-2"], ["M-1-2", "M-1-3"],
#         ["M-1-3", "M-1-4"], ["M-1-4", "M-1-5"], ['HQ', "M-2-1"], ["M-2-1", "M-2-2"],
#         ["M-2-2", "M-2-3"], ["M-2-3", "M-2-4"], ["M-2-4", "M-2-5"]]

#node_pos = [(256, 40), (128, 120), (128, 200), (128, 280), (128, 360), (128, 440),
#            (384, 120), (384, 200), (384, 280), (384, 360), (384, 440)]

#node_map = {}
#for n in nodes:
#    node_map[n] = {"name": n, "attr": 0}

#color_mapping = {0: (0, 255, 0), 1: (255, 255, 0), 2: (255, 165, 0), 3: (225, 0, 0), 4: (200, 0, 0)}


class CerereNet(gym.Env):
    #metadata = {'render_modes': ['human', 'rgb_array'], "render_fps": 4}
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, render_mode=None, rw_func=1, scenario='enterprise'):
#    def __init__(self, render_mode: Optional[str] = None, rw_func: Optional[int] = 1, scenario: Optional[str] = 'military'):
        #print("#### Beginn INIT new topology ###")
        if scenario == 'enterprise':
            infected_nodes = ["s3_10", "s1_7", "s2_9"]
            #path2topo = "/home/ubuntu/src/rl-test/rlearn/graphs/topo_generic.csv"
            #path2pos = "/home/ubuntu/src/rl-test/rlearn/graphs/pos_generic.csv" 
            path2topo = os.getcwd() + "/rlearn/graphs/topo_generic.csv"
            path2pos = os.getcwd() + "/rlearn/graphs/pos_generic.csv"
            self.init_critserver = self.critserver = "d3_1"
            self.init_optserver = self.optserver = ["d3_2", "d4_1"]
            self.block_traffic = 0
        if scenario == 'military':
            #infected_nodes = ["ac-m-2-5"]
            infected_nodes = ["ac-m-2-4"]
            path2topo = os.getcwd() + "/rlearn/graphs/topo_cerere.csv"
            path2pos = os.getcwd() + "/rlearn/graphs/pos_cerere.csv"
            self.init_critserver = self.critserver = "as-hq"
            self.init_optserver = self.optserver = ["as-hqa", "as-hqa2"] # ["ash_hqa"]  # what is ash_hqa?
            self.block_traffic = 0

        self.net = None  # mininet is not used
        self.mode = "none"  # net type wifi,lan and none in case of no mininet
        self.rw_function = rw_func
        # 0 = All attackers can infect one single node in the neighbourhood
        # 1 = All attacker infect all nodes in neighbourhood
        # 2 = One attacker infects one node in the neighbourhood
        self.attackmode = 0
        self.topology = network.getTopologyFromCsv2(path2topo, infected_nodes)
        #print(self.topology)
        self.pos = network.getPosFromCsv(path2pos)
        self.nwstate = network.getStateFromTopology(self.topology)
        #print(self.nwstate)
        self.actionSpace = defender.getactionSpace(self.nwstate, self.critserver, self.optserver, self.topology)
        print(self.actionSpace)
        self.netgraph = nx.Graph()
        self.netgraph = network.createNetwork(self.net, self.netgraph, self.topology, self.mode)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        #print("Size state space %d, Size action space %d" % (len(self.state), len(self.actionSpace)))
        #print("#### End INIT new topology ###")

        # Encode the network state as an adjacency matrix
        # self.observation_space = spaces.Box(0, 4, shape = (11,11), dtype=int)
        # Encode the network state as an adjacency vector
        #self.observation_space = spaces.Box(0, 4, shape=(11,), dtype=int)
        self.observation_space = spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32)
        #self.observation_state = np.zeros((self.size, self.size, self.size), dtype=np.float32)
        np.set_printoptions(threshold=sys.maxsize)
        self.observation_state = np.array(self.flatState, dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actionSpace))
        self.actualActionType = NOTYPE_ACTION
        self.actualAction = -1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        print("Render mode is %s" % self.render_mode)

        self.data_ex = 0
        self.mystep = 0


    def observe(self):
        return self.observation_state


    def _get_info(self):
        # Returns attacked: True in case any node has a value other than zero
        attack = False
        if [1, self.critserver] in self.nwstate:
            attack = True
            #print("Infected critical server")
            return {"Infected critical server": attack}
        else:
            return {"Infected critical server": attack}


    def _init(self):
        # Attack once in Order to achieve Initial Configuration of the Paper used
        self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, 1, self.mode, self.attackmode)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, self.data_ex = network.getNodeStatistic(self.critserver, self.optserver, self.topology, self.nwstate, self.netgraph, self.block_traffic)
        # print(self.state)
        self.observation_state = np.array(self.flatState, dtype=np.float32)
        # print(self.observation_state)
        self.actualActionType = NOTYPE_ACTION
        self.actualAction = -1


    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.data_ex = 0
        self.mystep = 0
        self.critserver = self.init_critserver
        self.optserver = self.init_optserver
        self.block_traffic = 0
        self.nwstate, self.netgraph = network.resetNetwork(self.topology, self.net, self.netgraph, self.mode) # Reset Network
        self._init()
        observation = self.observe()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info


    def step(self, action):
        truncated = False
        terminated = False

        #print("#### Begin STEP in openAI gym ###")
        #print(self.nwstate)
        self.mystep = self.mystep + 1
        self.actualActionType = DEFENSE_ACTION
        self.actualAction = action
        self.nwstate, self.netgraph, pFlag, self.critserver, self.optserver, self.block_traffic = defender.getAction(self.net, self.netgraph,self.critserver, self.optserver, action, self.actionSpace, self.topology, self.nwstate, self.block_traffic)
        
        if self.render_mode == "human":
             #self._render_frame()
             self.render()
        
        self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, 1, self.mode, self.attackmode)
        # reward function
        #reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes = network.getReward(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph)
        if self.rw_function == 2:
            reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward3(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
        else:
            reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward2(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
        self.actualActionType = ATTACK_ACTION
        self.actualAction = -2
        #print(self.nwstate)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        self.observation_state = np.array(self.flatState, dtype=np.float32)
 
        if self.render_mode == "human":
             self.render()
        #print(self.observation_state)

        observation = self.observe()
        info = self._get_info()
        if terminated2 == 1:
            terminated = True
        #print("#### End STEP in openAI gym ###")
        return observation, reward, terminated, truncated, info


    def render(self):
        title = ""
      
        if self.actualActionType == NOTYPE_ACTION or self.actualAction == -1:
            title = "Step " + str(self.mystep) + ": Initial state"
        if self.actualActionType == ATTACK_ACTION or self.actualAction == -2:
            title = "Step " + str(self.mystep) + ": Attack"
        
        if self.actualActionType == DEFENSE_ACTION:
            if self.actualAction < len(self.topology) and self.actualAction >= 0:
                title = "Step " + str(self.mystep) + ": Isolate and patch node " + self.actionSpace[self.actualAction]
                #title = "Isolate node " + self.actionSpace[self.actualAction]
            elif self.actualAction >= len(self.topology) and self.actualAction < len(self.topology) + 3 and self.actualAction > 0:
                title = "Step " + str(self.mystep) + ": Migrate crtitical server to node " + self.actionSpace[self.actualAction]
            elif self.actualAction >= len(self.topology) + 3 and self.actualAction < len(self.topology) + 4 and self.actualAction > 0:
                title = "Step " + str(self.mystep) + ": Block traffic"
#            elif self.actualAction == -1:
#               title = "Step " + str(self.mystep) + ": Initial state"
#            elif self.actualAction == -2:
#               title = "Step " + str(self.mystep) + ": Attack"
            else:
                title = "Step " + str(self.mystep) + ": Do nothing"
        plt.title(title)
        box_textstr = 'Data Ex = ' + str (self.data_ex)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(-1,-1, box_textstr, fontsize=12, verticalalignment='top', bbox=props)
        color_map = []
        for node in self.netgraph:
            if [1, node] in self.nwstate:
                color_map.append("red")
            else:
                color_map.append("green")
        nx.draw(self.netgraph, pos=self.pos, node_color=color_map, font_size=14, font_weight="bold", with_labels=True)
        plt.axis('off')
        plt.draw()
        #path = "/home/ubuntu/"+str(self.step)+".png"
        #plt.savefig(path)
        plt.pause(4.5)
        plt.clf()

    # helper function for older versions of gymnasium
    def get_wrapper_attr(self, name: str):
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)


