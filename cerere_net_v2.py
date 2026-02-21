import random
import pygame
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys

import functools
import time
import numpy as np
import gymnasium
from gymnasium import spaces


from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

sys.path.append("rlearn/rlearn/")

import attacker as attacker
import defender as defender
import network as network

NOTYPE_ACTION = 0
ATTACK_ACTION = 1
DEFENSE_ACTION = 2

__all__ = ["env", "parallel_env", "cerere_net_v2_env"]


def env(**kwargs):
    env = cerere_net_v2_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class cerere_net_v2_env(AECEnv):
    """
    The metadata holds environment constants. From gymnasium, we inherit the "render_modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "cerere_net_v2_env",
#        "name": "pettingzoo_cerere",
        "is_parallelizable": True
    }


    def __init__(self, render_mode: str | None = None, rw_func: int | None = 1, scenario: str | None = 'military'):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode
        """
        super().__init__()
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
            path2topo = os.getcwd() + "/rlearn/graphs/topo_cerere_multi.csv"
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
        print("Init: Action space (len {}) \n{}".format(len(self.actionSpace), self.actionSpace))
        self.netgraph = nx.Graph()
        self.netgraph = network.createNetwork(self.net, self.netgraph, self.topology, self.mode)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        #print("Size state space %d, Size action space %d" % (len(self.state), len(self.actionSpace)))
        #print("#### End INIT new topology ###")

        # Encode the network state as an adjacency matrix
        # self.observation_space = spaces.Box(0, 4, shape = (11,11), dtype=int)
        # Encode the network state as an adjacency vector
        #self.observation_space = spaces.Box(0, 4, shape=(11,), dtype=int)
        ####  replace # self.observation_space = spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32)
        #self.observation_state = np.zeros((self.size, self.size, self.size), dtype=np.float32)
        np.set_printoptions(threshold=sys.maxsize)
        ##### replace # self.observation_state = np.array(self.flatState, dtype=np.float32)
        ##### replace # self.action_space = spaces.Discrete(len(self.actionSpace))
        self.actualActionType = NOTYPE_ACTION
        self.actualAction = -1
        self.data_ex = 0
        self.mystep = 0

        #print("Number of nodes is %d" % self.netgraph.number_of_nodes())
        self.possible_agents = ["player_" + str(r) for r in range(2)]
        #self.possible_agents = ["player_" + str(r) for r in range(1)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        #self._action_spaces = {agent: spaces.Discrete(len(self.actionSpace)) for agent in self.possible_agents}
        self._action_spaces = {self.possible_agents[0]:spaces.Discrete(2), self.possible_agents[1]:spaces.Discrete(len(self.actionSpace))}
        
        #self._observation_spaces = {
        #    agent: spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32) for agent in self.possible_agents
        #}
        self._observation_spaces = {
            agent: spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32) for agent in self.possible_agents
        }

        self.observation_spaces = dict(
                zip(
                    self.possible_agents,
                    [spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32) for agent in self.possible_agents],
                )
       )

        self.render_mode = render_mode
        print("Render mode is %s" % self.render_mode)

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return spaces.Box(0, 1, shape=(len(self.flatState),), dtype=np.float32)
        #return self.observation_spaces[agent]

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(self.actionSpace))
        #return self.action_spaces[agent]

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn( "You are calling render method without specifying any render mode.")
            return

        title = ""
        
        if self.actualActionType == NOTYPE_ACTION or self.actualAction == -1:
            title = "Step " + str(self.mystep) + ": Initial state"
        if self.actualActionType == ATTACK_ACTION:
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

        print(title)
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
    
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass


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
        ####-----self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, self.mode, self.attackmode)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, self.data_ex = network.getNodeStatistic(self.critserver, self.optserver, self.topology, self.nwstate, self.netgraph, self.block_traffic)
        # print(self.state)
        ##### replace # self.observation_state = np.array(self.flatState, dtype=np.float32)
        
        # print(self.observation_state)
        self.actualActionType = NOTYPE_ACTION
        self.actualAction = -1


    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        ##### not needed # super().reset(seed=seed)
        self.data_ex = 0
        self.mystep = 0
        self.critserver = self.init_critserver
        self.optserver = self.init_optserver
        self.block_traffic = 0
        self.nwstate, self.netgraph = network.resetNetwork(self.topology, self.net, self.netgraph, self.mode) # Reset Network
        self._init()
        ##### not needed # observation = self.observe()
        ##### not needed # info = self._get_info()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        #self.observations = {agent: NONE for agent in self.agents}
        self.observations = {agent: np.array(self.flatState, dtype=np.float32)  for agent in self.agents}
        self.num_moves = 0
        # Our agent_selector utility allows easy cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def step(self, action):
        
        reward = 0
        #print("#### Begin STEP in pettingzoo ###")
        #print(self.nwstate)

        if ( self.terminations[self.agent_selection] or self.truncations[self.agent_selection] ):
            self._was_dead_step(action)
            return

        self.state[self.agent_selection] = action
        self._cumulative_rewards[self.agent_selection] = 0
        #print("AGENT %s TRY do %d" % (str(self.agent_selection), action))

        if self._agent_selector.is_first():
            print("AGENT %s do %d" % (str(self.agent_selection), action))
            self.actualAction = action
            self.actualAction = -2
            self.actualActionType = ATTACK_ACTION
            self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, action, self.mode, self.attackmode)
            self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
            if self.render_mode == "human":
                self.render()
            for i in self.agents:
                self.observations[i] =np.array(self.flatState, dtype=np.float32)

        if self._agent_selector.is_last():
            print("AGENT %s do %d" % (str(self.agent_selection), action))
            #print(self.nwstate)
            self.mystep = self.mystep + 1
            self.actualAction = action
            self.actualActionType = DEFENSE_ACTION
            self.nwstate, self.netgraph, pFlag, self.critserver, self.optserver, self.block_traffic = defender.getAction(self.net, self.netgraph,self.critserver, self.optserver, action, self.actionSpace, self.topology, self.nwstate, self.block_traffic)
        
            ####---if self.render_mode == "human":
            ####---    self.render()
        
            ####---self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, self.mode, self.attackmode)
            # reward function
            #reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes = network.getReward(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph)
            if self.rw_function == 2:
                reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward3(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
            else:
                reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward2(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
            ####--self.actualAction = -2
            #print(self.nwstate)
            self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
            ##### replace # self.observation_state = np.array(self.flatState, dtype=np.float32)
 
            if self.render_mode == "human":
               self.render()

            for i in self.agents:
                self.observations[i] =np.array(self.flatState, dtype=np.float32)


            ##### not needed # observation = self.observe()
            ##### not needed # info = self._get_info()
            if terminated2 == 1:
               #self.truncations[self.agent_selection] = True
               #my_terminated = True
               self.truncations = {
                  agent: True for agent in self.agents
            }


        self.rewards[self.agent_selection] = reward
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()
        #print("#### End STEP in pettingzoo ###")
        #print(self.nwstate)
        #print("Rewards = {} Accumulate_rewards = {}".format(self.rewards, self._cumulative_rewards))



