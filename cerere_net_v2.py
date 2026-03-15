import random
import pygame
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import json

import functools
import time
import numpy as np
import gymnasium
from gymnasium import spaces


from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector
from pettingzoo.utils.conversions import parallel_wrapper_fn

sys.path.append("rlearn/rlearn/")

import attacker as attacker
import defender as defender
import network as network

NOTYPE_ACTION = 0
ATTACK_ACTION = 1
DEFENSE_ACTION = 2

__all__ = [
    "env",
    "parallel_env",
    "cerere_net_v2_env",
    "hmarl_env",
    "cerere_hmarl_env",
]


def env(**kwargs):
    env = cerere_net_v2_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# Hierarchical MARL PettingZoo env variant
def hmarl_env(**kwargs):
    env = cerere_hmarl_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)
hmarl_parallel_env = parallel_wrapper_fn(hmarl_env)


def _make_base_env(render_mode: str | None, rw_func: int | None, scenario: str | None):
    return cerere_net_v2_env(render_mode=render_mode, rw_func=rw_func, scenario=scenario)


# Load enterprise infection configs from JSON.
def _load_enterprise_config_set(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    p = os.path.abspath(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format A
    if isinstance(data, dict) and "configs" in data and isinstance(data["configs"], dict):
        out: dict[str, list[str]] = {}
        for k, v in data["configs"].items():
            if isinstance(v, dict) and "infected_nodes" in v:
                out[str(k)] = list(v["infected_nodes"])
        return out

    # Format B
    if isinstance(data, dict):
        out2: dict[str, list[str]] = {}
        for k, v in data.items():
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                out2[str(k)] = list(v)
        return out2

    raise ValueError(f"Unsupported enterprise config-set format in {p}")


# HMARL wiring layer around the existing CERERE environment
class cerere_hmarl_env(AECEnv):

    metadata = {
        "render_modes": ["human"],
        "name": "cerere_hmarl_env",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        rw_func: int | None = 1,
        scenario: str | None = "military",
        enterprise_config_set: str | None = None,
        num_skills: int = 4,
    ):
        super().__init__()
        self.render_mode = render_mode

        self.base_env = cerere_net_v2_env(
            render_mode=None,
            rw_func=rw_func,
            scenario=scenario,
            enterprise_config_set=enterprise_config_set,
        )

        # HMARL agents
        self.manager_id = "manager"
        self.worker_ids = [f"worker_{i}" for i in range(4)]
        self.worker_mig_id = "worker_mig"
        self.possible_agents = [self.manager_id] + self.worker_ids + [self.worker_mig_id]
        self.agents = []

        # Manager selects from a valid set of (skill, worker) combinations        #
        self.num_skills = int(num_skills)
        self._manager_action_meanings: list[tuple[int, str]] = []
        self._manager_action_meanings += [(0, wid) for wid in self.worker_ids]
        self._manager_action_meanings += [(1, self.worker_mig_id)]
        self._manager_action_meanings += [(2, self.worker_mig_id)]
        self._manager_action_meanings += [(3, self.worker_mig_id)]
        self._manager_action_space = spaces.Discrete(len(self._manager_action_meanings))

        # All workers output a primitive action index
        self._worker_action_space = spaces.Discrete(len(self.base_env.actionSpace))

        self._action_spaces = {self.manager_id: self._manager_action_space}
        for wid in self.worker_ids + [self.worker_mig_id]:
            self._action_spaces[wid] = self._worker_action_space

        # manager obs is a plain Box(flatState)
        # workers obs is Dict({"observations": flatState, "action_mask": mask})
        base_box = spaces.Box(
            0, 1, shape=(len(self.base_env.flatState),), dtype=np.float32
        )
        mask_box = spaces.Box(
            0.0,
            1.0,
            shape=(self._worker_action_space.n,),
            dtype=np.float32,
        )
        worker_dict = spaces.Dict({"observations": base_box, "action_mask": mask_box})
        self._observation_spaces = {self.manager_id: base_box}
        for wid in self.worker_ids + [self.worker_mig_id]:
            self._observation_spaces[wid] = worker_dict

        # Internal HMARL control state
        self._selected_worker: str | None = None
        self._selected_skill: int | None = None

        self._subnets: list[list[str]] = self._compute_subnets_deterministic(
            self.base_env.netgraph
        )
        self._worker_subnet_nodes: dict[str, set[str]] = {
            wid: set(self._subnets[i]) for i, wid in enumerate(self.worker_ids)
        }
        self._worker_subnet_nodes[self.worker_mig_id] = set(self.base_env.topology.keys())

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def render(self):
        if self.render_mode != "human":
            return
        self.base_env.render_mode = "human"
        self.base_env.render()
        self.base_env.render_mode = None

    def observe(self, agent):
        obs = self.observations[agent]
        # Dict obs for workers (action masking)
        if isinstance(obs, dict):
            return {
                "observations": np.array(obs["observations"], dtype=np.float32),
                "action_mask": np.array(obs["action_mask"], dtype=np.float32),
            }
        return np.array(obs, dtype=np.float32)

    def close(self):
        self.base_env.close()

    def reset(self, seed=None, options=None):
        self.base_env.reset(seed=seed, options=options)

        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        base_info = {}
        try:
            base_info = dict(self.base_env.infos.get("player_0", {}))
        except Exception:
            base_info = {}
        self.infos = {agent: dict(base_info) for agent in self.agents}

        # Recompute subnets after reset
        self._subnets = self._compute_subnets_deterministic(self.base_env.netgraph)
        self._worker_subnet_nodes = {
            wid: set(self._subnets[i]) for i, wid in enumerate(self.worker_ids)
        }
        self._worker_subnet_nodes[self.worker_mig_id] = set(self.base_env.topology.keys())

        # Reset HMARL control variables
        self._selected_worker = None
        self._selected_skill = None

        # Set observations
        base_obs = np.array(self.base_env.flatState, dtype=np.float32)
        self.observations = {}
        self.observations[self.manager_id] = base_obs
        for wid in self.worker_ids + [self.worker_mig_id]:
            self.observations[wid] = {
                "observations": base_obs,
                "action_mask": self._compute_action_mask(wid),
            }
        self.state = {agent: None for agent in self.agents}

    def _sync_obs_from_base(self):
        base_obs = np.array(self.base_env.flatState, dtype=np.float32)
        self.observations[self.manager_id] = base_obs
        for wid in self.worker_ids + [self.worker_mig_id]:
            self.observations[wid] = {
                "observations": base_obs,
                "action_mask": self._compute_action_mask(wid),
            }

    def _compute_subnets_deterministic(self, g: nx.Graph) -> list[list[str]]:
        buckets: dict[str, list[str]] = {"s1": [], "s2": [], "s3": [], "s4": []}
        for n in sorted(g.nodes()):
            if isinstance(n, str) and n.startswith("s") and "_" in n:
                pref = n.split("_", 1)[0]
                if pref in buckets:
                    buckets[pref].append(n)
        subnets = [buckets["s1"], buckets["s2"], buckets["s3"], buckets["s4"]]

        if sum(len(s) for s in subnets) == 0:
            nodes = [n for n in sorted(g.nodes()) if isinstance(n, str) and n.startswith("s")]
            subnets = [nodes[i::4] for i in range(4)]
        return subnets

    def _compute_action_mask(self, worker_id: str) -> np.ndarray:
        n_actions = self._worker_action_space.n
        mask = np.zeros((n_actions,), dtype=np.float32)

        # Non-selected workers are forced to no-op
        if self._selected_worker is None or worker_id != self._selected_worker:
            mask[n_actions - 1] = 1.0
            return mask

        topo_len = len(self.base_env.topology)
        skill = int(self._selected_skill) if self._selected_skill is not None else 3

        if skill == 0:
            # Patch indices are [0 .. topo_len-1]
            allowed_nodes = self._worker_subnet_nodes.get(worker_id, set())
            for i, node in enumerate(self.base_env.actionSpace[:topo_len]):
                if node in allowed_nodes:
                    mask[i] = 1.0
        elif skill == 1:
            if worker_id == self.worker_mig_id:
                for i in range(topo_len, min(topo_len + 3, n_actions)):
                    mask[i] = 1.0
        elif skill == 2:
            idx = topo_len + 3
            if 0 <= idx < n_actions:
                mask[idx] = 1.0
        else:
            idx = n_actions - 1
            mask[idx] = 1.0

        # If everything is masked out, allow no-op
        if mask.sum() == 0:
            mask[n_actions - 1] = 1.0
        return mask

    def step(self, action):
        # If already done for this agent, do dead step
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self.state[agent] = action
        self._cumulative_rewards[agent] = 0.0

        # Default: no rewards unless we trigger a real transition
        self._clear_rewards()

        if agent == self.manager_id:
            skill, worker = self._manager_action_meanings[int(action)]
            self._selected_skill = int(skill)
            self._selected_worker = worker

            # Update worker masks for the upcoming worker steps
            self._sync_obs_from_base()

            # No environment dynamics on manager step.
            self.rewards[agent] = 0.0

        else:
            # Worker steps
            if self._selected_worker is None:
                # If manager hasn't acted yet, ignore
                self.rewards[agent] = 0.0
            elif agent != self._selected_worker:
                # Non-selected worker: no-op
                self.rewards[agent] = 0.0
            else:
                mask = self._compute_action_mask(agent)
                if int(action) < 0 or int(action) >= len(mask) or mask[int(action)] < 0.5:
                    action = self._worker_action_space.n - 1

                self.base_env.agent_selection = "player_0"
                self.base_env.step(self.base_env.action_space("player_0").sample())
                self.base_env.agent_selection = "player_1"
                self.base_env.step(int(action))

                defender_reward = float(self.base_env.rewards.get("player_1", 0.0))

                # Only manager + selected worker receive the reward
                # Non-selected workers get 0.0 to avoid training on unrelated signal
                self.rewards[self.manager_id] = defender_reward
                self.rewards[agent] = defender_reward

                # Sync observation and done flags
                self._sync_obs_from_base()
                done = any(self.base_env.truncations.values()) or any(
                    self.base_env.terminations.values()
                )
                if done:
                    self.truncations = {aid: True for aid in self.agents}

        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()


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


    def __init__(
        self,
        render_mode: str | None = None,
        rw_func: int | None = 1,
        scenario: str | None = 'military',
        enterprise_config_set: str | None = None,
    ):
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode
        """
        super().__init__()
        #print("#### Beginn INIT new topology ###")
        if scenario == 'enterprise':
            # Multiple fixed initial compromise patterns (loaded from file when provided)
            loaded = _load_enterprise_config_set(enterprise_config_set)
            if loaded:
                self.infection_configs = loaded
                self.enterprise_config_set_path = os.path.abspath(enterprise_config_set)
            else:
                # Fallback to the default patterns (kept for backwards compatibility)
                self.infection_configs = {
                    "C1": ["s3_10", "s1_7", "s2_9"],
                    # concentrated in one area (mostly s1/s2)
                    #"C1": ["s1_7", "s1_5", "s2_9"], 
                    # spread across multiple subnets (s1/s3/s4)
                    "C2": ["s1_7", "s3_10", "s4_11"],
                    # more dangerous (closer to critical server path via s3)
                    "C3": ["s3_6", "s3_7", "s3_9"],
                }
                self.enterprise_config_set_path = None
            self.config_keys = list(self.infection_configs.keys())
            #path2topo = "/home/ubuntu/src/rl-test/rlearn/graphs/topo_generic.csv"
            #path2pos = "/home/ubuntu/src/rl-test/rlearn/graphs/pos_generic.csv" 
            path2topo = os.getcwd() + "/rlearn/graphs/topo_generic.csv"
            path2pos = os.getcwd() + "/rlearn/graphs/pos_generic.csv"
            self.init_critserver = self.critserver = "d3_1"
            self.init_optserver = self.optserver = ["d3_2", "d4_1"]
            self.block_traffic = 0
            # Pre-build topologies once
            self.topologies_by_config = {
                k: network.getTopologyFromCsv2(path2topo, infected)
                for k, infected in self.infection_configs.items()
            }
            # Default (will be overwritten on first reset)
            self.selected_config_key = self.config_keys[0]
            #print(self.selected_config_key)
            self.topology = self.topologies_by_config[self.selected_config_key]
        if scenario == 'military':
            #infected_nodes = ["ac-m-2-4"]
            self.infection_configs = {
                "C1": ["ac-m-2-4"],
            }
            self.config_keys = list(self.infection_configs.keys())
            path2topo = os.getcwd() + "/rlearn/graphs/topo_cerere_multi.csv"
            path2pos = os.getcwd() + "/rlearn/graphs/pos_cerere.csv"
            self.init_critserver = self.critserver = "as-hq"
            self.init_optserver = self.optserver = ["as-hqa", "as-hqa2"] # ["ash_hqa"]  # what is ash_hqa?
            self.block_traffic = 0
# Pre-build topologies once
            self.topologies_by_config = {
                k: network.getTopologyFromCsv2(path2topo, infected)
                for k, infected in self.infection_configs.items()
            }
# Default (will be overwritten on first reset)
            self.selected_config_key = self.config_keys[0]
            #print(self.selected_config_key)
            self.topology = self.topologies_by_config[self.selected_config_key]

        self.net = None  # mininet is not used
        self.mode = "none"  # net type wifi,lan and none in case of no mininet
        self.rw_function = rw_func
        # 0 = All attackers can infect one single node in the neighbourhood
        # 1 = All attacker infect all nodes in neighbourhood
        # 2 = One attacker infects one node in the neighbourhood
        self.attackmode = 0
        ##########if scenario != 'enterprise':
        ##########    self.topology = network.getTopologyFromCsv2(path2topo, infected_nodes)
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
        ###---self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, 1, self.mode, self.attackmode)
        self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
        all_reachable_nodes, reachable_healthy_nodes, reachable_infected_nodes, healthy_nodes_no_infected_subg, self.data_ex = network.getNodeStatistic(self.critserver, self.optserver, self.topology, self.nwstate, self.netgraph, self.block_traffic)
        # print(self.state)
   
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

        # Sample one of the patterns on reset (only for enterprise yet)
        if hasattr(self, "topologies_by_config"):
            forced = None
            if isinstance(options, dict):
                forced = options.get("config_key") or options.get("config")
            if forced is not None:
                if forced not in self.topologies_by_config:
                    raise ValueError(
                        f"Unknown enterprise config_key={forced}. Allowed: {list(self.topologies_by_config.keys())}"
                    )
                self.selected_config_key = forced
            else:
                if seed is not None:
                    random.seed(seed)
                self.selected_config_key = random.choice(self.config_keys)
            self.topology = self.topologies_by_config[self.selected_config_key]
        self.netgraph = nx.Graph()
        self.netgraph = network.createNetwork(self.net, self.netgraph, self.topology, self.mode)
        self.nwstate, self.netgraph = network.resetNetwork(self.topology, self.net, self.netgraph, self.mode) # Reset Network
        self._init()
        ##### not needed # observation = self.observe()
        ##### not needed # info = self._get_info()

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        cfg_info = {}
        if hasattr(self, "selected_config_key"):
            cfg_info = {
                "config_key": self.selected_config_key,
                "config_id": self.config_keys.index(self.selected_config_key),
            }
            if getattr(self, "enterprise_config_set_path", None):
                cfg_info["config_set"] = self.enterprise_config_set_path
        self.infos = {agent: dict(cfg_info) for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        #self.observations = {agent: NONE for agent in self.agents}
        self.observations = {agent: np.array(self.flatState, dtype=np.float32)  for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = AgentSelector(self.agents)
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
        ###print("AGENT %s TRY do %d" % (str(self.agent_selection), action))

        if self._agent_selector.is_first():
            #print("AGENT %s do %d" % (str(self.agent_selection), action))
            self.actualAction = action
            self.actualActionType = ATTACK_ACTION
            self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, action, self.mode, self.attackmode)
            self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
            if self.render_mode == "human":
                self.render()
            for i in self.agents:
                self.observations[i] =np.array(self.flatState, dtype=np.float32)

        if self._agent_selector.is_last():
            #print("AGENT %s do %d" % (str(self.agent_selection), action))
            #print(self.nwstate)
            self.mystep = self.mystep + 1
            self.actualAction = action
            self.actualActionType = DEFENSE_ACTION
            self.nwstate, self.netgraph, pFlag, self.critserver, self.optserver, self.block_traffic = defender.getAction(self.net, self.netgraph,self.critserver, self.optserver, action, self.actionSpace, self.topology, self.nwstate, self.block_traffic)
        
            ###---if self.render_mode == "human":
            ###---    self.render()
        
            ###---self.nwstate = attacker.attack(self.net, self.netgraph, self.nwstate, self.critserver, 1,  self.mode, self.attackmode)
            # reward function
            #reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes = network.getReward(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph)
            if self.rw_function == 2:
                reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward3(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
            else:
                reward, terminated2, reachable_healthy_nodes, reachable_infected_nodes, self.data_ex = network.getReward2(self.critserver, self.optserver, self.topology, self.nwstate, pFlag, self.netgraph, action, self.actionSpace, self.block_traffic)
            ###--self.actualAction = -2
            ###---self.actualActionType = ATTACK_ACTION
            #print(self.nwstate)
            self.flatState = network.getVectorFromState2(self.nwstate, self.critserver, self.netgraph)
 
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
