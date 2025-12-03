import functools
import time
import argparse
import os
import random
import datetime
import sys
import time
import gymnasium
import numpy as np
from gymnasium.spaces import Discrete

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

import cerere_net_v2

MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
#Papier schlägt Stein
#Stein schlägt Schere
#Schere schlägt Papier




def train_model(in_render_mode, in_steps, in_scenario, path2tar, in_rwf):

    env_kwargs = dict(render_mode=in_render_mode,rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.parallel_env(**env_kwargs)   			  
    env.reset(seed=42)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    print(f"Starting training on {str(path2tar)}.")

    tmp_path = "./tb_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=in_steps)
    model.save(path2tar)
   
    print(f"Finished training on {str(path2tar)}")
    env.close()

"""
def eval(render_mode="human"):		   			
    env_kwargs = dict(render_mode="human", max_cycles=10)	
    			
    ## Using the env defined
    #env = rps_v2.env(render_mode="human", max_cycles=10)
    env = rps_aur.env(**env_kwargs)

    #model = PPO.load("rps_v2.zip")
    model = PPO.load("rps_aur.zip")

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("########### Game %d" % i)
        env.reset(seed=42)

        for agent in env.agent_iter():
            # gibt die letzte Aktion des Gegners aus
            observation, reward, termination, truncation, info = env.last()
            print("Agent {} call last state {}".format(str(agent), MOVES[observation]))
            #print("Acting agent %s" % str(agent))
            #print("Programm termination %s, truncation %s"% (str(termination), str(truncation)))

#            for a in env.agents:
#                myrewards[a] += env.rewards[a]
#                print("Agent %s, Reward %f" % (str(a), env.rewards[a]))

    
            if termination or truncation:
                action = None
            else:
                if agent == env.possible_agents[0]:
                    #print("Rand agent %s" % str(agent))
                    #print("Rand agent")
                    action = env.action_space(agent).sample()
                    print("Agent {} (rand) selects {}".format(str(agent), MOVES[action]))
                else: 
                    #print("Model agent %s" % str(agent))
                    #print("Traind agent {} use state {}".format(str(agent), MOVES[observation]))
                    action = model.predict(observation, deterministic=True)[0]
                    print("Agent {} (train) selects {}".format(str(agent), MOVES[action]))

                # this is where you would insert your policy
                #action = env.action_space(agent).sample()
            env.step(action)
            if str(agent) == "player_1":
                for a in env.agents:
                    myrewards[a] += env.rewards[a]
                    print("Agent %s, Reward %f" % (str(a), env.rewards[a]))
                print("####################################### Both agents have played")


    myavg_reward = sum(myrewards.values()) / len(myrewards.values())
    print("Rewards: ", myrewards)
    print(f"Avg reward: {myavg_reward}")
"""

def eval_model(in_render_mode, in_scenario, path2tar):	

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)

    model = PPO.load(path2tar)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("######################### Eval Round %d ########################" % i)
        env.reset(seed=42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
                print("Agent %s do None" % (str(agent)))
            else:
                action = model.predict(observation, deterministic=True)[0]
                print("Agent %s do %d" % (str(agent), action))

            env.step(action)    
  
            if agent == env.possible_agents[-1]:
                for a in env.agents:
                    # quick fix
                    #myrewards[a] += env.rewards[a]
                    #print("Agent %s, Reward %f" % (str(a), env.rewards[a]))
                    print("Agent %s, Reward %f" % (str(a), 0.0))
                print("+++++++++++ All agents have played +++++++++++")  

    myavg_reward = sum(myrewards.values()) / len(myrewards.values())
    print("Rewards: ", myrewards)
    print(f"Avg reward: {myavg_reward}")
    env.close()


def eval_model_parallel(in_render_mode, in_scenario, path2tar):	

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)       			
    ## Using the env defined
    env = cerere_net_v2.parallel_env(**env_kwargs)

    model = PPO.load(path2tar)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("######################### Test Round %d ########################" % i)
        observation, info = env.reset(seed=42)

        while env.agents:
            # this is where you would insert your policy
            actions = {agent: model.predict(observation[agent], deterministic=True)[0] for agent in env.agents}
#           env.step(actions)
            observations, rewards, terminations, truncations, infos = env.step(actions)
#            print("Obersvation {}".format(observations))
            for a in env.agents:
                # quick fix
                #myrewards[a] += env.rewards[a]
                #print("Agent %s, Reward %f" % (str(a), env.rewards[a]))
                print("Agent %s, Reward %f" % (str(a), 0.0))
            print("+++++++++++ All agents have played +++++++++++") 

    myavg_reward = sum(myrewards.values()) / len(myrewards.values())
    print("Rewards: ", myrewards)
    print(f"Avg reward: {myavg_reward}")
    env.close()



def test(in_render_mode, in_scenario):	

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("######################### Test Round %d ########################" % i)
        env.reset(seed=42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
                print("Agent %s do None" % (str(agent)))
            else:
                action = env.action_space(agent).sample()
                if agent == env.possible_agents[0]:
                    #action = env.action_space(agent).sample()
                    print("AGENT %s do %d" % (str(agent), action))
                else: 
                    #action = env.action_space(agent).sample()
                    print("AGENT %s do %d" % (str(agent), action))         
            env.step(action)          

            if agent == env.possible_agents[-1]:
                for a in env.agents:
                    myrewards[a] += env.rewards[a]
                    print("Agent %s, Reward %f" % (str(a), env.rewards[a]))
                print("+++++++++++ Both agents have played +++++++++++")  

    myavg_reward = sum(myrewards.values()) / len(myrewards.values())
    print("Rewards: ", myrewards)
    print(f"Avg reward: {myavg_reward}")

    env.close()


def test_parallel(in_render_mode, in_scenario):	

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.parallel_env(**env_kwargs)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("######################### Test Round %d ########################" % i)
        observation, info = env.reset(seed=42)

        while env.agents:
            # this is where you would insert your policy
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            #env.step(actions)
            observations, rewards, terminations, truncations, infos = env.step(actions)
#            print("Obersvation {}".format(observations))
            for a in env.agents:
                myrewards[a] += env.rewards[a]
                print("Agent %s, Reward %f" % (str(a), env.rewards[a]))
            print("+++++++++++ All agents have played +++++++++++") 

    myavg_reward = sum(myrewards.values()) / len(myrewards.values())
    print("Rewards: ", myrewards)
    print(f"Avg reward: {myavg_reward}")
    env.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Test random action values in the spezified env", action="store_true")
    parser.add_argument('--eval', help="Eval a trained model in the specified env", action="store_true")
    parser.add_argument('--train', help="Train model in the specified env", action="store_true")
    parser.add_argument('--steps', type=int, default=20000,
                        help='Number of trainings steps (ent=100000/mil=20000) , default = 20000')
    parser.add_argument('--rwf', type=int, default=1,
                        help='Used reward function (iso-patch=1/bt=2) , default = 1')   
    parser.add_argument('--path2tar', type=str, default=os.getcwd() + 'targetmodel_ai_gym.pt',
                        help='Path to the neural network')
    parser.add_argument('--scen', type=str, default='none',
                        help='Scenario: [enterprise , military, none] , default = none')
    args = parser.parse_args()
    ENVIRONMENT = "Cerere"
    SCENARIO = args.scen

    if args.test:
        print("Start Test action values in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        test("human", SCENARIO)
        #test_parallel("human", SCENARIO)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop Test action values in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.eval:
        print("Eval model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        #eval_model("human", SCENARIO, args.path2tar)
        eval_model_parallel("human", SCENARIO, args.path2tar)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.train:
        print("Train model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model(None, args.steps, SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)




