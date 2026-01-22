import gymnasium
from gymnasium import envs
import gym_examples
import argparse
import numpy as np
import torch
#import torch.optim as optim
#import torch.nn as nn
import os
import random
import datetime
import sys
#import networkx as nx
import time
import json
from pathlib import Path
import re

# Ray RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import UnifiedLogger
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)

from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.utils.numpy import convert_to_numpy

import cerere_net_v2

#MOVES = ["ROCK", "PAPER", "SCISSORS", "None"]
#Papier schlägt Stein
#Stein schlägt Schere
#Schere schlägt Papier

# Custom logger function 
def custom_logger_creator(custom_path, custom_str=""):
    logdir = os.path.join(custom_path, custom_str)
    os.makedirs(logdir, exist_ok=True)
    
    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator


###### Test Begin
def train_model(in_render_mode, iterations, in_scenario, path2tar, in_rwf):

    env_kwargs = dict(render_mode=in_render_mode,rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.parallel_env(**env_kwargs)   			  
    env.reset(seed=42)

  
    print(f"Starting training on {str(path2tar)}.")

    register_env(
       "pettingzoo_cerere",
       lambda _: ParallelPettingZooEnv(env),
    )
    
    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
                
    # Set up logging directory
    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)
        
    config = (
        PPOConfig()
        .environment("pettingzoo_cerere")
        .resources(
              num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .training(
            train_batch_size = 1000,
            lr=0.0003,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff = 0.02,     
            vf_loss_coeff=0.25,
            kl_coeff = 0.005,  
         )
        .env_runners(
           env_to_module_connector=lambda env: (
                # `agent_ids=...`: Only flatten obs for the learning RLModule.
                FlattenObservations(multi_agent=True, agent_ids={"player_1"}),
            ),
           num_env_runners=0
        )
        .multi_agent(
            policies={"p0", "p1"},
            # `player_0` uses `p0`, `player_1` uses `p1`.
            policy_mapping_fn=lambda aid, episode: re.sub("^player_", "p", aid),
            policies_to_train=["p1"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(module_class=RandomRLModule),
                    "p1": RLModuleSpec(),
                }
            ),
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )
    #  Make PPO
    model = config.build()
    print("Print PPO model")
    print(model.get_config().model)
    #exit()         
           
    stop_rw = 0.83
    # Train the model
    print(f"Starting training for {iterations} iterations ...")
    for i in range(iterations):  
       result = model.train()
       print(f"Iter: {i}, Reward mean: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
       print(f"Iter: {i}, Reward min: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN]}")
       print(f"Iter {i}, Num steps: {result[NUM_ENV_STEPS_SAMPLED_LIFETIME]}")
       if result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] >= stop_rw: ##0.64 ent, 0.83 mil
           print(f"Reached episode return of {stop_rw} -> stopping ")
           print(result)
           # Save model
           checkpoint_path = model.save()
           print(f"Model saved at {checkpoint_path}")
           break
       #print(result)   
    print(f"Stop training after {i} iterations ...")
    env.close()   
###### Train Endegin  


###### Eval Begin
def eval_model(in_render_mode, in_scenario, path2tar, in_rwf):


    #env_kwargs = dict(render_mode=in_render_mode,rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
    #env = cerere_net_v2.parallel_env(**env_kwargs)   			  
    #env.reset(seed=42)

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)
  
    print(f"Starting evaluation on {str(path2tar)}.")

    register_env(
       "pettingzoo_cerere",
       lambda _: PettingZooEnv(env),
    )

    # Set up logging directory
    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    eval_config = (
        PPOConfig()
        .environment("pettingzoo_cerere")
        .resources(
              num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
        )
        .env_runners(
            num_env_runners=0
        )
        .multi_agent(
            policies={"p0", "p1"},
            # `player_0` uses `p0`, `player_1` uses `p1`.
            policy_mapping_fn=lambda aid, episode: re.sub("^player_", "p", aid),
            policies_to_train=["p1"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(module_class=RandomRLModule),
                    "p1": RLModuleSpec(),
                }
            ),
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .evaluation(
            evaluation_interval=9999999999,
#            evaluation_interval=1, # one eval per iteration
            evaluation_duration_unit="episodes",
            evaluation_duration=1,
            evaluation_config=PPOConfig.overrides(exploration=False),
        )
    )

    eval_algo = eval_config.build_algo()
    print("####################### Algo Beginn #########################")
    print(eval_algo)
    print("####################### Algo End    #########################")
    eval_algo.restore_from_path(path2tar)
    module_lpo = eval_algo.get_module("p1")
    weights = convert_to_numpy(next(iter(module_lpo.parameters())))
    np.set_printoptions(threshold=sys.maxsize)
    print(weights)
    results = eval_algo.evaluate()
    print("####################### Results Eval Beginn #########################")
    Reval = results['env_runners']['episode_return_mean']
    Leval = results['env_runners']['episode_len_mean']
    print(f" R(eval)={Reval}, L(eval)={Leval}", end="\n")
    print("####################### Results Eval End    #########################")
###### Eval End

###### Test Begin
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
                    print("AGENT %s attempted to do %d" % (str(agent), action))
                else: 
                    #action = env.action_space(agent).sample()
                    print("AGENT %s attempted do %d" % (str(agent), action))         
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
###### Test End


###### Test_parallel Begin
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
###### Test_parallel End


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Test random action values in the spezified env", action="store_true")
    parser.add_argument('--eval', help="Eval a trained model in the specified env", action="store_true")
    parser.add_argument('--train', help="Train model in the specified env", action="store_true")
    parser.add_argument('--iter', type=int, default=50000,
                        help='Number of trainings iterations (ent=100000/mil=100000) , default = 50000')
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
        eval_model_fnc("human", SCENARIO, args.path2tar, args.rwf)
         end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.train:
        print("Train model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model(None, args.iter, SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)




