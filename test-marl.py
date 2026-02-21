import gymnasium
from gymnasium.spaces import Discrete, Box
import argparse
import numpy as np
import torch
import random
import datetime
import sys
import time
import json
from pathlib import Path
import re
import os
import tree  # pip install dm_tree

# Ray RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
#from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
#from ray.rllib.examples.rl_modules.classes import (
#    AlwaysSameHeuristicRLM,
#    BeatLastHeuristicRLM,
#)
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

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.spaces.space_utils import batch as batch_func

torch, _ = try_import_torch()

import cerere_net_v2



class HeuristicAttackModule(RLModule):
    @override(RLModule)
    def _forward(self, batch, **kwargs):
        obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
#        print(f"Obs Batch size {obs_batch_size}")
#        actions = batch_func(
#            [self.action_space.sample() for _ in range(obs_batch_size)]
#        )
        actions = batch_func(
            [1 for _ in range(obs_batch_size)]
        )
        return {SampleBatch.ACTIONS: actions}

    @override(RLModule)
    def _forward_train(self, *args, **kwargs):
        # HeuristicAttackModule should always be configured as non-trainable.
        # To do so, set in your config:
        # `config.multi_agent(policies_to_train=[list of ModuleIDs to be trained,
        # NOT including the ModuleID of this RLModule])`
        raise NotImplementedError("HeuristicAttackModule: Should not be trained!")

    def compile(self, *args, **kwargs):
        """Dummy method for compatibility with TorchRLModule.

        This is hit when RolloutWorker tries to compile TorchRLModule."""
 


# Custom logger function 
def custom_logger_creator(custom_path, custom_str=""):
    logdir = os.path.join(custom_path, custom_str)
    os.makedirs(logdir, exist_ok=True)
    
    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator


###### Train Begin
def train_model(iterations, stop_rw, env_name, scenario_name, path2tar, rwf):

    print(f"Starting training on {str(path2tar)}.")

#    env_kwargs = dict(render_mode=None,rw_func=rwf, scenario=scenario_name)    			
#    ## Using the env defined
#    env = cerere_net_v2.parallel_env(**env_kwargs)   			  
#    #env.reset(seed=42)

#    register_env(
#       "pettingzoo_cerere",
#       lambda _: ParallelPettingZooEnv(env),
#    )

    env_kwargs = dict(render_mode=None,rw_func=rwf, scenario=scenario_name) 			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)
    env.reset(seed=42)

    register_env(
       "pettingzoo_cerere",
       lambda _: PettingZooEnv(env),
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
        .training(
            train_batch_size = 1000,
            lr=0.0003,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff = 0.02,     
            vf_loss_coeff=0.25,
            kl_coeff = 0.005,  
#            model={
#                "fcnet_hiddens": [256,256],
#                "fcnet_activation": "relu",
#                "vf_share_layers": True,
#            },
         )
        .env_runners(
#           env_to_module_connector=lambda env: (
                # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                FlattenObservations(multi_agent=True, agent_ids={"player_1"}),
#            ),
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
                    "p0": RLModuleSpec(module_class=HeuristicAttackModule),
                    "p1": RLModuleSpec(),
                }
            ),
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
        .framework(framework="torch")
    )
    #  Make PPO
    model = config.build_algo()
#    print("####################### Print Algo Beginn #########################")
#    print(model.get_config().model)
#    print("####################### Print Algo End    #########################")
#    exit()         
           
    # Train the model
    print(f"Starting training for {iterations} iterations ...")
    for i in range(iterations):  
       result = model.train()
       print(f"Iter: {i}, Reward mean: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
       print(f"Iter: {i}, Reward min: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN]}")
       print(f"Iter {i}, Num steps: {result[NUM_ENV_STEPS_SAMPLED_LIFETIME]}")
       #exit()
       if result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] >= stop_rw: ##0.64 ent, 0.83 mil
           print(f"Reached episode return of {stop_rw} -> stopping ")
           print("####################### Results Train Beginn #########################")
           print(result)
           print("####################### Results Train End    #########################")
           # Save model
           checkpoint_path = model.save(path2tar)
           print(f"Model saved at {checkpoint_path}")
           break
       #print(result)   
    print(f"Stop training after {i} iterations ...")
    env.close()   
###### Train Ende 


###### Train with Tune Begin
def train_model_with_tune(iterations, stop_rw, env_name, scenario_name, path2tar, rwf):
 
    env_kwargs = dict(render_mode=None,rw_func=rwf, scenario=scenario_name) 			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)
    env.reset(seed=42)

    register_env(
       "pettingzoo_cerere",
       lambda _: PettingZooEnv(env),
    ) 

    # Set up logging directory
    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    config = (
        PPOConfig()
        .environment("pettingzoo_cerere")
        .training(
            train_batch_size = 1000,
            lr=0.0003,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff = 0.02,     
            vf_loss_coeff=0.25,
            kl_coeff = 0.005,  
            model={
                   "fcnet_hiddens": [256,256],
                   "fcnet_activation": "relu",
                   "vf_share_layers": True,
               },
        )
#           .env_runners(
#           env_to_module_connector=lambda env: (
                # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                FlattenObservations(multi_agent=True, agent_ids={"player_1"}),
#            ),
#            num_env_runners=0
#           )
        .multi_agent(
             policies={"p0", "p1"},
             # `player_0` uses `p0`, `player_1` uses `p1`.
             policy_mapping_fn=lambda aid, episode: re.sub("^player_", "p", aid),
             policies_to_train=["p1"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(module_class=HeuristicAttackModule),
                    "p1": RLModuleSpec(	),
                }
            ),
        )
        .debugging(log_level="DEBUG", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
        .framework(framework="torch")
    )

    stop = {"num_env_steps_sampled_lifetime": iterations, "env_runners/episode_return_mean": stop_rw} ##0.64 ent, 0.83 mil

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(
            num_gpus=int(torch.cuda.is_available()),           
            num_cpus=6,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
    # execute training 
    analysis = ray.tune.run(
        "PPO",
        config=config,
        stop=stop,
        metric="env_runners/episode_return_mean",
        mode="max",
        checkpoint_at_end=True,
    )
    env.close()
    print("###### Analysis Train with Tune ######")
    #print(analysis.best_result)
    #print(analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max"))
    #print(analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max").checkpoint)
    #print("Model path %s" % (analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max").checkpoint.path))
    #print(analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max").config)
    path=analysis.get_best_trial(metric="env_runners/episode_return_mean", mode="max").checkpoint.path
    os.rename(path, path2tar)
 ###### Train with Tune Ende



###### Eval Begin --double env registration, reason unkonwn
def eval_model(in_render_mode, in_scenario, path2tar, in_rwf):

    print(f"Starting evaluation on {str(path2tar)}.")

#    env_kwargs = dict(render_mode=in_render_mode,rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
#    env = cerere_net_v2.parallel_env(**env_kwargs)   			  
    #env.reset(seed=42)

#    register_env(
#       "pettingzoo_cerere",
#       lambda _: ParallelPettingZooEnv(env),
#    )

    env_kwargs = dict(render_mode=in_render_mode, rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)

    register_env(
       "pettingzoo_cerere",
       lambda _: PettingZooEnv(env),
    )

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Set up logging directory
    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    eval_config = (
        PPOConfig()
        .environment("pettingzoo_cerere")
        .env_runners(
            num_env_runners=0,
#            env_to_module_connector=lambda env: (
#                # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                FlattenObservations(multi_agent=True, agent_ids={"player_0"}),
#            ),
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
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=9999999999,
#            evaluation_interval=1, # one eval per iteration
            evaluation_duration_unit="episodes",
            evaluation_duration=1,
            evaluation_config={
                "explore": False,
            },
        )
    )

    eval_algo = eval_config.build_algo()
#    print("####################### Print Algo Beginn #########################")
#    print(eval_algo)
#    print("####################### Print Algo End    #########################")
    eval_algo.restore_from_path(path2tar)
    results = eval_algo.evaluate()
    print("####################### Results Eval Beginn #########################")
    Reval = results['env_runners']['episode_return_mean']
    Leval = results['env_runners']['episode_len_mean']
    print(f" R(eval)={Reval}, L(eval)={Leval}", end="\n")
    print("####################### Results Eval End    #########################")
    env.close()
###### Eval End


###### Eval2 Begin
def eval_model2(in_render_mode, in_scenario, path2tar, in_rwf):

    env_kwargs = dict(render_mode=in_render_mode,rw_func=in_rwf, scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)

    register_env(
        "pettingzoo_cerere",
        lambda _: PettingZooEnv(env),
    )

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


    print("Restore RLModule from checkpoint ...", end="")
    rl_module = RLModule.from_checkpoint(
        os.path.join(
            path2tar,
            "learner_group",
            "learner",
            "rl_module",
            "p1",
        )
    )
    print(" ok")

    myrewards = {agent: 0 for agent in env.possible_agents}
    n_episodes = 1
    i = 0

    for i in range(n_episodes):
        i += 1
        print("######################### Test Round %d ########################" % i)
        env.reset(seed=42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            #print(f" Observations {observation}")

            if termination or truncation:
                action = None
                print("Agent %s do None" % (str(agent)))
            else:
                #action = env.action_space(agent).sample()
                if agent == env.possible_agents[0]:
                    #action = env.action_space(agent).sample()
                    action = 1
                    print("AGENT %s attempted to do %d" % (str(agent), action))
                else: 
                    #action = env.action_space(agent).sample()
                    input_dict = {Columns.OBS: torch.from_numpy(observation).unsqueeze(0)}
                    rl_module_out = rl_module.forward_inference(input_dict)
                    logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
                    # Perform the sampling step in numpy for simplicity.
                    # action = np.random.choice(env.action_space(agent).n, p=softmax(logits[0]))
                    # Select action without exploration
                    action = np.argmax(softmax(logits[0]))
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
###### Eval2 End


###### Test Begin
def test(in_render_mode, in_scenario):	

    env_kwargs = dict(render_mode=in_render_mode,scenario=in_scenario)    			
    ## Using the env defined
    env = cerere_net_v2.env(**env_kwargs)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_tests = 1
    i = 0

    for i in range(num_tests):
        i += 1
        print("######################### Test No %d ########################" % i)
        env.reset(seed=42)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
                print("Agent %s do None" % (str(agent)))
            else:
                if agent == env.possible_agents[0]:
                    #action = env.action_space(agent).sample()
                    action = 1
                    print("AGENT %s attempted to do %d" % (str(agent), action))
                else: 
                    action = env.action_space(agent).sample()
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
    parser.add_argument('--trainWithTune', help="Train with ray tune model in the specified env", action="store_true")
    parser.add_argument('--iter', type=int, default=50000,
                        help='Number of trainings iterations (ent=100000/mil=100000) , default = 50000')
    parser.add_argument('--stop_rw', type=float, default=0.1,
                        help='Mean reward to stop the training (ent=0.64 ent/ mil=0.83 mil), default = 0.1')
    parser.add_argument('--rwf', type=int, default=1,
                        help='Used reward function (iso-patch=1/bt=2) , default = 1')   
    parser.add_argument('--path2tar', type=str, default=os.getcwd() + 'targetmodel_ai_gym.pt',
                        help='Path to the neural network')
    parser.add_argument('--scen', type=str, default='none',
                        help='Scenario: [enterprise , military, none] , default = none')
    args = parser.parse_args()
    ENVIRONMENT = "NewCerere"
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
        #eval_model("human", SCENARIO, args.path2tar, args.rwf)
        eval_model2("human", SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.train:
        print("Train model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model(args.iter, args.stop_rw, ENVIRONMENT, SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.trainWithTune:
        print("Train model with Tune in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model_with_tune(args.iter, args.stop_rw, ENVIRONMENT, SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model with Tune in env %s after %s" % (ENVIRONMENT, elapsed))
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)




