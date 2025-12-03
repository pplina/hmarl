import gymnasium
from gymnasium import envs
import gym_examples
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
import datetime
import sys
import networkx as nx
import time
import json
from pathlib import Path

# Ray RLlib imports
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import UnifiedLogger
#from ray.tune.result import DEFAULT_RESULTS_DIR

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EPISODE_RETURN_MIN,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)


"""
sys.path.append("./rlearn/rlearn/")

"""


#ENVIRONMENT = "MountainCar"
ENVIRONMENT = "NEWCerere"

"""
def select_action(env, state):
    sample = random.random()
    threshold = 1
    if sample >= threshold:
        pass
        # select intelligent action according some policy
    else:
        return torch.tensor(env.action_space.sample(), device=device)
"""


# Custom logger function to replicate stable-baselines3 logger
def custom_logger_creator(custom_path, custom_str=""):
    logdir = os.path.join(custom_path, custom_str)
    os.makedirs(logdir, exist_ok=True)
    
    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator

###### Test Begin
def test(n_episodes, env_name, scenario_name, rwf):
    if env_name == "MountainCar":
        env = gymnasium.make('MountainCar-v0', render_mode="human")
        obs_space = env.observation_space
        action_space = env.action_space
        print("### MountainCar-v0, no different scenarios ###")
        print("Env observation space: {}".format(obs_space))  # velocity and position
        print("Env observation Upper Bound", env.observation_space.high)
        print("Env observation Lower Bound", env.observation_space.low)
        print("Action space: {}".format(action_space))

        obs, info = env.reset()
        print("The initial observation (position, velocity) is {}".format(obs))

        for step in range(n_episodes):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print("The new observation (position, velocity) is {}".format(obs))

            # Render the env
            env_screen = env.render()

            # Wait a bit before the next frame
            time.sleep(0.01)

            # If the episode is up, then start another one
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    if env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode="human", scenario=scenario_name)
        obs, info = env.reset()

        for step in range(n_episodes):
            action = env.action_space.sample()
            print("Action %d" % action)
            obs, reward, terminated, truncated, info = env.step(action)
            print(reward, info)
            if terminated:
                break
        env.close()
###### Test End

###### Eval Begin
def eval_model(env_name, scenario_name, path2tar, rwf):
    if env_name == "MountainCar":
        env = gymnasium.make('MountainCar-v0', render_mode="human")
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        #Register the environment with Ray
        register_env("MountainCar-v0", lambda config: env)
        
        # Load the trained model
        if "ppo" in path2tar.lower():
            model = PPO({ "env" : "MountainCar-v0", "evaluation_interval" : 9999999999, "evaluation_duration_unit" : "episodes",        "evaluation_duration" : 3, "explore" : False})
        else:
            model = DQN({ "env" : "MountainCar-v0", "evaluation_interval" : 9999999999, "evaluation_duration_unit" : "episodes",        "evaluation_duration" : 3, "explore" : False})
            
        model.restore(path2tar)
        
        print("Env observation space: {}".format(env.observation_space))
        print("Action space: {}".format(env.action_space))

        results = model.evaluate()
        print(f"Reward mean: {results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
        print(results)
        env.close()

    if env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode="human", rw_func=rwf, scenario=scenario_name)
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        #Register the environment with Ray
        register_env("CERERE-v0", lambda config: env)
            
        # Load the trained model
        if "ppo" in path2tar.lower():
            #model = PPO({ "env" : "CERERE-v0", "evaluation_interval" : 9999999999, "evaluation_duration_unit" : "episodes",        "evaluation_duration" : 5, "explore" : False})            
            config_eval = (
                PPOConfig()
                .environment(env="CERERE-v0")
                .training(
                    model={
                        "fcnet_hiddens": [256,256],
                        "fcnet_activation": "relu",
                        "vf_share_layers": True,
                    },
                 )
                .evaluation(
                    evaluation_interval=9999999999,
                    evaluation_duration_unit="episodes",
                    evaluation_duration=3,
                    evaluation_config={"explore": False},
                )        
            )
            model = config_eval.build_algo()

        else:
            #model = DQN({ "env" : "CERERE-v0", "evaluation_interval" : 9999999999, "evaluation_duration_unit" : "episodes",        "evaluation_duration" : 3, "explore" : False})           
            config_eval = (
                DQNConfig()
                .environment(env="CERERE-v0")
                .training(
                    model={
                        "fcnet_hiddens": [256,256],
                        "fcnet_activation": "relu",
                        "vf_share_layers": True,
                    },
                 )
                .evaluation(
                    evaluation_interval=9999999999,
                    evaluation_duration_unit="episodes",
                    evaluation_duration=3,
                    evaluation_config={"explore": False},
                )        
            )
            model = config_eval.build_algo()
            
        model.restore(path2tar)
        print("Print model")
        print(model.get_config().model)
        #exit()
        
        print("Env observation space: {}".format(env.observation_space))
        print("Action space: {}".format(env.action_space))
              
        results = model.evaluate()
        print(f"Reward mean: {results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
        print(results)
        env.close()
        
###### Eval End

###### Train Begin
def train_model(iterations, stop_rw, env_name, scenario_name, path2tar, rwf):
    if env_name == "MountainCar":
        env = gymnasium.make('MountainCar-v0', render_mode="rgb_array")
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Set up logging directory
        tmp_path = "./tb_log/"
        os.makedirs(tmp_path, exist_ok=True)
        # Configure DQN
        dqn_config = {
            "env": "MountainCar-v0",
            "framework": "torch",
            "num_gpus": int(torch.cuda.is_available()),
            "seed": 100,
            "num_cpus": 6,
            "num_workers": 1,
            "train_batch_size": 64,
            "model": {
                "fcnet_hiddens": [128],
                "fcnet_activation": "linear",
            },
            "gamma": 0.99,
            "lr": 0.0001,
            "epsilon" : [(0, 1.0), (100000, 0.02)],
            "replay_buffer_config": {
                "type": "PrioritizedEpisodeReplayBuffer", 
                "capacity": 50000, 
                "alpha": 0.6, 
                "beta": 0.4,
            },
            "num_training_step_calls_per_iteration": 64,
            "rollout_fragment_length": 200,
        }
        # train the model
        model = DQN(config=dqn_config, logger_creator=custom_logger_creator(tmp_path, "dqn_mountaincar"))
        print("Print DQN model")
        print(model.get_config().model)
        """
        config2 = (
             DQNConfig()
             .environment(env="MountainCar-v0")
#             .api_stack(
#                 enable_env_runner_and_connector_v2=False,
#                 enable_rl_module_and_learner=False,
#             )
             .training(
#                 num_epochs= 8,
                 train_batch_size = 1000,
                 lr=0.0005,
                 gamma=0.99,
                 replay_buffer_config = {
                     "type": "EpisodeReplayBuffer",
                     "capacity": 50000,
                 },

#                 replay_buffer_config={
#                     "type": "MultiAgentPrioritizedReplayBuffer",
#                     "prioritized_replay_alpha": 0.6, 
#                     "prioritized_replay_beta": 0.4, 
#                     "prioritized_replay_eps": 1e-6,
#                     "capacity": 50000,
#                     "alpha": 0.6,
#                     "beta": 0.4,
#                 },

                 model={
                        "fcnet_hiddens": [128],
                        "fcnet_activation": "linear",
                        "vf_share_layers": True,
                 },
                 epsilon=[(0, 1.0), (100000, 0.02)],
             )
             .env_runners(
                 num_env_runners=0
             )
             .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "dqn_mcar"))
             .framework(framework="torch")
             
        )
        model = config2.build_algo()
        print("Print DQN model")
        print(model.get_config().model)
        """
        # Train the model
        for i in range(iterations):  
            result = model.train()
            print(f"Iteration {i}, Num steps: {result[NUM_ENV_STEPS_SAMPLED_LIFETIME]}")
            print(f"Iteration {i}, Reward mean: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}")
            if result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN] >= -150:
                print(f"Reached episode return of -150 -> stopping ")
                print(result)
                break       
        
        # Save model
        checkpoint_path = model.save()
        print(f"Model saved at {checkpoint_path}")
        
        # Rename/move the checkpoint to the desired path
        if not os.path.exists(os.path.dirname(path2tar)):
            os.makedirs(os.path.dirname(path2tar), exist_ok=True)
        model.save(path2tar)

    if env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode=None, rw_func=rwf, scenario=scenario_name)
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Register the environment with Ray
        register_env("CERERE-v0", lambda config: env)
        
        # Set up logging directory
        tmp_path = "./tb_log/"
        os.makedirs(tmp_path, exist_ok=True)
        
        # Determine algorithm type from path name
        is_ppo = "ppo" in path2tar.lower()
        
        if is_ppo:
            # Configure PPO for the CERERE environment
            config = {
                "env": "CERERE-v0",
                "framework": "torch",
                "num_gpus": int(torch.cuda.is_available()),
                "num_workers": 1,
                "train_batch_size": 1000,  # Keep large batch for stable learning
                "sgd_minibatch_size": 256,  # Maintain good gradient estimates
                "lr": 2e-4,               # Start with higher learning rate
                "gamma": 0.99,
                "lambda": 0.97,            # GAE parameter
                "clip_param": 0.2,         # Standard PPO clip
                "entropy_coeff": 0.02,     # Keep high exploration
                "vf_loss_coeff": 0.75,     # Balance value estimation
                "kl_coeff": 0.005,         # Allow policy updates
                "num_sgd_iter": 8,         # Increased for better convergence
                "model": {
                    "fcnet_hiddens": [256,256],
                    "fcnet_activation": "relu",
                },
                "rollout_fragment_length": 200,
                "checkpoint_frequency" : 1000,
                "checkpoint_at_end" : True,
            }
            #  Make PPO
            model = PPO(config=config, logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
            """
            config2 = (
                PPOConfig()
                .environment(env="CERERE-v0")
                .resources(
                    num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0"))
                )
                .api_stack(
                    enable_env_runner_and_connector_v2=False,
                    enable_rl_module_and_learner=False,
                )
                .training(
                    num_epochs= 8,
                    train_batch_size = 500,
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
                .env_runners(
                    num_env_runners=0
                )
#                .rl_module(
#                    model_config=DefaultModelConfig(
#                    fcnet_hiddens=[256,256],
#                    fcnet_activation="relu",
#                    vf_share_layers=True,
#                    ),
#                 ) 
                .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
                .framework(framework="torch")
                .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
            )
            #  Make PPO
            model = config2.build_algo()
            """
            print("Print PPO model")
            print(model.get_config().model)           
        else:
            # Configure DQN for CERERE environment
            config = {
                "env": "CERERE-v0",
                "framework": "torch",
                "num_gpus": int(torch.cuda.is_available()),
                "num_workers": 1,
                "train_batch_size": 32,
                "num_steps_sampled_before_learning_starts": 1000, 
                "lr": 1e-3,
                "gamma": 0.99,
                "target_network_update_freq": 2000,
                "replay_buffer_config": {
                    "type": "PrioritizedEpisodeReplayBuffer",
                    "capacity": 100000,
                },
                "model": {
                    "fcnet_hiddens": [256,256],
                    "fcnet_activation": "relu",
                },
                "checkpoint_frequency" : 1000,
                "checkpoint_at_end" : True,
            }
            # Make DQN
            model = DQN(config=config, logger_creator=custom_logger_creator(tmp_path, "dqn_cerere"))
            print("Print DQN model")
            print(model.get_config().model)
            
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
               break
        #print(result)   

        # Save model
        checkpoint_path = model.save()
        print(f"Model saved at {checkpoint_path}")
        
        # Rename/move the checkpoint to the desired path
        if not os.path.exists(os.path.dirname(path2tar)):
            os.makedirs(os.path.dirname(path2tar), exist_ok=True)
        model.save(path2tar)

###### Train End


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Test random action values in the spezified env", action="store_true")
    parser.add_argument('--eval', help="Eval a trained model in the specified env", action="store_true")
    parser.add_argument('--train', help="Train model in the specified env", action="store_true")
    parser.add_argument('--iter', type=int, default=50000,
                        help='Number of trainings steps (ent=100000/mil=100000) , default = 50000')
    parser.add_argument('--stop_rw', type=float, default=0.1,
                        help='Mean reward to stop the training (ent=0.64 ent/ mil=0.83 mil), default = 0.1')
    parser.add_argument('--rwf', type=int, default=1,
                        help='Used reward function (iso-patch=1/bt=2) , default = 1')
    parser.add_argument('--path2tar', type=str, default=os.getcwd() + 'targetmodel_ai_gym.pt',
                        help='Path to the neural network')
    parser.add_argument('--env', type=str, default='NewCerere',
                        help='Environment: [MountainCar, NewCerere] , default = NewCerere')
    parser.add_argument('--scen', type=str, default='none',
                        help='Scenario: [enterprise , military, none] , default = none')
    args = parser.parse_args()
    ENVIRONMENT = args.env
    SCENARIO = args.scen
    path2tar = args.path2tar
    rwf= args.rwf

    if SCENARIO == 'enterprise' and rwf == 2:
        print("Not supported")
        exit(1)

    random.seed(24)

    if args.test:
        print("Start Test action values in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        test(200, ENVIRONMENT, SCENARIO, rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop Test action values in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.eval:
        print("Eval model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        eval_model(ENVIRONMENT, SCENARIO, path2tar, rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.train:
        print("Train model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model(args.iter, args.stop_rw, ENVIRONMENT, SCENARIO, path2tar, rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)
