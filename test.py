import gymnasium
from gymnasium import envs
import gym_examples
from stable_baselines3 import DQN as sb_DQN
from stable_baselines3 import PPO as sb_PPO
from stable_baselines3.common.logger import configure
#import mobile_env
#import gridworld
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import os
import random
import datetime
import sys
import networkx as nx
import time

sys.path.append("./rlearn/rlearn/")
from ddqn import (
    gamma_by_frame,
    epsilon_by_frame,
    beta_by_frame,
    DQN,
    update_target,
    NaivePrioritizedBuffer,
    ReplayBuffer,
    compute_td_loss
)
from config import DDQN_Config
#import emu_network as emun

# ENVIRONMENT = "GridWorld"
# ENVIRONMENT = "Mobile-Env"
#ENVIRONMENT = "MountainCar"
ENVIRONMENT = "Cerere"

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


###### Test Beginn
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

        # Number of steps you run the agent for
        #num_steps = 1500

        obs = env.reset()
        print("The initial observation (position, velocity) is {}".format(obs))

        #for step in range(num_steps):
        for step in range(n_episodes):
            # take random action, but you can also do something more intelligent
            # action = my_intelligent_agent_fn(obs)
            action = env.action_space.sample()
            # apply the action
            obs, reward, terminated, truncated, info = env.step(action)
            print("The new observation (position, velocity) is {}".format(obs))

            # Render the env
            env_screen = env.render()
            #plt.imshow(env_screen)

            # Wait a bit before the next frame unless you want to see a crazy fast video
            time.sleep(0.01)

            # If the episode is up, then start another one
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    #    if env_name == "GridWorld":
    #        env = gymnasium.make('gym-examples/GridWorld-v0', render_mode="human")
    #        observation, info = env.reset()
    #
    #        for _ in range(n_episodes):
    #            action = env.action_space.sample()  # agent policy that uses the observation and info
    #            observation, reward, terminated, truncated, info = env.step(action)
    #
    #            print(observation, reward, terminated, truncated, info)
    #
    #            if terminated or truncated:
    #                observation, info = env.reset()
    #        env.close()

    #    if env_name == "Mobile-Env":
    #        env = gymnasium.make("mobile-medium-central-v0", render_mode="human")
    #        obs, info = env.reset()
    #        done = False

    #        while not done:
    #            action = env.action_space.sample()
    #            obs, reward, terminated, truncated, info = env.step(action)
    #            done = terminated or truncated
    #            env.render()
    #        env.close()

    if env_name == "Cerere" or env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode="human", scenario=scenario_name)
        obs, info = env.reset()
        #print("Action space is %d" % len(env.get_wrapper_attr('actionSpace')))
        #print("Action spaces is %s" % env.action_space)

        #for _ in range(n_episodes):
        #action = 27 block traffic
        for step in range(n_episodes):
            #action +=1
            action = env.action_space.sample()
            print("Action %d" % action)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Observation after action is {}".format(obs))
            print(reward, info)
            if terminated:
                break
        env.close()
###### Test End


###### Eval Beginn
def eval_model(env_name, scenario_name, path2tar, rwf):
    if env_name == "MountainCar":
        #print("Eval of env MountainCar not supported")
        env = gymnasium.make('MountainCar-v0', render_mode="human")
        #model = sb_PPO.load(path2tar)
        model = sb_DQN.load(path2tar)
        #obs_space = env.observation_space
        #action_space = env.action_space
        #print("### MountainCar-v0, no different scenarios ###")
        print("Env observation space: {}".format(env.observation_space))  # velocity and position
        print("Env observation Upper Bound", env.observation_space.high)
        print("Env observation Lower Bound", env.observation_space.low)
        print("Action space: {}".format(env.action_space))

        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            print("Observation (position, velocity) is {}{}".format(obs, reward))
            if terminated or truncated:
                obs, info = env.reset()
                print("Success round %d" % step)
                return True
        return False

    if env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode="human", rw_func=rwf, scenario=scenario_name)
        model = sb_PPO.load(path2tar)
        # model = sb_DQN.load(path2tar)
        print("Env observation space: {}".format(env.observation_space))
        #print("Env observation Upper Bound", env.observation_space.high)
        #print("Env observation Lower Bound", env.observation_space.low)
        print("Action space: {}".format(env.action_space))
		## should create the mininet env
        #emun.create(env.get_wrapper_attr('topology'))
        """
        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            #print("Observation after action is {}".format(obs))
            if terminated or truncated:
                #obs, info = env.reset()
                print("Success round %d" % step)
                break;
                #return True

        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            #print("Observation after action is {}".format(obs))
            if terminated or truncated:
                #obs, info = env.reset()
                print("Success round %d" % step)
                break;
                #return True

        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            #print("Observation after action is {}".format(obs))
            if terminated or truncated:
                #obs, info = env.reset()
                print("Success round %d" % step)
                break;
                #return True

        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            #print("Observation after action is {}".format(obs))
            if terminated or truncated:
                #obs, info = env.reset()
                print("Success round %d" % step)
                break;
                #return True
        """
        obs, info = env.reset()
        step = 0
        while True:
            step = step + 1
            action, _states = model.predict(obs, deterministic=True)
            #print("Observation before action is {}".format(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            #print("Round %d: action %d " % (step, action))
            #print("Observation after action is {}".format(obs))
            if terminated or truncated:
                #obs, info = env.reset()
                print("Success round %d" % step)
                return True
        return False

    if env_name == "Cerere":
        episodereward = 0
        terminated = False
        states = []
        rewards = []
        actions = []

        USE_CUDA = torch.cuda.is_available()
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode="human", scenario=scenario_name)
        observation, info = env.reset()
        print("Action space is %d" % len(env.get_wrapper_attr('actionSpace')))
        print("Action spaces is %s" % env.action_space)

        currentmodel = torch.load(path2tar)
        currentmodel.eval()
        step = 0
        max_round = 25
        while not terminated and i < max_round:
            step = step + 1
            #print(env.get_wrapper_attr('flatState'))
            #print(len(env.get_wrapper_attr('actionSpace')))
            #print(env.get_wrapper_attr('actionSpace'))
            #print("Observation is {}".format(observation))
            #print("Flatstate is {}".format(env.get_wrapper_attr('flatState')))
            #action = currentmodel.act(env.get_wrapper_attr('flatState'), 0, len(env.get_wrapper_attr('actionSpace')), USE_CUDA)
            action = currentmodel.act(observation, 0, env.action_space, USE_CUDA)
            #print(action)
            observation, reward, terminated, truncated, info = env.step(action)
            #print(env.get_wrapper_attr('nwstate'))
            #print(observation, reward, terminated, truncated, info)
            print("Reward %f, Info %s" % (reward, info))
            states.append(observation)
            rewards.append(reward)
            actions.append(action + 1)
            episodereward += reward

        print("Received rewards: %s" % rewards)
        env.close()
###### Eval End


###### Train Beginn
def train_model(steps, env_name, scenario_name, path2tar, rwf):
    if env_name == "MountainCar":
        #print("Eval of env MountainCar not supported")
        #env = gymnasium.make('MountainCar-v0', render_mode="human")
        env = gymnasium.make('MountainCar-v0', render_mode="rgb_array")
        #model = sb_PPO("MlpPolicy", env, verbose=1)
        model = sb_DQN(
            "MlpPolicy",
            env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=0.98,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
        model.learn(total_timesteps=100000, log_interval=4)
        model.save(path2tar)

    if env_name == "NewCerere":
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode=None, rw_func=rwf, scenario=scenario_name)
        #observation, info = env.reset()
        tmp_path = "./tb_log/"
        # set up logger
        new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

        model = sb_PPO("MlpPolicy", env, verbose=1)

        # model = sb_DQN(
        # "MlpPolicy",
        # env,
        # verbose=1,
        # train_freq=16,
        # gradient_steps=8,
        # gamma=0.9,
        # exploration_fraction=0.2,
        # exploration_final_eps=0.01,
        # target_update_interval=600,
        # learning_starts=1000,
        # buffer_size=10000,
        # batch_size=128,
        # learning_rate=4e-3,
        # policy_kwargs=dict(net_arch=[256, 256]),
        # )
        #steps = 20000  # 100000
        model.set_logger(new_logger)
        model.learn(total_timesteps=steps, log_interval=5)
        model.save(path2tar)

    if env_name == "Cerere":
        USE_CUDA = torch.cuda.is_available()
        env = gymnasium.make('gym_examples/CERERE-v0', render_mode=None, scenario=scenario_name)
        observation, info = env.reset()
        losses = []
        terminated_tmp = 0
        actionsPerEpisode = []

        currentmodel = DQN(len(env.get_wrapper_attr('flatState')), len(env.get_wrapper_attr('actionSpace')))
        targetmodel = DQN(len(env.get_wrapper_attr('flatState')), len(env.get_wrapper_attr('actionSpace')))

        update_target(currentmodel, targetmodel)
        config = DDQN_Config.parse_config()
        replay_buffer = NaivePrioritizedBuffer(config.lenBuffer)
        optimizer = optim.Adam(currentmodel.parameters(), lr=0.001)
        random.seed(24)
        print("Action space is %d" % len(env.get_wrapper_attr('actionSpace')))
        print("Action spaces is %s" % env.action_space)
        print(config.num_steps)

        for step in range(1, config.num_steps + 1):
            epsilon = epsilon_by_frame(step, config)
            # tmp conversions
            state_tmp = env.get_wrapper_attr('flatState')  #ok
            #state_tmp = observation.tolist()
            #state_tmp = observation
            action = currentmodel.act(env.get_wrapper_attr('flatState'), epsilon,
                                      len(env.get_wrapper_attr('actionSpace')), USE_CUDA)  #ok
            #action = currentmodel.act(observation, 0, env.action_space, USE_CUDA)
            gamma = gamma_by_frame(step, config)
            observation, reward, terminated, truncated, info = env.step(action)
            # tmp conversions
            nextstate_tmp = env.get_wrapper_attr('flatState')  #ok
            #nextstate_tmp = observation.tolist()
            #nextstate_tmp = observation
            if terminated == True:
                terminated_tmp = 1
            actionsPerEpisode.append([action + 1, reward])
            # Push experience into Replay Buffer
            replay_buffer.push(state_tmp, action, reward, nextstate_tmp, terminated_tmp)
            if terminated == True:
                observation, info = env.reset()
                terminated_tmp = 0
                print("Step %d: Action per episode %s" % (step, actionsPerEpisode))
                actionsPerEpisode = []
            if len(replay_buffer) > config.batchsize:
                beta = beta_by_frame(step, config)
                loss = compute_td_loss(config.batchsize, currentmodel, targetmodel, optimizer, beta, gamma,
                                       replay_buffer, USE_CUDA, step)
                losses.append(loss.data)
            if step % 100 == 0:
                print("Step %d: Update targetmodel" % step)
                update_target(currentmodel, targetmodel)
        #print(losses)
        if os.path.exists(path2tar):
            os.remove(path2tar)
        print("Save model as %s" % path2tar)
        torch.save(targetmodel, path2tar)  # Store targetmodel into .pt-File
        env.close()
###### Train End


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    parser.add_argument('--env', type=str, default='NewCerere',
                        help='Environment: [MountainCar , Cerere, NewCerere] , default = Cerere')
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
        train_model(args.steps, ENVIRONMENT, SCENARIO, path2tar, rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)
