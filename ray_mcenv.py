import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

from typing import Optional

# ws-template-imports-end

import ray
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray import tune


#parser = add_rllib_example_script_args(
#    default_reward=0.9,
#    default_iters=50,
#    default_timesteps=100000,
#)

parser = add_rllib_example_script_args(
    default_iters=20,
#    default_timesteps=100000,
    default_timesteps=10000,
)


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("MountainCar-v0")
        .rl_module(
        # Use a non-default 32,32-stack with ReLU activations.
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
        ) )
    )

    results = run_rllib_example_script_experiment(base_config, args)
    print(results)
    best_result = results.get_best_result()
    print("My Checkpoint " , best_result.checkpoint)
    model_path = best_result.checkpoint
    #model_path = "/home/ubuntu/336d0f59-5906-47ef-a919-91fc5b236a1c"
    #model_path="/home/ubuntu/ray_results/PPO_2025-08-27_14-12-09/PPO_MountainCar-v0_0e65e_00000_0_2025-08-27_14-12-09/checkpoint_000000"
    #model_path="/home/ubuntu/ray_results/PPO_2025-08-27_17-11-21/PPO_MountainCar-v0_16d41_00000_0_2025-08-27_17-11-21/checkpoint_000000"
    #model_path ="/home/ubuntu/ray_results/DQN_2025-08-27_17-49-11/DQN_MountainCar-v0_603c7_00000_0_2025-08-27_17-49-11/checkpoint_000000"
    print("######################## Evaluation")

    eval_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
#        PPOConfig()
        .environment("MountainCar-v0", 
            env_config={"render_mode": "human"})
        .rl_module(
        # Use a non-default 32,32-stack with ReLU activations.
        model_config=DefaultModelConfig(
            fcnet_hiddens=[256, 256],
            fcnet_activation="relu",
        ))
        .env_runners(
             num_env_runners=0
        )
        .evaluation(
            evaluation_interval=9999999999,
#            evaluation_interval=1, # one eval per iteration
            evaluation_duration_unit="episodes",
            evaluation_duration=1,
        )
    )
    eval_algo = eval_config.build_algo()
    eval_algo.restore(model_path)
    results_eval = eval_algo.evaluate()
    print(results_eval)
 



