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

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray.tune.registry import register_env
from ray.tune.logger import UnifiedLogger
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
# from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
# from ray.rllib.examples.rl_modules.classes import (
#    AlwaysSameHeuristicRLM,
#    BeatLastHeuristicRLM,
# )
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
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

        #print(self)

        obs_batch_size = len(tree.flatten(batch[SampleBatch.OBS])[0])
        #print(f"Obs Batch {batch[SampleBatch.OBS]}")
        #print(f"Actions Batch {batch[SampleBatch.ACTIONS]}")
        #print(f"Obs Batch size {obs_batch_size}")
        #print(f"Obs Batch2 {tree.flatten(batch[SampleBatch.OBS])[0]}")
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

    def manual_forward():
        print("Fuction manual_forward")
        action = 1
        return action
 
def _extract_cfg_id_from_infos(infos: dict) -> int | None:
    if not infos:
        return None
    # RLlib generally returns infos keyed by agent_id
    for _agent_id, info in infos.items():
        if isinstance(info, dict) and "config_id" in info:
            return info.get("config_id")
    return None


def _resolve_checkpoint_path(path: str) -> str:
    if not path:
        return path

    if path.startswith("file://"):
        path = path[len("file://"):]
    if os.path.isfile(os.path.join(path, "rllib_checkpoint.json")):
        return path

    if os.path.isdir(path):
        ckpts = [
            os.path.join(path, d)
            for d in os.listdir(path)
            if d.startswith("checkpoint_") and os.path.isdir(os.path.join(path, d))
        ]
        ckpts = [d for d in ckpts if os.path.isfile(os.path.join(d, "rllib_checkpoint.json"))]
        if ckpts:
            return max(ckpts, key=lambda p: os.path.getmtime(p))

    return path


def _parse_fixed_config_value(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        vals = [v.strip() for v in s.split(",") if v.strip()]
        return vals or None
    return None


def _enterprise_cfgs_from_config_set(path: str) -> list[str]:
    p = os.path.abspath(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "configs" not in data or not isinstance(data["configs"], dict):
        return ["C1", "C2", "C3"]

    all_cfgs = list(data["configs"].keys())
    fixed = _parse_fixed_config_value(data.get("fixed_config"))
    if fixed is None:
        return all_cfgs

    eff = [c for c in fixed if c in data["configs"]]
    return eff or all_cfgs


# Custom logger function 
def custom_logger_creator(custom_path, custom_str=""):
    logdir = os.path.join(custom_path, custom_str)
    os.makedirs(logdir, exist_ok=True)
    
    def logger_creator(config):
        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator


###### Train Begin
def train_model(
    iterations,
    stop_rw,
    env_name,
    scenario_name,
    path2tar,
    rwf,
    enterprise_config_set: str | None = None,
    enterprise_fixed_config_key: str | None = None,
    enterprise_config_keys: list[str] | None = None,
    min_iters: int = 0,
):

    print(f"Starting training on {str(path2tar)}.")

#    env_kwargs = dict(render_mode=None,rw_func=rwf, scenario=scenario_name)    			
#    ## Using the env defined
#    env = cerere_net_v2.parallel_env(**env_kwargs)   			  
#    #env.reset(seed=42)

#    register_env(
#       "pettingzoo_cerere",
#       lambda _: ParallelPettingZooEnv(env),
#    )

    env_kwargs = dict(
        render_mode=None,
        rw_func=rwf,
        scenario=scenario_name,
        enterprise_config_set=enterprise_config_set,
        enterprise_fixed_config_key=enterprise_fixed_config_key,
        enterprise_config_keys=enterprise_config_keys,
    )

    register_env(
        "pettingzoo_cerere",
        lambda env_config: PettingZooEnv(cerere_net_v2.env(**env_kwargs)),
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
            num_env_runners=8,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
#           env_to_module_connector=lambda env: (
                # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                FlattenObservations(multi_agent=True, agent_ids={"player_1"}),
#            ),
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
       print(f"Iter: {i}, Reward mean p1: {result['env_runners']['module_episode_returns_mean']['p1']}")
       print(f"Iter: {i}, Reward min: {result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MIN]}")
       print(f"Iter {i}, Num steps: {result[NUM_ENV_STEPS_SAMPLED_LIFETIME]}")
       #exit()
       # Early stopping only after min_iters
       if (i + 1) >= int(min_iters) and result['env_runners']['module_episode_returns_mean']['p1'] >= stop_rw:
           print(f"Reached episode return of {stop_rw} at iter {i} (min_iters={min_iters}) -> stopping")
           print("####################### Results Train Beginn #########################")
           print(result)
           print("####################### Results Train End    #########################")
           # Save model
           checkpoint_path = model.save(os.path.abspath(path2tar))
           print(f"Model saved at {checkpoint_path}")
           break
       #print(result)   
    print(f"Stop training after {i} iterations ...")
    # Always write a checkpoint so that --eval can be run deterministically right after training
    checkpoint_path = model.save(os.path.abspath(path2tar))
    print(f"Model saved at {checkpoint_path}")
###### Train Ende 


###### Train HMARL Begin
def train_hmarl(
    iterations,
    stop_rw,
    scenario_name,
    path2tar,
    rwf,
    enterprise_config_set: str | None = None,
    enterprise_fixed_config_key: str | None = None,
    enterprise_config_keys: list[str] | None = None,
    shared_patch_policy: bool = False,
    min_iters: int = 0,
    min_rw: float = float("-inf"),
    min_save_iters: int = 0,
):

    env_kwargs = dict(
        render_mode=None,
        rw_func=rwf,
        scenario=scenario_name,
        enterprise_config_set=enterprise_config_set,
        enterprise_fixed_config_key=enterprise_fixed_config_key,
        enterprise_config_keys=enterprise_config_keys,
    )
    register_env(
        "pettingzoo_cerere_hmarl",
        lambda env_config: PettingZooEnv(cerere_net_v2.hmarl_env(**env_kwargs)),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    # Policy IDs
    if shared_patch_policy:
        policies = {
            "pi_attacker",
            "pi_manager",
            "pi_worker_patch",
            "pi_worker_mig",
        }
    else:
        policies = {
            "pi_attacker",
            "pi_manager",
            "pi_worker_0",
            "pi_worker_1",
            "pi_worker_2",
            "pi_worker_3",
            "pi_worker_mig",
        }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id == "attacker":
            return "pi_attacker"
        if agent_id == "manager":
            return "pi_manager"
        if agent_id.startswith("worker_"):
            if shared_patch_policy and agent_id in {"worker_0", "worker_1", "worker_2", "worker_3"}:
                return "pi_worker_patch"
            return f"pi_{agent_id}"
        if agent_id == "worker_mig":
            return "pi_worker_mig"
        raise ValueError(f"Unknown HMARL agent_id: {agent_id}")

    config = (
        PPOConfig()
        .environment("pettingzoo_cerere_hmarl")
        .training(
            train_batch_size=1000,
            lr=0.0003,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.02,
            vf_loss_coeff=0.25,
            kl_coeff=0.005,
        )
        .env_runners(
            num_env_runners=8,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[p for p in policies if p != "pi_attacker"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "pi_attacker": RLModuleSpec(module_class=HeuristicAttackModule),
                    "pi_manager": RLModuleSpec(),
                    **(
                        {
                            "pi_worker_patch": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                        }
                        if shared_patch_policy
                        else {
                            "pi_worker_0": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_1": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_2": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_3": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                        }
                    ),
                    "pi_worker_mig": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                }
            )
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere_hmarl"))
        .framework(framework="torch")
    )

    algo = config.build_algo()
    base_save_path = os.path.abspath(path2tar)
    print(f"Starting HMARL training for {iterations} iterations ...")
    last_mean = None
    last_saved_mean_rw = float("-inf")
    for i in range(iterations):
        result = algo.train()
        last_mean = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        print(f"Iter: {i}, Reward mean: {last_mean}")

        if (
            (i + 1) >= int(min_save_iters)
            and float(last_mean) >= float(min_rw)
            and float(last_mean) >= float(last_saved_mean_rw)
        ):
            checkpoint_save_path = f"{base_save_path}_best_iter_{i + 1}"
            checkpoint_path = algo.save(checkpoint_save_path)
            last_saved_mean_rw = float(last_mean)
            print(
                f"HMARL model checkpoint saved at iter {i} with mean_rw={last_mean:.6f}: {checkpoint_path}"
            )

        if last_mean >= stop_rw and (i + 1) < int(min_iters):
            print(
                f"Reached stop_rw={stop_rw} at iter={i}, but min_iters={min_iters} not reached yet; continuing."
            )
        if (i + 1) >= int(min_iters) and last_mean >= stop_rw:
            print(f"Reached episode return of {stop_rw} at iter {i} (min_iters={min_iters}) -> stopping")
            break

    final_checkpoint_path = algo.save(base_save_path)
    print(f"Final HMARL model checkpoint saved at {final_checkpoint_path}")

    if last_saved_mean_rw == float("-inf"):
        print(
            "No best-checkpoint matched min-save criteria during training; only final checkpoint was saved."
        )


###### Eval HMARL Begin
def eval_hmarl(
    in_scenario,
    path2tar,
    in_rwf,
    n_episodes: int = 10,
    base_seed: int = 42,
    enterprise_config_set: str | None = None,
    shared_patch_policy: bool = False,
):
    """Evaluate a HMARL checkpoint by running RLlib's evaluate() API."""
    env_kwargs = dict(
        render_mode=None,
        rw_func=in_rwf,
        scenario=in_scenario,
        enterprise_config_set=enterprise_config_set,
    )
    register_env(
        "pettingzoo_cerere_hmarl",
        lambda env_config: PettingZooEnv(cerere_net_v2.hmarl_env(**env_kwargs)),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    if shared_patch_policy:
        policies = {
            "pi_attacker",
            "pi_manager",
            "pi_worker_patch",
            "pi_worker_mig",
        }
    else:
        policies = {
            "pi_attacker",
            "pi_manager",
            "pi_worker_0",
            "pi_worker_1",
            "pi_worker_2",
            "pi_worker_3",
            "pi_worker_mig",
        }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id == "attacker":
            return "pi_attacker"
        if agent_id == "manager":
            return "pi_manager"
        if agent_id.startswith("worker_"):
            if shared_patch_policy and agent_id in {"worker_0", "worker_1", "worker_2", "worker_3"}:
                return "pi_worker_patch"
            return f"pi_{agent_id}"
        if agent_id == "worker_mig":
            return "pi_worker_mig"
        raise ValueError(f"Unknown HMARL agent_id: {agent_id}")

    eval_config = (
        PPOConfig()
        .environment("pettingzoo_cerere_hmarl")
        .env_runners(
            num_env_runners=8,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[p for p in policies if p != "pi_attacker"],
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "pi_attacker": RLModuleSpec(module_class=HeuristicAttackModule),
                    "pi_manager": RLModuleSpec(),
                    **(
                        {
                            "pi_worker_patch": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                        }
                        if shared_patch_policy
                        else {
                            "pi_worker_0": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_1": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_2": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                            "pi_worker_3": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                        }
                    ),
                    "pi_worker_mig": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                }
            )
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere_hmarl"))
        .framework(framework="torch")
        .evaluation(
            evaluation_num_env_runners=0,
            evaluation_interval=9999999999,
            evaluation_duration_unit="episodes",
            evaluation_duration=n_episodes,
            evaluation_config={"explore": False, "seed": base_seed},
        )
    )

    algo = eval_config.build_algo()
    ckpt = _resolve_checkpoint_path(path2tar)
    print(f"Restoring HMARL algo from checkpoint: {ckpt}")
    algo.restore_from_path(os.path.abspath(ckpt))
    results = algo.evaluate()
    mean_ret = results.get("env_runners", {}).get("episode_return_mean")
    print(f"HMARL eval: episode_return_mean={mean_ret}")
    return results


###### Train with Tune Begin
def train_model_with_tune(
    iterations,
    stop_rw,
    env_name,
    scenario_name,
    path2tar,
    rwf,
    enterprise_config_set: str | None = None,
    enterprise_fixed_config_key: str | None = None,
    enterprise_config_keys: list[str] | None = None,
):
 
    env_kwargs = dict(
        render_mode=None,
        rw_func=rwf,
        scenario=scenario_name,
        enterprise_config_set=enterprise_config_set,
        enterprise_fixed_config_key=enterprise_fixed_config_key,
        enterprise_config_keys=enterprise_config_keys,
    )
    register_env(
        "pettingzoo_cerere",
        lambda env_config: PettingZooEnv(cerere_net_v2.env(**env_kwargs)),
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
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
        .framework(framework="torch")
    )

    #stop = {"num_env_steps_sampled_lifetime": iterations, "env_runners/episode_return_mean": stop_rw} ##0.64 ent, 0.83 mil
    stop = {"num_env_steps_sampled_lifetime": iterations, "env_runners/module_episode_returns_mean/p1": stop_rw} ##0.64 ent, 0.83 mil

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(
            num_gpus=int(torch.cuda.is_available()),           
            num_cpus=8,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
        )
    # execute training 
    analysis = ray.tune.run(
        "PPO",
        config=config,
        stop=stop,
#        metric="env_runners/episode_return_mean",
        metric="env_runners/module_episode_returns_mean/p1",
        mode="max",
        checkpoint_at_end=True,
    )
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
    register_env(
        "pettingzoo_cerere",
        lambda env_config: PettingZooEnv(cerere_net_v2.env(**env_kwargs)),
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
            create_env_on_local_worker=True,
            num_env_runners=8,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
#            env_to_module_connector=lambda env: (
#                # `agent_ids=...`: Only flatten obs for the learning RLModule.
#                FlattenObservations(multi_agent=True, agent_ids={"player_0"}),
#            ),
    
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
        .evaluation(
            evaluation_duration_unit="episodes",
            evaluation_duration=1,
            evaluation_config={"explore": False},
        )
    )

    eval_algo = eval_config.build_algo()
#    print("####################### Print Algo Beginn #########################")
#    print(eval_algo)
#    print("####################### Print Algo End    #########################")
    ckpt = _resolve_checkpoint_path(path2tar)
    print(f"Restoring from checkpoint: {ckpt}")
    eval_algo.restore_from_path(os.path.abspath(ckpt))
    results = eval_algo.evaluate()
    print("####################### Results Eval Beginn #########################")
    Reval = results['env_runners']['episode_return_mean']
    Leval = results['env_runners']['episode_len_mean']

    # If the env provides config_id in info, RLlib will store it in the evaluation hist_stats
    # We aggregate per config_id when available
    per_cfg = {}
    hist = results.get('env_runners', {}).get('hist_stats', {})
    cfg_ids = hist.get('config_id', [])
    ep_returns = hist.get('episode_return', [])
    if cfg_ids and ep_returns and len(cfg_ids) == len(ep_returns):
        for cid, ret in zip(cfg_ids, ep_returns):
            per_cfg.setdefault(int(cid), []).append(float(ret))

    print(f" R(eval)={Reval}, L(eval)={Leval}")
    if per_cfg:
        print(" Per-config returns:")
        for cid in sorted(per_cfg):
            vals = per_cfg[cid]
            print(f"  C{cid+1}: n={len(vals)}, mean={sum(vals)/len(vals):.4f}")
    print("####################### Results Eval End    #########################")
###### Eval End


###### Eval2 Begin
def eval_model2(
    in_render_mode,
    in_scenario,
    path2tar,
    in_rwf,
    n_episodes: int = 60,
    base_seed: int = 42,
    verbose: bool = False,
):

    env_kwargs = dict(render_mode=in_render_mode, rw_func=in_rwf, scenario=in_scenario)
    #env_kwargs = dict(render_mode=None, rw_func=in_rwf, scenario=in_scenario)

    env = cerere_net_v2.env(**env_kwargs)
    register_env(
        "pettingzoo_cerere",
        lambda env_config: PettingZooEnv(cerere_net_v2.env(**env_kwargs)),
    )

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


    ckpt = _resolve_checkpoint_path(path2tar)
    print(f"Restore RLModule from checkpoint: {ckpt} ...", end="")
    rl_module = RLModule.from_checkpoint(
        os.path.abspath(
            os.path.join(
                ckpt,
                "learner_group",
                "learner",
                "rl_module",                
                "p1",
            )
        )
    )
    print(" ok")

    ham = HeuristicAttackModule()

    myrewards = {agent: 0 for agent in env.possible_agents}
    n_episodes = 1
    i = 0

    by_cfg = {}
    by_cfg_returns: dict[int, list[float]] = {}
    by_cfg_success: dict[int, list[int]] = {}

    for i in range(n_episodes):
        env.reset(seed=None , options={"config_key": "C1"})
        cfg_key = getattr(env.unwrapped, "selected_config_key", None)
        cfg_id = env.infos[env.possible_agents[0]].get("config_id") if env.infos else None

        # Episode return per agent
        ep_returns = {agent: 0.0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            #ep_returns[agent] += float(reward)

            if termination or truncation:
                action = None
            else:
                if agent == env.possible_agents[0]:
                    #action = env.action_space(agent).sample()
                    input_dict = {Columns.OBS: torch.from_numpy(observation).unsqueeze(0)}
                    #ham.manual_forward()
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
            # Episode return per agent
            ep_returns[agent] = float(reward)

        # Success criterion: critical server not infected at episode end
        crit = getattr(env.unwrapped, "critserver", None)
        nwstate = getattr(env.unwrapped, "nwstate", None)
        success = 0
        if crit is not None and nwstate is not None:
            success = 0 if [1, crit] in nwstate else 1

        # Consider player_1 as the defender policy
        ep_ret_def = float(ep_returns.get("player_1", 0.0))

        #if verbose:
        print(f"Episode {i+1:03d}: cfg={cfg_key} (id={cfg_id}) return={ep_ret_def:.4f} success={success}")

        if cfg_id is not None:
            cid = int(cfg_id)
            by_cfg_returns.setdefault(cid, []).append(ep_ret_def)
            by_cfg_success.setdefault(cid, []).append(success)

    print("\nPer-config evaluation (defender=player_1)")
    print("config\tn\tmean_return\tsuccess_rate")
    for cid in sorted(by_cfg_returns):
        rets = by_cfg_returns[cid]
        succ = by_cfg_success.get(cid, [])
        mean_ret = sum(rets) / len(rets) if rets else float("nan")
        succ_rate = (sum(succ) / len(succ)) if succ else float("nan")
        print(f"C{cid+1}\t{len(rets)}\t{mean_ret:.4f}\t\t{succ_rate:.3f}")

    env.close()
###### Eval2 End


def eval_model2_forced_configs(
    in_scenario: str,
    path2tar: str,
    in_rwf: int,
    configs: list[str] = ["C1", "C2", "C3"],
    n_episodes_per_cfg: int = 20,
    base_seed: int = 42,
    verbose: bool = False,
    deterministic: bool = True,
    enterprise_config_set: str | None = None,
):

    def _safe_softmax(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        vec = np.where(np.isfinite(vec), vec, -1e9)
        m = np.max(vec)
        exps = np.exp(vec - m)
        denom = np.sum(exps)
        if not np.isfinite(denom) or denom <= 0:
            return np.ones_like(vec, dtype=np.float64) / float(len(vec))
        p = exps / denom
        if not np.all(np.isfinite(p)) or np.any(p < 0) or np.sum(p) <= 0:
            return np.ones_like(vec, dtype=np.float64) / float(len(vec))
        return p

    env_kwargs = dict(
        render_mode=None,
        rw_func=in_rwf,
        scenario=in_scenario,
        enterprise_config_set=enterprise_config_set,
    )
    env = cerere_net_v2.env(**env_kwargs)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    ckpt = _resolve_checkpoint_path(path2tar)
    rl_module = RLModule.from_checkpoint(
        os.path.abspath(
            os.path.join(
                ckpt,
                "learner_group",
                "learner",
                "rl_module",
                "p1",
            )
        )
    )

    results = {}
    for cfg in configs:
        rets: list[float] = []
        wins: list[int] = []
        reasons: list[str] = []
        for ep in range(n_episodes_per_cfg):
            seed = base_seed + ep
            env.reset(seed=seed, options={"config_key": cfg})

            ep_returns = {agent: 0.0 for agent in env.possible_agents}
            for agent in env.agent_iter():
                observation, reward, termination, truncation, info = env.last()
                ep_returns[agent] += float(reward)
                if termination or truncation:
                    action = None
                else:
                    if agent == env.possible_agents[0]:
                        action = 1
                    else:
                        input_dict = {Columns.OBS: torch.from_numpy(observation).unsqueeze(0)}
                        out = rl_module.forward_inference(input_dict)
                        logits = convert_to_numpy(out[Columns.ACTION_DIST_INPUTS])[0]
                        if deterministic:
                            action = int(np.argmax(logits))
                        else:
                            probs = _safe_softmax(logits)
                            action = int(
                                np.random.choice(
                                    env.action_space(agent).n, p=probs
                                )
                            )
                env.step(action)

            win = 0
            reason = "unknown"
            try:
                outcome = getattr(env.unwrapped, "last_outcome", None)
                if isinstance(outcome, dict) and outcome.get("defender_win") is not None:
                    win = 1 if outcome.get("defender_win") is True else 0
                    reason = outcome.get("term_reason") or "unknown"
                else:
                    info_any = next(iter(env.infos.values())) if env.infos else None
                    win = 1 if (isinstance(info_any, dict) and info_any.get("defender_win") is True) else 0
                    reason = (info_any.get("term_reason") if isinstance(info_any, dict) else None) or "unknown"
            except Exception:
                pass

            ret = float(ep_returns.get("player_1", 0.0))
            rets.append(ret)
            wins.append(win)
            reasons.append(str(reason))
            if verbose:
                print(f"baseline cfg={cfg} ep={ep} seed={seed} return={ret:.4f} win={win} reason={reason}")

        results[cfg] = {
            "n": len(rets),
            "mean_return": float(sum(rets) / len(rets)) if rets else float("nan"),
            "success_rate": float(sum(wins) / len(wins)) if wins else float("nan"),
            "term_reason_counts": {r: reasons.count(r) for r in sorted(set(reasons))},
        }

    env.close()
    return results


def eval_hmarl_forced_configs(
    in_scenario: str,
    path2tar: str,
    in_rwf: int,
    configs: list[str] = ["C1", "C2", "C3"],
    n_episodes_per_cfg: int = 20,
    base_seed: int = 42,
):

    env_kwargs = dict(render_mode=None, rw_func=in_rwf, scenario=in_scenario)
    register_env(
        "pettingzoo_cerere_hmarl",
        lambda env_config: PettingZooEnv(cerere_net_v2.hmarl_env(**env_kwargs)),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    policies = {
        "pi_manager",
        "pi_worker_0",
        "pi_worker_1",
        "pi_worker_2",
        "pi_worker_3",
        "pi_worker_mig",
    }

    def policy_mapping_fn(agent_id, episode, **kwargs):
        if agent_id == "manager":
            return "pi_manager"
        if agent_id.startswith("worker_"):
            return f"pi_{agent_id}"
        if agent_id == "worker_mig":
            return "pi_worker_mig"
        raise ValueError(f"Unknown HMARL agent_id: {agent_id}")

    eval_config = (
        PPOConfig()
        .environment("pettingzoo_cerere_hmarl")
        .env_runners(
            num_env_runners=8,
            num_envs_per_env_runner=1,
            num_cpus_per_env_runner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies),
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "pi_manager": RLModuleSpec(),
                    "pi_worker_0": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                    "pi_worker_1": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                    "pi_worker_2": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                    "pi_worker_3": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                    "pi_worker_mig": RLModuleSpec(module_class=ActionMaskingTorchRLModule),
                }
            )
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere_hmarl"))
        .framework(framework="torch")
    )

    algo = eval_config.build_algo()
    ckpt = _resolve_checkpoint_path(path2tar)
    algo.restore_from_path(os.path.abspath(ckpt))

    out = {}
    for cfg in configs:
        algo.get_config().evaluation_config = {
            "explore": False,
            "seed": base_seed,
            "env_options": {"config_key": cfg},
        }
        algo.get_config().evaluation_duration_unit = "episodes"
        algo.get_config().evaluation_duration = n_episodes_per_cfg
        r = algo.evaluate()
        mean_ret = r.get("env_runners", {}).get("episode_return_mean")
        out[cfg] = {
            "n": n_episodes_per_cfg,
            "mean_return": float(mean_ret) if mean_ret is not None else None,
        }

    return out


def eval_hmarl_manual_forced_configs(
    in_scenario: str,
    path2tar: str,
    in_rwf: int,
    configs: list[str] = ["C1", "C2", "C3"],
    n_episodes_per_cfg: int = 20,
    base_seed: int = 42,
    deterministic: bool = True,
    enterprise_config_set: str | None = None,
    shared_patch_policy: bool = False,
):

    env = cerere_net_v2.hmarl_env(
        render_mode=None,
        rw_func=in_rwf,
        scenario=in_scenario,
        enterprise_config_set=enterprise_config_set,
    )
    ckpt = _resolve_checkpoint_path(path2tar)

    # Load all policies
    def _load_module(policy_id: str) -> RLModule:
        return RLModule.from_checkpoint(
            os.path.abspath(
                os.path.join(
                    ckpt,
                    "learner_group",
                    "learner",
                    "rl_module",
                    policy_id,
                )
            )
        )

    if shared_patch_policy:
        modules = {
            "pi_attacker": _load_module("pi_attacker"),
            "pi_manager": _load_module("pi_manager"),
            "pi_worker_patch": _load_module("pi_worker_patch"),
            "pi_worker_mig": _load_module("pi_worker_mig"),
        }
    else:
        modules = {
            "pi_attacker": _load_module("pi_attacker"),
            "pi_manager": _load_module("pi_manager"),
            "pi_worker_0": _load_module("pi_worker_0"),
            "pi_worker_1": _load_module("pi_worker_1"),
            "pi_worker_2": _load_module("pi_worker_2"),
            "pi_worker_3": _load_module("pi_worker_3"),
            "pi_worker_mig": _load_module("pi_worker_mig"),
        }

    def policy_for_agent(agent_id: str) -> str:
        if agent_id == "attacker":
            return "pi_attacker"
        if agent_id == "manager":
            return "pi_manager"
        if agent_id.startswith("worker_"):
            if shared_patch_policy and agent_id in {"worker_0", "worker_1", "worker_2", "worker_3"}:
                return "pi_worker_patch"
            return f"pi_{agent_id}"
        if agent_id == "worker_mig":
            return "pi_worker_mig"
        raise ValueError(f"Unknown HMARL agent_id={agent_id}")

    def pick_action(mod: RLModule, obs, action_space):
        # obs may be dict (workers) or np array (manager)
        if isinstance(obs, dict):
            obs_t = {
                "observations": torch.from_numpy(obs["observations"]).unsqueeze(0),
                "action_mask": torch.from_numpy(obs["action_mask"]).unsqueeze(0),
            }
            input_dict = {Columns.OBS: obs_t}
        else:
            input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}

        out = mod.forward_inference(input_dict)

        if isinstance(out, dict) and (Columns.ACTIONS in out or SampleBatch.ACTIONS in out):
            acts = out.get(Columns.ACTIONS, out.get(SampleBatch.ACTIONS))
            acts = convert_to_numpy(acts)
            act0 = acts[0]
            if isinstance(action_space, gymnasium.spaces.MultiDiscrete):
                return np.array(act0, dtype=np.int64)
            return int(act0)

        if not isinstance(out, dict) or Columns.ACTION_DIST_INPUTS not in out:
            keys = list(out.keys()) if isinstance(out, dict) else type(out)
            raise KeyError(
                "RLModule.forward_inference() output does not contain action logits nor direct actions"
            )

        logits = convert_to_numpy(out[Columns.ACTION_DIST_INPUTS])[0]

        def _safe_softmax(vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(vec, dtype=np.float64)
            vec = np.where(np.isfinite(vec), vec, -1e9)
            m = np.max(vec)
            exps = np.exp(vec - m)
            denom = np.sum(exps)
            if not np.isfinite(denom) or denom <= 0:
                return np.ones_like(vec, dtype=np.float64) / float(len(vec))
            p = exps / denom
            if not np.all(np.isfinite(p)) or np.any(p < 0) or np.sum(p) <= 0:
                return np.ones_like(vec, dtype=np.float64) / float(len(vec))
            return p

        if isinstance(action_space, gymnasium.spaces.MultiDiscrete):
            nvec = list(action_space.nvec)
            idx = 0
            action = []
            for n in nvec:
                seg = logits[idx : idx + n]
                if deterministic:
                    action.append(int(np.argmax(seg)))
                else:
                    p = _safe_softmax(seg)
                    action.append(int(np.random.choice(n, p=p)))
                idx += n
            return np.array(action, dtype=np.int64)

        if deterministic:
            return int(np.argmax(logits))

        p = _safe_softmax(logits)
        return int(np.random.choice(action_space.n, p=p))

    out = {}
    for cfg in configs:
        rets = []
        wins = []
        reasons = []

        for ep in range(n_episodes_per_cfg):
            seed = base_seed + ep
            env.reset(seed=seed, options={"config_key": cfg})
            ep_returns = {aid: 0.0 for aid in env.possible_agents}

            for agent in env.agent_iter():
                obs, reward, termination, truncation, info = env.last()
                ep_returns[agent] += float(reward)
                if termination or truncation:
                    action = None
                else:
                    pid = policy_for_agent(agent)
                    action = pick_action(modules[pid], obs, env.action_space(agent))
                env.step(action)

            win = 0
            reason = "unknown"
            try:
                outcome = getattr(env.unwrapped, "last_outcome", None)
                if isinstance(outcome, dict) and outcome.get("defender_win") is not None:
                    win = 1 if outcome.get("defender_win") is True else 0
                    reason = outcome.get("term_reason") or "unknown"
                else:
                    info_any = next(iter(env.infos.values())) if env.infos else None
                    win = 1 if (isinstance(info_any, dict) and info_any.get("defender_win") is True) else 0
                    reason = (info_any.get("term_reason") if isinstance(info_any, dict) else None) or "unknown"
            except Exception:
                pass

            rets.append(float(ep_returns.get("manager", 0.0)))
            wins.append(win)
            reasons.append(str(reason))

        out[cfg] = {
            "n": len(rets),
            "mean_return": float(sum(rets) / len(rets)) if rets else float("nan"),
            "success_rate": float(sum(wins) / len(wins)) if wins else float("nan"),
            "term_reason_counts": {r: reasons.count(r) for r in sorted(set(reasons))},
        }

    env.close()
    return out


def print_comparison_table(baseline: dict, hmarl: dict, title: str = "Baseline vs HMARL"):
    print("\n" + title)
    print("cfg\tbaseline_mean\tbaseline_succ\thmarl_mean\thmarl_succ")
    for cfg in ["C1", "C2", "C3"]:
        b = baseline.get(cfg, {})
        h = hmarl.get(cfg, {})
        print(
            f"{cfg}\t"
            f"{b.get('mean_return', float('nan')):.4f}\t\t{b.get('success_rate', float('nan')):.3f}\t\t"
            f"{h.get('mean_return', float('nan')):.4f}\t\t{h.get('success_rate', float('nan')):.3f}"
        )


###### Test Begin
def test(in_render_mode, in_scenario):	

    env_kwargs = dict(render_mode=None, scenario=in_scenario)
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


def test_hmarl(in_scenario: str, rwf: int, max_env_cycles: int = 300):
    """Smoke test for the HMARL PettingZoo env without RLlib.

    Runs a single episode with random manager + random workers and ensures the
    episode terminates.
    """
    env = cerere_net_v2.hmarl_env(render_mode=None, rw_func=rwf, scenario=in_scenario)
    env.reset(seed=123)

    cycles = 0
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            if agent == "attacker":
                action = 1
            else:
                action = env.action_space(agent).sample()
        env.step(action)
        cycles += 1
        if cycles > max_env_cycles:
            raise RuntimeError(
                f"HMARL smoke test exceeded max cycles ({max_env_cycles}); likely stuck."
            )

    env.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', help="Test random action values in the spezified env", action="store_true")
    parser.add_argument('--eval', help="Eval a trained model in the specified env", action="store_true")
    parser.add_argument('--train', help="Train model in the specified env", action="store_true")
    parser.add_argument('--trainWithTune', help="Train with ray tune model in the specified env", action="store_true")
    parser.add_argument('--test_hmarl', help="Smoke-test HMARL env (no RLlib)", action="store_true")
    parser.add_argument('--train_hmarl', help="Train PPO on HMARL env", action="store_true")
    parser.add_argument('--eval_hmarl', help="Eval PPO checkpoint on HMARL env", action="store_true")
    parser.add_argument(
        '--hmarl_shared_patch',
        help='If set, share parameters across worker_0..worker_3 via single policy pi_worker_patch.',
        action='store_true'
    )
    parser.add_argument('--iter', type=int, default=50000,
                        help='Number of trainings iterations (ent=100000/mil=100000) , default = 50000')
    parser.add_argument('--stop_rw', type=float, default=0.1,
                        help='Mean reward to stop the training (ent=0.64 ent/ mil=0.83 mil), default = 0.1')
    parser.add_argument(
        '--min_rw',
        type=float,
        default=float('-inf'),
        help='HMARL only: minimum mean reward required before a checkpoint can be saved. Default=-inf',
    )
    parser.add_argument(
        '--min_iters',
        type=int,
        default=0,
        help='Minimum number of training iterations before stop_rw early-stopping is allowed. Default=0',
    )                    
    parser.add_argument(
        '--min_save_iters',
        type=int,
        default=0,
        help='HMARL only: minimum training iterations before checkpoint saving is allowed. Default=0',
    )
    parser.add_argument('--rwf', type=int, default=1,
                        help='Used reward function (iso-patch=1/bt=2) , default = 1')   
    parser.add_argument('--eval_episodes', type=int, default=1,
                        help='Number of evaluation episodes (enterprise uses multi-config reset). Default=60')
    parser.add_argument('--eval_seed', type=int, default=42,
                        help='Base seed for evaluation episode resets. Default=42')
    parser.add_argument('--eval_table', help='Prints a per-config (C1/C2/C3) table', action='store_true')
    parser.add_argument(
        '--hmarl_config_set',
        '--eval_config_set',
        dest='hmarl_config_set',
        type=str,
        default='config_sets/enterprise/default.json',
        help=(
            'Path to an enterprise config-set JSON file (C1/C2/C3 definitions). '
            'Used by HMARL evaluation table and HMARL RLlib evaluate() run. '
            'Alias: --eval_config_set. Default=config_sets/enterprise/default.json'
        ),
    )
    parser.add_argument(
        '--eval_deterministic',
        help='If set, choose argmax actions during manual per-config evaluation (no sampling).',
        action='store_true'
    )
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
        eval_model("human", SCENARIO, args.path2tar, args.rwf)
        #eval_model2("human", SCENARIO, args.path2tar, args.rwf, n_episodes=args.eval_episodes, base_seed=args.eval_seed,  verbose=False )
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.train:
        print("Train model in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model(
            args.iter,
            args.stop_rw,
            ENVIRONMENT,
            SCENARIO,
            args.path2tar,
            args.rwf,
            enterprise_config_set=(args.hmarl_config_set if SCENARIO == "enterprise" else None),
            enterprise_fixed_config_key=None,
            enterprise_config_keys=None,
            min_iters=args.min_iters,
        )
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.trainWithTune:
        print("Train model with Tune in env %s, scenario %s" % (ENVIRONMENT, SCENARIO))
        start = datetime.datetime.now().replace(microsecond=0)
        train_model_with_tune(
            args.iter,
            args.stop_rw,
            ENVIRONMENT,
            SCENARIO,
            args.path2tar,
            args.rwf,
            enterprise_config_set=(args.hmarl_config_set if SCENARIO == "enterprise" else None),
            enterprise_fixed_config_key=None,
            enterprise_config_keys=None,
        )
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train model with Tune in env %s after %s" % (ENVIRONMENT, elapsed))
    elif args.test_hmarl:
        print("HMARL smoke test in scenario %s" % SCENARIO)
        start = datetime.datetime.now().replace(microsecond=0)
        test_hmarl(SCENARIO, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("HMARL smoke test done after %s" % elapsed)

    elif args.train_hmarl:
        print("Train HMARL PPO in scenario %s" % SCENARIO)
        start = datetime.datetime.now().replace(microsecond=0)
        train_hmarl(
            args.iter,
            args.stop_rw,
            SCENARIO,
            args.path2tar,
            args.rwf,
            enterprise_config_set=(args.hmarl_config_set if SCENARIO == "enterprise" else None),
            enterprise_fixed_config_key=None,
            enterprise_config_keys=None,
            shared_patch_policy=bool(args.hmarl_shared_patch),
            min_iters=args.min_iters,
            min_rw=args.min_rw,
            min_save_iters=args.min_save_iters,
        )
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train HMARL PPO after %s" % elapsed)

    elif args.eval_hmarl:
        print("Eval HMARL PPO in scenario %s" % SCENARIO)
        start = datetime.datetime.now().replace(microsecond=0)

        eval_hmarl(
            SCENARIO,
            args.path2tar,
            args.rwf,
            n_episodes=args.eval_episodes,
            base_seed=args.eval_seed,
            enterprise_config_set=(args.hmarl_config_set if SCENARIO == "enterprise" else None),
            shared_patch_policy=bool(args.hmarl_shared_patch),
        )

        if args.eval_table:
            if SCENARIO != "enterprise":
                print("--eval_table is intended for enterprise only (C1/C2/C3).")
            else:
                cfgs = _enterprise_cfgs_from_config_set(args.hmarl_config_set)
                print("\nHMARL evaluation (forced configs)")
                hm = eval_hmarl_manual_forced_configs(
                    in_scenario=SCENARIO,
                    path2tar=args.path2tar,
                    in_rwf=args.rwf,
                    configs=cfgs,
                    n_episodes_per_cfg=args.eval_episodes,
                    base_seed=args.eval_seed,
                    deterministic=bool(args.eval_deterministic),
                    enterprise_config_set=args.hmarl_config_set,
                    shared_patch_policy=bool(args.hmarl_shared_patch),
                )
                print("cfg\tn\tmean_return\tsuccess_rate")
                for cfg in cfgs:
                    d = hm.get(cfg, {})
                    print(
                        f"{cfg}\t{d.get('n')}\t{d.get('mean_return', float('nan')):.4f}\t\t{d.get('success_rate', float('nan')):.3f}"
                    )
                    trc = d.get("term_reason_counts")
                    if trc:
                        print(f"  term_reason_counts: {trc}")
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval HMARL PPO after %s" % elapsed)

    elif args.eval_table:
        if SCENARIO == "enterprise":
            cfgs = _enterprise_cfgs_from_config_set(args.hmarl_config_set)
            baseline = eval_model2_forced_configs(
                in_scenario=SCENARIO,
                path2tar=args.path2tar,
                in_rwf=args.rwf,
                configs=cfgs,
                n_episodes_per_cfg=args.eval_episodes,
                base_seed=args.eval_seed,
                verbose=False,
                deterministic=bool(args.eval_deterministic),
                enterprise_config_set=args.hmarl_config_set,
            )
            print("\nBaseline evaluation (forced configs)")
            print("cfg\tn\tmean_return\tsuccess_rate")
            for cfg in cfgs:
                d = baseline.get(cfg, {})
                print(
                    f"{cfg}\t{d.get('n')}\t{d.get('mean_return', float('nan')):.4f}\t\t{d.get('success_rate', float('nan')):.3f}"
                )
                trc = d.get("term_reason_counts")
                if trc:
                    print(f"  term_reason_counts: {trc}")
    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)
