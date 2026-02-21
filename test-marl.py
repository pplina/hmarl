import gymnasium
from gymnasium.spaces import Discrete, Box
import argparse
import numpy as np
import torch
import os
import random
import datetime
import sys
import time
import json
from pathlib import Path
import re
import os

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
from ray.rllib.examples.rl_modules.classes.random_rlm import RandomRLModule
from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
    ActionMaskingTorchRLModule,
)
from ray.rllib.examples.rl_modules.classes import (
    AlwaysSameHeuristicRLM,
    BeatLastHeuristicRLM,
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

torch, _ = try_import_torch()

import cerere_net_v2


def _extract_cfg_id_from_infos(infos: dict) -> int | None:
    """Try to extract a config_id from RLlib infos structure.

    With PettingZooEnv wrapper, `infos` is typically a dict keyed by agent_id for
    the currently active agent only. We store `config_id` in PettingZoo env's
    `self.infos[agent]` for all agents at reset.
    """
    if not infos:
        return None
    # RLlib generally returns infos keyed by agent_id.
    for _agent_id, info in infos.items():
        if isinstance(info, dict) and "config_id" in info:
            return info.get("config_id")
    return None


def _resolve_checkpoint_path(path: str) -> str:
    """Resolve an RLlib checkpoint path.

    `Algorithm.save(checkpoint_dir=...)` usually creates a subdirectory like
    `checkpoint_000001/` and returns that full path.

    In our CLI, users sometimes pass the *parent directory* (containing multiple
    checkpoint_* subdirs). This helper resolves to the newest checkpoint dir.
    """
    if not path:
        return path

    # Normalize file:// URIs to local filesystem paths for our directory scanning.
    if path.startswith("file://"):
        path = path[len("file://"):]
    # If this already looks like a checkpoint dir, use it.
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

    # Fall back to given path; RLlib will raise a clearer error.
    return path


def _to_file_uri(path: str) -> str:
    """Convert a local path to a file:// URI (required by RLlib+pyarrow in Ray>=2.5x)."""
    if not path:
        return path
    if path.startswith("file://"):
        return path
    return "file://" + os.path.abspath(path)


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

    env_kwargs = dict(render_mode=None, rw_func=rwf, scenario=scenario_name)

    # IMPORTANT: RLlib expects the env creator to construct a fresh env instance
    # (RLlib may create multiple envs for sampling/evaluation). Do not capture
    # a single pre-created env in the closure.
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
                    "p0": RLModuleSpec(module_class=RandomRLModule),
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
           checkpoint_path = model.save(_to_file_uri(path2tar))
           print(f"Model saved at {checkpoint_path}")
           break
       #print(result)   
    print(f"Stop training after {i} iterations ...")
    # Always write a checkpoint (even if stop criterion was not met) so that
    # `--eval` can be run deterministically right after training.
    checkpoint_path = model.save(_to_file_uri(path2tar))
    print(f"Model saved at {checkpoint_path}")
    # Note: We register an env creator that constructs env instances inside RLlib.
    # Do not attempt to close a local `env` object here.
###### Train Ende 


###### Train HMARL Begin
def train_hmarl(iterations, stop_rw, scenario_name, path2tar, rwf):
    """Train PPO on the HMARL env (manager + 5 workers)."""

    env_kwargs = dict(render_mode=None, rw_func=rwf, scenario=scenario_name)
    register_env(
        "pettingzoo_cerere_hmarl",
        lambda env_config: PettingZooEnv(cerere_net_v2.hmarl_env(**env_kwargs)),
    )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    tmp_path = "./tb_log/"
    os.makedirs(tmp_path, exist_ok=True)

    # Policy IDs (6 policies as requested).
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
            # worker_0..worker_3
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
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies),
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    # Default PPO Torch RLModule for each policy.
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

    algo = config.build_algo()
    print(f"Starting HMARL training for {iterations} iterations ...")
    last_mean = None
    for i in range(iterations):
        result = algo.train()
        last_mean = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        print(f"Iter: {i}, Reward mean: {last_mean}")
        if last_mean >= stop_rw:
            print(f"Reached episode return of {stop_rw} -> stopping")
            break

    checkpoint_path = algo.save(_to_file_uri(path2tar))
    print(f"HMARL model saved at {checkpoint_path}")


###### Eval HMARL Begin
def eval_hmarl(in_scenario, path2tar, in_rwf, n_episodes: int = 10, base_seed: int = 42):
    """Evaluate a HMARL checkpoint by running RLlib's evaluate() API."""
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
        .env_runners(num_env_runners=0)
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
    algo.restore_from_path(_to_file_uri(ckpt))
    results = algo.evaluate()
    mean_ret = results.get("env_runners", {}).get("episode_return_mean")
    print(f"HMARL eval: episode_return_mean={mean_ret}")
    return results


###### Train with Tune Begin
def train_model_with_tune(iterations, stop_rw, env_name, scenario_name, path2tar, rwf):
 
    env_kwargs = dict(render_mode=None, rw_func=rwf, scenario=scenario_name)
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
#               model={
#                   "fcnet_hiddens": [256,256],
#                   "fcnet_activation": "relu",
#                   "vf_share_layers": True,
#               },
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
                    "p0": RLModuleSpec(module_class=RandomRLModule),
                    "p1": RLModuleSpec(),
                }
            ),
        )
        .debugging(log_level="ERROR", logger_creator=custom_logger_creator(tmp_path, "ppo_cerere"))
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
    ckpt = _resolve_checkpoint_path(path2tar)
    print(f"Restoring from checkpoint: {ckpt}")
    eval_algo.restore_from_path(_to_file_uri(ckpt))
    results = eval_algo.evaluate()
    print("####################### Results Eval Beginn #########################")
    Reval = results['env_runners']['episode_return_mean']
    Leval = results['env_runners']['episode_len_mean']

    # If the env provides config_id in `info`, RLlib will store it in the evaluation
    # hist_stats. We aggregate per config_id when available.
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
    env.close()
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

    # Avoid interactive rendering during CLI eval runs (matplotlib pauses would
    # block automated evaluation). Force render_mode=None.
    env_kwargs = dict(render_mode=None, rw_func=in_rwf, scenario=in_scenario)
    # We'll use a local env instance for the manual stepping loop below.
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
        _to_file_uri(
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

    # Per-config aggregates.
    by_cfg_returns: dict[int, list[float]] = {}
    by_cfg_success: dict[int, list[int]] = {}

    for i in range(n_episodes):
        env.reset(seed=base_seed + i)
        cfg_key = getattr(env.unwrapped, "selected_config_key", None)
        cfg_id = env.infos[env.possible_agents[0]].get("config_id") if env.infos else None

        # Episode return per agent.
        ep_returns = {agent: 0.0 for agent in env.possible_agents}

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            ep_returns[agent] += float(reward)

            if termination or truncation:
                action = None
            else:
                if agent == env.possible_agents[0]:
                    # Attacker/other player is random (baseline).
                    action = env.action_space(agent).sample()
                else:
                    input_dict = {Columns.OBS: torch.from_numpy(observation).unsqueeze(0)}
                    rl_module_out = rl_module.forward_inference(input_dict)
                    logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
                    action = int(
                        np.random.choice(env.action_space(agent).n, p=softmax(logits[0]))
                    )

            env.step(action)

        # Success criterion: critical server not infected at episode end.
        crit = getattr(env.unwrapped, "critserver", None)
        nwstate = getattr(env.unwrapped, "nwstate", None)
        success = 0
        if crit is not None and nwstate is not None:
            success = 0 if [1, crit] in nwstate else 1

        # We consider player_1 as the defender policy.
        ep_ret_def = float(ep_returns.get("player_1", 0.0))

        if verbose:
            print(
                f"Episode {i+1:03d}: cfg={cfg_key} (id={cfg_id}) return={ep_ret_def:.4f} success={success}"
            )

        if cfg_id is not None:
            cid = int(cfg_id)
            by_cfg_returns.setdefault(cid, []).append(ep_ret_def)
            by_cfg_success.setdefault(cid, []).append(success)

    # Print evaluation table.
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


###### Test Begin
def test(in_render_mode, in_scenario):	

    # Smoke tests should never use interactive rendering (matplotlib `plt.pause()`
    # would block CI/terminal runs). We therefore force `render_mode=None`.
    env_kwargs = dict(render_mode=None, scenario=in_scenario)
    env = cerere_net_v2.env(**env_kwargs)

    myrewards = {agent: 0 for agent in env.possible_agents}
    num_games = 1
    i = 0

    for i in range(num_games):
        i += 1
        print("######################### Test Round %d ########################" % i)
        # `render_mode` is already forced to None above.
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
    parser.add_argument('--iter', type=int, default=50000,
                        help='Number of trainings iterations (ent=100000/mil=100000) , default = 50000')
    parser.add_argument('--stop_rw', type=float, default=0.1,
                        help='Mean reward to stop the training (ent=0.64 ent/ mil=0.83 mil), default = 0.1')
    parser.add_argument('--rwf', type=int, default=1,
                        help='Used reward function (iso-patch=1/bt=2) , default = 1')   
    parser.add_argument('--eval_episodes', type=int, default=60,
                        help='Number of evaluation episodes (enterprise uses multi-config reset). Default=60')
    parser.add_argument('--eval_seed', type=int, default=42,
                        help='Base seed for evaluation episode resets. Default=42')
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
        eval_model2(
            "human",
            SCENARIO,
            args.path2tar,
            args.rwf,
            n_episodes=args.eval_episodes,
            base_seed=args.eval_seed,
            verbose=False,
        )
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
        train_hmarl(args.iter, args.stop_rw, SCENARIO, args.path2tar, args.rwf)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop train HMARL PPO after %s" % elapsed)

    elif args.eval_hmarl:
        print("Eval HMARL PPO in scenario %s" % SCENARIO)
        start = datetime.datetime.now().replace(microsecond=0)
        eval_hmarl(SCENARIO, args.path2tar, args.rwf, n_episodes=args.eval_episodes, base_seed=args.eval_seed)
        end = datetime.datetime.now().replace(microsecond=0)
        elapsed = end - start
        print("Stop eval HMARL PPO after %s" % elapsed)

    else:
        print("Do not know what to do in env %s" % ENVIRONMENT)



















