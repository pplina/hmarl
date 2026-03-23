######################################################################
##
## CERERE reinforcement learning with gymnasium
##
######################################################################

## Requirements
- gymnasium
- stablebaseline3
- ray

## Setup
1. Install the Gymnasium API
```
pip3 install gymnasium==1.0.0
```
2. From the main directory, install the CERERE environment for gymnasium
```
pip3 install -e gym-examples
```

3. To use the implementation execute the command lines below

# NewCERERE
```
python3 test.py --train --rwf 2  --steps 20000 --env NewCerere --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_mil_bt_ms4_new.zip 
python3 test.py --eval --rwf 2  --steps 20000 --env NewCerere --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_mil_bt_ms4.zip

python3 test.py --train --rwf 1  --steps 20000 --env NewCerere --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_mil_ip_ms4_new.zip 
python3 test.py --eval --rwf 1  --steps 20000 --env NewCerere --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_mil_ip_ms4.zip

python3 test.py --train --rwf 1  --steps 20000 --env NewCerere --scen enterprise --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_ent_new.zip
python3 test.py --eval --rwf 1  --steps 20000 --env NewCerere --scen enterprise --path2tar  /home/ubuntu/src/rl-test/ppo_cerere_ent.zip
```

# CERERE
```
No longer supported
python3 test.py --train --env Cerere --scen military --path2tar ./rlearn/networks/targetmodel_ai_gym_mil_new.pt
python3 test.py --eval --env Cerere --scen military --path2tar ./rlearn/networks/targetmodel_ai_gym_mil.pt

python3 test.py --train -env Cerere --scen enterprise --path2tar ./rlearn/networks/targetmodel_ai_gym_ent_new.pt
python3 test.py --eval -env Cerere --scen enterprise --path2tar ./rlearn/networks/targetmodel_ai_gym_ent.pt
```

#MountainCar
```
python3 test.py --train --env MountainCar  --path2tar=./dqn_mountaincar_new.zip
python3 test.py --eval --env MountainCar  --path2tar=./dqn_mountaincar.zip
```



######################################################################
##
## CERERE reinforcement learning with ray
##
######################################################################

## Setup
1. Install ray
```
pip3 install ray=1.50.1
```

## Usage
2. To use the implementation execute the command lines below

# NewCERERE
```
python3 testRayRL.py --train  --rwf 1  --iter 100000 --stop_rw 0.64 --env NewCerere --scen enterprise --path2tar /home/ubuntu/src/rl-test/ppo_ent_rllib_new/
python3 testRayRL.py --eval --rwf 1  --iter 100000 --stop_rw 0.64 --env NewCerere --scen enterprise --path2tar /home/ubuntu/src/rl-test/ppo_ent_rllib/


python3 testRayRL.py --train  --rwf 1  --iter 100000 --stop_rw 0.83 --env NewCerere --scen military --path2tar /home/ubuntu/src/rl-test/ppo_mil_rllib_new/
python3 testRayRL.py --eval  --rwf 1  --iter 100000 --stop_rw 0.83 --env NewCerere --scen military --path2tar /home/ubuntu/src/rl-test/ppo_mil_rllib/
```



######################################################################
##
## CERERE multi agent reinforcement learning with ray
##
######################################################################

## Usage
To use the implementation execute the command lines below

# NewCERERE
```
python3 test-marl.py --train --rwf 1  --stop_rw 0.83 --iter 100000 --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_mil_rllib_marl_new

python3 test-marl.py --eval --rwf 1  --stop_rw 0.83 --iter 100000 --scen military --path2tar  /home/ubuntu/src/rl-test/ppo_mil_rllib_marl
```

# NewCERERE + HMARL
```
python3 test-marl.py --train_hmarl --scen enterprise --rwf 1 --iter 10000 --stop_rw 999 --path2tar ./exp_hmarl_ppo

python3 test-marl.py --eval_hmarl --eval_table --eval_deterministic --scen enterprise --rwf 10 --eval_episodes 100 --eval_seed 100 --path2tar ./exp_hmarl_ppo_10k
```

*HMARL training*
```
python3 test-marl.py --train_hmarl --scen enterprise --rwf 4 --iter 10000 --stop_rw 999 --path2tar ./exp_hmarl_ppo_heuristic --hmarl_shared_patch
```

*HMARL evaluation*
```
python3 test-marl.py --eval_hmarl --scen enterprise --rwf 4 --path2tar ./exp_hmarl_ppo_heuristic --eval_table --eval_episodes 100 --eval_seed 1 --hmarl_shared_patch
```
