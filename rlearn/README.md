Reinforcement Learning Instructions
===================================

This document describes how to setup the environment for reinforcement learning and run 
simulations . 

You should have received all the files and subdirs with this file or can
download the current version from the FKIE/KOM Bitbucket server under
https://team.fkie.fraunhofer.de/stash/scm/siglab/rlearn.git. 
See instructions below for a git repository clone.



Setting up the simulation environment under Debian/Ubuntu GNU Linux
----------------------------------------------------------------------

- Install the necessary Debian/Ubuntu packages

```shell
sudo apt install git python3-pip python3-pandas python3-scipy python3-numpy
```

```shell
pip install matplotlib itertools networkx
```

This can take some time as there are lots of dependencies.

- Clone this repository into a directory

With https:
```shell
git clone https://team.fkie.fraunhofer.de/stash/scm/siglab/rlearn.git
```

- Install the mininet and mininet-wifi


Running Simulations
----------------------

- For training and evaluation
	* execute sudo python main.py --attmode=0 --path2tar=./targetmodel_new.pt --train=1 --eval=1 --display=1 --anodes s1_7 s3_10 s2_9

- For evaluation
	* execute sudo python main.py --attmode=0 --path2tar=./targetmodel1.pt --train=0 --eval=1 --display=1 --anodes s1_7 s3_10 s2_9

- Remark
    * targetmodel1.pt trained with inital compromised nodes s1_7 s3_10 s2_9
    * targetmodel2.pt trained with inital compromised nodes s1_7 s2_9
    * targetmodel3.pt trained with inital compromised nodes s1_7 s2_9 a4_9

A's rec input:
sudo /home/ubuntu/src/rlearn/.venv/bin/python3.8 ./rlearn/main.py --anodes s2_9 s1_7 s3_10 --attmode=0 --path2res=./results --path2tar=./networks --path2topo=./graphs/topo_generic.csv --method=ddqn --eval=0 --train=1 --name=targetmodel_graph_38_experimental_rew --mode=none

