# SFOLS


## (Personal modifications)
Four purpose let's consider two toy examples; a small room with several 

Consider a room environment with four doorways, one on each of the cardinal directions which are goal states. We want to have a set of policies/$Q$-functions such that, given a new reward function that is a convex combination of those for which we have previously learned an optimal policy, the agent could retrieve an optimal policy for this new task. 


<img src="hallway.png">

$\phi_i(s, a, s') = \delta_{s'=i}$ 

For such purpose, I have created the toy example in ```envs/room_modified.py``` that represents a room with 4 doorways, each of which is a goal state.

I had to modify the way the SFs are constructed with regard to the other problems. Here we want a one-hot encoding of the SFs for each of the goal states, and not the states that are one step away like their examples.

However, the size of the CSS depends on the initial state distribution. For a uniform distribution the size seems to be 22.

My results can be reproduced with the following script and the initial states can be manually set in the environment above.
``` 
python experiments/run_room.py
```
_________________

Code for the paper "Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer" at ICML 2022.

Paper: https://arxiv.org/abs/2206.11326

## Install

To install run:
```bash
git clone https://github.com/LucasAlegre/sfols
cd sfols
pip install -e .
```

## Run Experiments

### Deep Sea Treasure
```bash
python experiments/run_dst.py -algo SFOLS
```
usage: run_dst.py [-h] [-algo {SFOLS,WCPI,Random}]

### Four Room
```bash
python experiments/run_fourroom.py -algo SFOLS
```
usage: run_fourroom.py [-h] [-algo {SFOLS,WCPI,SIP,Random}]

### Reacher
```bash
python experiments/run_reacher.py -SFOLS    
```
usage: run_reacher.py [-h] [-algo {SFOLS,WCPI,Random}]

## Citing

```bibtex
@inproceedings{Alegre+2022,
    author = {Lucas N. Alegre and Ana L. C. Bazzan and Bruno C. da Silva},
    title = {Optimistic Linear Support and Successor Features as a Basis for Optimal Policy Transfer},
    booktitle = {Proceedings of the Thirty-ninth International Conference on Machine Learning},
    address = {Baltimore, MD},
    year = {2022}
}
```
