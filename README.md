# Learning multiple gaits of quadruped robot using hierarchical reinforcement learning

We propose a method to learn multiple gaits of quadruped robot using hierarchical reinforcement learning. We designed a hierarchical controller and a learning framework that could learn and show multiple gaits in a single policy. Every experiment was done in RAISIM simulator [link](https://raisim.com/sections/Introduction.html).

Using our method, quadruped robot can learn multiple gaits including Trot, Pace, and Bound(imperfect). We further successfully learned multiple gait in a single policy using our framework. To show the existence of optimal gaits for specific velocity range, we held an analysis of mechanical energy usage for each learned gaits. Check the paper for detailed results.

- Paper: http://arxiv.org/abs/2112.04741
- Project slides: https://bit.ly/3ADOjV1

# Method
<div>
  <img width=600 src='hierarchical_controller.png'>
</div>

# Result
1. Trot
<img width=300 src='video/trot.gif'>
2. Pace
<img width=300 src='video/pace.gif'>
3. Bound (imperfect)
<img width=300 src='video/bound.gif'>
4. Multiple gaits in a single policy (Pace & Bound)
<img width=300 src='video/multiple_gait.gif'>

# Cite
```
@article{kim2021learning,
	title={Learning multiple gaits of quadruped robot using hierarchical reinforcement learning},
	author={Kim, Yunho and Son, Bukun and Lee, Dongjun},
	journal={arXiv preprint arXiv:2112.04741},
	year={2021}
}
```
