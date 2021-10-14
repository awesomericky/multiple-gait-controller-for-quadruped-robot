# Learning multiple gaits of quadruped robot using hierarchical reinforcement learning

We propose a method to learn multiple gaits of quadruped robot using hierarchical reinforcement learning. We designed a hierarchical controller and a learning framework that could learn and show multiple gaits in a single policy. Every experiment was done in RAISIM simulator.

Using our method, quadruped robot can learn multiple gaits including Trot, Pace, and Bound(imperfect).

We further successfully learned multiple gaits in a single policy using our framework.

In the project paper, we held a analysis of energy usage and concluded that optimal gait exists for specific velocity range. **However**, our analysis has critical drawback. We only considered the mechanical energy usage (which is lost by ground reaction impulse) and not consider the electrical energy usage (which is lost by heat energy in motor and PC). For strict analysis, we should use real robot and measure both mechanical and electrical energy usage (In simulator, it does not consider the electrical energy usage).

Therefore, for the people who saw this repository, just consider that there is such a way to learn multiple gaits of quadruped robot (**Don't take the energy analysis seriously.**)

# Method
<div>
  <img width=600 src='hierarchical_controller.png'>
</div>

# Results
1. Trot
<img width=300 src='video/trot.gif'>
2. Pace
<img width=300 src='video/pace.gif'>
3. Bound (imperfect)
<img width=300 src='video/bound.gif'>
4. Multiple gaits in a single policy (Pace & Bound)
<img width=300 src='video/multiple_gait.gif'>

# Further materials
- Project report: https://bit.ly/3mPlMqB
- Project slides: https://bit.ly/3ADOjV1
