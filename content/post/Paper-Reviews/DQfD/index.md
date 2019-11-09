---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Deep Q-learning from Demonstrations"
summary: "Leveraging human demonstrations to warmstart deep Q-learning"
categories: [Paper Summary]
tags: [Reinforcement Learning]
date: 2019-11-08
math: true
diagram: true
markup: mmark
draft: false
featured: false
reading_time: true
share: false
profile: false
commentable: false
editable: false
---

**Paper:** Download [here](https://arxiv.org/abs/1704.03732)  
**Authors:** Todd Hester et. al  
**Published in:** AAAI 2018  
**In-depth Analysis:** Daniel Sieta's [post](https://danieltakeshi.github.io/2019/04/30/il-and-rl/) 
## Takeaway message
Naively adding expert demonstration data to the replay buffer or pre-training an agent with this data does not offer benefits and sometimes can be detrimental. The combination of the supervised loss and the TD loss during pre-training is critical for the agent to learn a representation that is not rendered useless once training starts. During training the agent must continue using both the expert and the self-generated data using a prioritized experience replay. Demonstration data is used more as the number of iterations in the training phase increases. This is because the agent encounters new states in the game.

## Motivations
Training DQN for real-world applications can be challenging as it requires huge amount of data. Accurate simulators aren't available for many real-world tasks. The learning can be accelerated by using demonstration data to pre-train an agent so that it can perform well in the task from the start.

## Proposed Solution
The agent was pre-trained using 4 losses using the demonstration data.

1. *Supervised loss:* Large margin loss to push the expert's actions over other actions.
2. *TD loss:* To help the agent learn the value function. It also ensures that the network satisfies the Bellman equation.
3. *n-step (n=10) TD loss:* To help propagate the values of expert's trajectory to earlier states.
4. *L-2 regularization loss:* To prevent the agent from overfitting to the demonstration data.

The large margin classifier forces the values of other actions to be at least a margin lower than the value of the expert's action. This grounds the values of unseen actions to reasonable values and makes the greedy policy induced by the value function imitate the demonstrator.  

During training, the demonstration data is never overwritten in the replay buffer. Sampling is done using a prioritized experience replay buffer which also decides the ratio between demonstration and self-generated data. The supervised loss is omitted during training.

## Evaluation of the solution
Comparison was made between DQfD, Prioritized Duelling Double (PDD) DQN, and imitation learning. DQfD results in large boost in initial performance. It acheives SOTA performance in 11 Atari games. DQfD performs well in the hardest exploration games (Montezuma's Revenge, Pitfall, Private Eye) where demonstration data is used instead of intelligent exploration.

## Analysis of the problem, idea, and evaluation
The losses are designed to make the agent learn that other actions are less better than the expert action. So the goal is to create an agent that imitates the expert and the same time have a rough idea about the Q-values of the other states (unseen states) visited during the pre-training step.  

## Contributions
The paper presents a principled way to combine imitation learning with deep Q-learning as naive pre-training isn't guaranteed to perform well.

## Questions
1. How is the regularization loss helpful while training the RL agent? Doesn't it result in sub-optimal behavior?