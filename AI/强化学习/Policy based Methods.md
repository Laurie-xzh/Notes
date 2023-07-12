#Reinforcement-learning 

# Introduction

之前的方法都需要估计、近似出一个value $V(s)$ 或$Q(s,a)$，再根据value的值推导出policy.

Policy based methods则是直接把policy参数化
$$\pi(a \mid s, \boldsymbol{\theta})=\operatorname{Pr}\left\{A_{t}=a \mid S_{t}=s, \boldsymbol{\theta}_{t}=\boldsymbol{\theta}\right\}$$
会最大化一些函数$J(\theta)$来学习$\theta$。

>How to parameterize π(a | s, θ)?用函数

   用函数进行估计

用函数近似一个policy的优点
- Can inject prior knowledge easily (maybe the most important)
- Can learn stochastic policies
- Stronger convergence guarantees
- Effective in high-dimensional or continuous action spaces


Compare the function representation methods with the tabular representation methods for policy.

>How to define proper J(θ)?


>How to compute ∇J(θ)?


# 如何对$\pi(a \mid s,\theta)$进行参数化

soft-max in action preferences 是一种方法
步骤一


# Reinforce 算法


## Reinforce——Monte Carlo Policy Gradient





## Using baseline
