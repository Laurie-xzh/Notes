#Reinforcement-learning 

# Introtion
在无法获取马尔可夫决策过程的模型情况下(环境不是complete knowledgable)，我们可以通过蒙特卡洛方法和时序差分方法来估计某个给定策略的价值。



# 蒙特卡罗方法
**Key idea** : state/action value = averaging sample returns

假设：
- experience is divided into episodes
- learn from complete episodes

Basic procedure: use MC for policy evaluation, and then improve the policy based on estimation

## Monte Carlo Policy Evaluation(MCPE)
**蒙特卡洛**方法是基于采样的方法，给定策略 $\pi$，我们让智能体与环境进行交互，可以得到很多轨迹（episodes）。每个轨迹都有对应的回报（return）：
$$G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots$$
我们求出所有轨迹（episodes）的回报（return）的平均值，就可以知道某一个策略对应状态的价值，即 
$$V_{\pi}(s)=\mathbb{E}_{\tau \sim \pi}\left[G_{t} \mid  s_{t}=s\right]$$
这里用采样均值代替了期望

蒙特卡洛方法使用经验平均回报（empirical mean return）的方法来估计，它不需要马尔可夫决策过程的状态转移函数和奖励函数，并且不需要像动态规划那样用自举的方法。

> first-visit v.s. every-visit to $s$ in an episode


此外，蒙特卡洛方法有一定的局限性，它只能用在有终止的马尔可夫决策过程中。

MC的收敛

![[Pasted image 20230511031606.png]]

## On-policy Improvement 

![[Pasted image 20230511031921.png]]

## Off-policy Evaluation via Importance Sampling

![[Pasted image 20230511031948.png]]


# 时序差分法
![[Pasted image 20230511031818.png]]



# Boostraping method


# SARSA

![[Pasted image 20230511032408.png]]

# Q-learning 

![[Pasted image 20230511032521.png]]

# n-step prediction

![[Pasted image 20230511032545.png]]

![[Pasted image 20230511032557.png]]

![[Pasted image 20230511032603.png]]

