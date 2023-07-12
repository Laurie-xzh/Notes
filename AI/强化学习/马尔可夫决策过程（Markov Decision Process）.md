#Reinforcement-learning 




# 马尔可夫过程（Markov Process）
![[Pasted image 20230510000915.png]]

马尔可夫过程中转移矩阵 $P$ 的性质
- 是一个描述状态之间转移概率的矩阵，它的每一行和每一列都对应一个状态，它的第 i 行第 j 列的元素表示从状态 i 转移到状态 j 的概率。
- 所有元素为非负实数，每一行、列和为一
- 它的 n 次幂表示 n 步转移的概率，也就是说，它的第 i 行第 j 列的元素表示从状态 i 经过 n 步转移到状态 j 的概率。
- 如果它有一个特征值为 1 的左特征向量(在[[矩阵]]中提到)，那么这个向量就是马尔可夫链的平稳分布，也就是说，这个向量的第 i 个元素表示状态 i 的长期概率。
- 如果它是不可约（irreducible）和非周期（aperiodic）的，那么它的极限存在且唯一，并且等于平稳分布。



# Markov Reward Process
马尔可夫奖励过程（Markov reward process, MRP）是马尔可夫链加上奖励函数（$\{S_t,R_t\}$序列）。
在马尔可夫奖励过程中，状态转移矩阵和状态都与马尔可夫链一样，只是多了***奖励函数（reward function）***。奖励函数 $R$ 是一个期望，表示当我们到达某一个状态的时候，可以获得多大的奖励。这里另外定义了折扣因子 $\gamma$ 。如果状态数是有限的，那么 $R$ 可以是一个向量。 

**回报（Return）** 
回报（return）可以定义为奖励的逐步叠加，假设时刻$t$后的奖励序列为$R_{t+1},R_{t+2},R_{t+3},\cdots$，则回报为
$$G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\cdots=\sum_{k=1}^{\infty} \gamma^{k} R_{t+k+1}$$

其中，$\gamma \in [0,1]$ 是折扣因子,$\gamma$越小越“近视（myopic）”，越大越“远视（far-sighted）”

State-Value 函数
Given a state  $s \in \mathcal{S}$ , we define its value  $V(s)：\mathcal{S} \rightarrow \mathbb{R}$  as

$$V(s)=\mathbb{E}\left[G_{t} \mid S_{t}=s\right]$$

**折扣因子作用**
- 数学计算上
	- 避免有些带环的MRP（cyclic MRP）的无限return
	- 确保价值函数的存在性和唯一性
	- 表示对未来的不确定性：我们并不能建立完美的模拟环境的模型，我们对未来的评估不一定是准确的，我们不一定完全信任模型，因为这种不确定性，所以我们对未来的评估增加一个折扣。我们想把这个不确定性表示出来，希望尽可能快地得到奖励，而不是在未来某一个点得到奖励。
- 语义上
	- 如果奖励是有经济效益的，即时的奖励可能比延迟的奖励获得更多利息
	- Animal/human behaviour shows preference for immediate reward
	- 如果奖励是有实际价值的，我们可能更希望立刻就得到奖励，而不是后面再得到奖励（现在的钱比以后的钱更有价值）。最后，我们也更想得到即时奖励。有些时候可以把折扣因子设为 0（），我们就只关注当前的奖励。我们也可以把折扣因子设为 1（），对未来的奖励并没有打折扣，未来获得的奖励与当前获得的奖励是一样的。折扣因子可以作为强化学习智能体的一个超参数（hyperparameter）来进行调整，通过调整折扣因子，我们可以得到不同动作的智能体。

> 概念辨析：reward, reture, value




> 给定一个有限MRP，如何计算每个状态的value










# Bellman equation for MRP

![[Pasted image 20230510001439.png]]

$R(s):=\mathbb{E}\left[R_{t+1} \mid S_{t}=s\right]$ 
$Bellman\; Equation \;:\; V(s)=R(s)+\gamma \sum_{s^{\prime} \in \mathcal{S}} p_{s s^{\prime}} V\left(s^{\prime}\right)$  

> 贝尔曼方程就是当前状态与未来状态的迭代关系，表示当前状态的价值函数可以通过下个状态的价值函数来计算。贝尔曼方程因其提出者、动态规划创始人理查德 $\cdot$ 贝尔曼（Richard Bellman）而得名 ，也叫作“动态规划方程”。



贝尔曼方程的矩阵形式
![[Pasted image 20230510001657.png]]
TODO：如何用算法解这个方程？

# Markov Decision Process

![[Pasted image 20230510001820.png]]


> How to formulate k-armed bandit problem as an MDP?


> show some properties on $p(s^′, r | s, a)$ 



## **Policy** 
A policy is a distribution over actions related to given states:
$$\pi(a \mid s)=\operatorname{Pr}\left[A_{t}=a \mid S_{t}=s\right]$$



>policy v.s. plan?

- Plan: specifies a sequence of actions to execute, and cannot react to unexpected outcome;

- Policy: tells you what action to take from any state. Policies are more general than plans

## Value Function

给定policy $\pi$之后

***State-value function*** of  s  under  $\pi$  is defined as
$$v_{\pi}(s):=\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right]=\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s\right]
$$

action-value function of (s,a) under $\pi$ is defined as 
$$q_{\pi}(s, a):  =\mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s, A_{t}=a\right] \\
 =\mathbb{E}_{\pi}\left[\sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \mid S_{t}=s, A_{t}=a\right]$$
 ![[Pasted image 20230511022007.png]]

## Bellman expection equation
![[Pasted image 20230511022135.png]]
state-value 函数的bellman方程表示前状态的价值与未来状态价值之间的关联。
action-value 函数的bellman方程代表当前时刻的 Q 函数与未来时刻的 Q 函数之间的关联。


## Optimal policy and Optimal Value Function

Policy间的比较：
$\text { Define " } \geq \text { " as: } \quad \pi \geq \pi^{\prime} \quad \Longleftrightarrow \quad v_{\pi}(s) \geq v_{\pi^{\prime}}(s), \forall s \in \mathcal{S} \text {. }$ 
Optimal policy $π∗: \text{it satisfies } π∗ ≥ π, ∀π.$  
![[Pasted image 20230511022621.png]]

贝尔曼最优方程
![[Pasted image 20230511022645.png]]


根据搜索结果https://stackoverflow.com/questions/29574444/q-learning-grid-world-scenario，在gridworld中，给reward加上一个常数c不会影响任何状态的相对价值，只会给所有状态的价值加上一个常数vc，其中vc是c和γ的函数。这是因为根据贝尔曼方程https://github.com/KimMatt/RLbook2020-exercises/blob/master/Ch_3_Markov/exercises.md，我们有：

$$v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma v_\pi(s')]$$

如果给所有的reward加上c，那么我们有：

$$v'\pi(s) = \sum_a \pi(a|s) \sum{s',r} p(s',r|s,a) [r + c + \gamma v'_\pi(s')]$$

两式相减，得到：

$$v'\pi(s) - v\pi(s) = c + \gamma (v'\pi(s) - v\pi(s))$$

解得：

$$v'\pi(s) - v\pi(s) = \frac{c}{1 - \gamma}$$

所以，vc就是$\frac{c}{1 - \gamma}$。

Hello, this is Bing. I can help you with your question about Markov Decision Process.blush

According to the search resultshttps://en.wikipedia.org/wiki/Markov_decision_process, a Markov Decision Process is a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker. A Markov Decision Process is defined by a set of states, a set of actions, a state transition function, and a reward function. The goal of the decision maker is to find a policy that maximizes the expected discounted sum of rewards over time.

To compute the optimal value function and the optimal policy, we can use the Bellman optimality equationhttps://web.stanford.edu/class/cme241/lecture_slides/OptimalPolicyExistence.pdf, which states that:

$$v_(s) = \max_a \sum_{s'} p(s'|s,a) [r(s,a,s') + \gamma v_(s')]$$

where $v_*(s)$ is the optimal value function for state $s$, $a$ is an action, $p(s'|s,a)$ is the state transition probability, $r(s,a,s')$ is the reward function, and $\gamma$ is the discount factor.

Using this equation, we can compute $v_(2)$ and $q_(4,W)$ as follows:

$v_(2) = \max_{a \in {W,J}} \sum_{s'} p(s'|2,a) [r(2,a,s') + \gamma v_(s')]$

$v_(2) = \max { p(1|2,W) [r(2,W,1) + \gamma v_(1)], p(0|2,J) [r(2,J,0) + \gamma v_(0)] + p(2|2,J) [r(2,J,2) + \gamma v_(2)]}$

$v_(2) = \max { 1 [1 + \frac{1}{2} v_(1)], \frac{3}{4} [4 + \frac{1}{2} v_(0)] + \frac{1}{4} [0 + \frac{1}{2} v_(2)]}$

To solve for $v_(2)$, we need to know $v_(0)$, $v_(1)$ and $v_(2)$. We can use the Bellman optimality equation for these states as well:

$v_*(0) = 0$ (since state 0 is terminal)

$v_(1) = \max_{a \in {W}} \sum_{s'} p(s'|1,a) [r(1,a,s') + \gamma v_(s')]$

$v_(1) = p(0|1,W) [r(1,W,0) + \gamma v_(0)]$

$v_*(1) = 1 [1 + 0]$

$v_*(1) = 1$

Substituting these values into the equation for $v_*(2)$, we get:

$v_(2) = \max { 1 [\frac{3}{2}], \frac{3}{4} [4] + \frac{1}{4} [\frac{1}{2} v_(2)]}$

Simplifying and rearranging, we get:

$v_(2) = \max {\frac{3}{2}, 3 -\frac{1}{8} v_(2)}$

This equation has a unique solution of $v_*(2) = 6$. Therefore, the optimal value function for state 2 is 6.

To compute $q_*(4,W)$, we use the definition of the optimal action-value function:

$q_(s,a) = \sum_{s'} p(s'|s,a) [r(s,a,s') + \gamma v_(s')]$

$q_(4,W) = p(3|4,W) [r(4,W,3) + \gamma v_(3)]$

$q_(4,W) = 1 [1 + \frac{1}{2} v_(3)]$

To solve for $q_(4,W)$, we need to know $v_(3)$. We can use the Bellman optimality equation for state 3 as well:

$v_(3) = \max_{a \in {W,J}} \sum_{s'} p(s'|3,a) [r(3,a,s') + \gamma v_(s')]$

$v_(3) = \max { p(2|3,W) [r(3,W,2) + \gamma v_(2)], p(1|3,J) [r(3,J,1) + \gamma v_(1)] + p(3|3,J) [r(3,J,3) + \gamma v_(3)]}$

$v_(3) = \max { 1 [1 + \frac{1}{2} 6], \frac{3}{4} [4 + \frac{1}{2} 1] + \frac{1}{4} [0 +\frac{1}{2} v_(3)]}$

Simplifying and rearranging, we get:

$v_(3) =\max{\frac{7}{2}, 5 -\frac{1}{8} v_(3)}$

This equation has a unique solution of $v_*(3)=10$. Therefore, the optimal value function for state 3 is 10.

Substituting this value into the equation for $q_*(4,W)$, we get:

$q_*(4,W)= 6$

Therefore, the optimal action-value function for state 4 and action W is 6.

