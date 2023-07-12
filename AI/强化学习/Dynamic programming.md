#Reinforcement-learning 

# Introduction

动态规划（dynamic programming，DP）适合解决满足最优子结构（optimal substructure）和重叠子问题（overlapping subproblem）两个性质的问题。

最优子结构意味着，问题可以拆分成一个个的小问题，通过解决这些小问题，我们能够组合小问题的答案，得到原问题的答案，即最优的解。重叠子问题意味着，子问题出现多次，并且子问题的解决方案能够被重复使用，我们可以保存子问题的首次计算结果，在再次需要时直接使用。

马尔可夫决策过程是满足动态规划的要求的，在贝尔曼方程里面，我们可以把它分解成递归的结构。当我们把它分解成递归的结构的时候，如果子问题的子状态能得到一个值，那么它的未来状态因为与子状态是直接相关的，我们也可以将之推算出来。价值函数可以存储并重用子问题的最佳的解。动态规划应用于马尔可夫决策过程的规划问题（DP is used for planning in an MDP）而不是学习问题，我们必须对环境是<font color="#ff0000">完全已知</font>的，才能做动态规划，也就是要知道状态转移概率和对应的奖励。使用动态规划完成预测问题和控制问题的求解，是解决马尔可夫决策过程预测问题和控制问题的非常有效的方式。

![[Pasted image 20230511024107.png]]

# 策略评估（Policy Evaluation）

已知马尔可夫决策过程以及要采取的策略 $\pi$ ，计算价值函数 $V_{\pi}(s)$ 的过程就是**策略评估**。

Problem: given policy $\pi$, compute value function $v_{\pi}$.
Solution: iterative application of Bellman equation

$$v_{k+1}(s)=\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left(r+\gamma v_{k}\left(s^{\prime}\right)\right)$$ 
This algorithm is called iterative policy evaluation.
![[Pasted image 20230511024344.png]]

> What is the analogous iteration for the action-value function $q_{\pi}$?


# 策略提升（Policy Improvement）

Given a deterministic policy π and its action value function $q_{\pi}$, we can update it by acting greedily
$$\pi^{\prime}(s)=\underset{a \in \mathcal{A}}{\operatorname{argmax}} q_{\pi}(s, a)$$ ![[Pasted image 20230511024658.png]]

![[Pasted image 20230511024731.png]]

# 策略迭代（Policy Iteration）

策略迭代由两个步骤组成：策略评估和策略改进（policy improvement）。如图 2.21a 所示，第一个步骤是策略评估，当前我们在优化策略 $\pi$，在优化过程中得到一个最新的策略。我们先保证这个策略不变，然后估计它的价值，即给定当前的策略函数来估计状态价值函数。
![[Pasted image 20230511025426.png]]

策略迭代的过程与踢皮球一样。我们先给定当前已有的策略函数，计算它的状态价值函数。算出状态价值函数后，我们会得到一个 Q 函数。我们对Q 函数采取贪心的策略，这样就像踢皮球，“踢”回策略。然后进一步改进策略，得到一个改进的策略后，它还不是最佳的策略，我们再进行策略评估，又会得到一个新的价值函数。基于这个新的价值函数再进行 Q 函数的最大化，这样逐渐迭代，状态价值函数和策略就会收敛。

Policy evaluation中的停止条件
![[Pasted image 20230511025634.png]]



# 价值迭代（Value Iteration）

**最优性原理定理（principle of optimality theorem）**：
一个策略$\pi(a|s)$ 在状态 $s$ 达到了最优价值，也就是 $V_{\pi}(s) = V^{*}(s)$ 成立，当且仅当对于任何能够从 $s$ 到达的 $s'$，都已经达到了最优价值。也就是对于所有的 $s'$，$V_{\pi}(s') = V^{*}(s')$ 恒成立。

如果我们知道子问题 $V^{*}(s')$ 的最优解，就可以通过价值迭代来得到最优的 $V^{*}(s)$ 的解。价值迭代就是把贝尔曼最优方程当成一个更新规则来进行，即
$$V(s) \leftarrow \max _{a \in A}\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V\left(s^{\prime}\right)\right)$$ 只有当整个马尔可夫决策过程已经达到最佳的状态时，式(2.22)才满足。但我们可以把它转换成一个备份的等式。备份的等式就是一个迭代的等式。我们不停地迭代贝尔曼最优方程，价值函数就能逐渐趋向于最佳的价值函数，这是价值迭代算法的精髓。

为了得到最佳的 $V^*$ ，对于每个状态的 $V$，我们直接通过贝尔曼最优方程进行迭代，迭代多次之后，价值函数就会收敛。这种价值迭代算法也被称为确认性价值迭代（deterministic value iteration）。

价值迭代算法的过程如下。

（1）初始化：令$k=1$ ，对于所有状态 $s$，$V_0(s)=0$。

（2）对于 $k=1:H$（$H$是让$V(s)$收敛所需的迭代次数）
		(a) 对于所有状态$s$：
    $$Q_{k+1}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{k}\left(s^{\prime}\right) $$
    (b) $k \leftarrow k+1$ 
（3）在迭代后提取最优策略：
	$$\pi(s)=\underset{a}{\arg \max } \left[R(s, a)+\gamma \sum_{s^{\prime} \in S} p\left(s^{\prime} \mid s, a\right) V_{H+1}\left(s^{\prime}\right)\right]$$
	