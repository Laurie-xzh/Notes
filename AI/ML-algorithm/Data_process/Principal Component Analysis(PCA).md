**用通过样本均值$m$的直线（单位向量为$e$）上的点重构样本点**

希望重构误差尽量的小。

$x_k = m+a_k\cdot e$ 其中$a_k$唯一决定了$x_k$.

**准则函数：最小化平方重构误差（Minimize square reconstruction errors）** 

$$\begin{array}{l}
J_{1}\left(a_{1}, \ldots, a_{n}, \mathbf{e}\right)=\sum_{k=1}^{n}\left\|\left(\mathbf{m}+a_{k} \mathbf{e}-\mathbf{x}_{k}\right)\right\|^{2}=\sum_{k=1}^{n}\left\|\left(a_{k} \mathbf{e}-\left(\mathbf{x}_{k}-\mathbf{m}\right)\right)\right\|^{2} \\
=\sum_{k=1}^{n} a_{k}^{2}\|\mathbf{e}\|^{2}-2 \sum_{k=1}^{n} a_{k} \mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right)+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2} \\
\frac{\partial J_{1}\left(a_{1}, \ldots, a_{n}, \mathbf{e}\right)}{\partial a_{k}}=2 a_{k}-2 \mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right)=0 \\
a_{k}=\mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right) \\
\end{array}$$
可以看到，$a_k$是$x_k-m$在$e$上的投影。可是$e$的方向该如何选取？如何找到最优的投影方向？
再把这个$a_k$代回上面的准则函数

$$
\begin{array}{l} 
J_{1}\left(a_{1}, \ldots, a_{n}, \mathbf{e}\right)=\sum_{k=1}^{n} a_{k}^{2}\|\mathbf{e}\|^{2}-2 \sum_{k=1}^{n} a_{k} \mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right)+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2} \\
J_{1}(\mathbf{e})=\sum_{k=1}^{n} a_{k}^{2}-2 \sum_{k=1}^{n} a_{k}^{2}+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2} \\
=-\sum_{k=1}^{n}\left[\mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right)\right]^{2}+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2} \\
=-\sum_{k=1}^{n} \mathbf{e}^{t}\left(\mathbf{x}_{k}-\mathbf{m}\right)\left(\mathbf{x}_{k}-\mathbf{m}\right)^{t} \mathbf{e}+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2} \\
=-\mathbf{e}^{t} \mathbf{S e}+\sum_{k=1}^{n}\left\|\mathbf{x}_{k}-\mathbf{m}\right\|^{2}
\end{array}
$$
再加上$e^te=1$,引入拉格朗日乘子法，就能迎刃而解。

最后$Se=\lambda e$

$\lambda$是散布矩阵$S$的最大特征值,$e$是对应的特征向量

对于散布矩阵进行分解的计算量比较大

也可以说，PCA降维是希望维度降低的同时能够尽可能保留更多的信息，对$e$的选取也是希望找到数据分布最分散的方向，即方差最大的方向，作为主成分。

![[Pasted image 20230803104819.png]] 

![[Pasted image 20230803104705.png]]