# Attention

生理学中,生物的注意力受非自主性提示和自主性提示影响（分别对应不随意线索和随意线索）

之前的卷积、全连接、池化层都只考虑不随意线索，而注意力机制，就是想对随意线索进行建模

随意线索被称为查询(query)，输入是非意志线索(key)和值(value)的一个对。注意力机制（注意力池化）通过**query**来**有偏向性**的选择某些输入（key-value对）。

![[Pasted image 20230707114926.png]]


# Attention Pooling by similarity

$f(x)= \sum_i \alpha(x,x_i)y_i$，$\alpha(x,x_i)$是注意力权重

**非参注意力池化层**

给定数据$(x_i,y_i),i=1,...,n.$其中括号内是$(key,value)$

**Nadaraya-Watson核回归**：$f(x)=\sum_{i=1}^{n} \frac{K(x-x_i)}{\sum_{j=1}^{n}K(x-x_j)}y_i$，其中$x$是**query**

使用高斯核$K(u)=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{u^{2}}{2}\right)$，则

$$ \begin{array}{c}f(x) = \sum_{i = 1}^{n} \frac{\exp \left(-\frac{1}{2}\left(x-x_{i}\right)^{2}\right)}{\sum_{j = 1}^{n} \exp \left(-\frac{1}{2}\left(x-x_{j}\right)^{2}\right)} y_{i} \\ = \sum_{i = 1}^{n} \operatorname{softmax}\left(-\frac{1}{2}\left(x-x_{i}\right)^{2}\right) y_{i}\end{array} $$

式中的$x$是Query, $x_i$是Key, $y_i$是Value。Key越接近Query，相似度越大，对应的注意力权重也就分的越多。

**参数化的注意力池化层**

参数化的attention pooling就是在之前的基础上引入可以学习的$w$

$$ f(x)=\sum_{i=1}^{n} \operatorname{softmax}\left(-\frac{1}{2}\left(\left(x-x_{i}\right) w\right)^{2}\right) y_{i} $$

# **注意力分数**

注意力分数是query和key的相似度，注意力权重是分数的softmax结果。最后Attention Pooling输出的结果就是这些注意力权重的加权和。


![[Pasted image 20230707115003.png]]

下面把上文的内容扩展到高维

假设有一个查询$q \in R^q$和m个“键-值”对$(k_1,v_1),...,(k_m,v_m)$，其中$k_i \in R^k,v_i \in R^v$

$f(1,(k_1,v_1),...,(k_m,v_m))=\sum_{i=1}^{m}\alpha(a,k_i)v_i \in R^v$

$\alpha(a,k_i)=softmax(\alpha(q,k_i))$

**Masked Softmax**

下面是两种常用的Attention Score计算法

**Additive Attention(加性注意力)**

- 可学参数：$W_k \in R^{h \times k},W_q\in R^{h \times q}，v\in R^h\\ \alpha(k,q)=v^Ttanh(W_k k+W_q q)$
- 等价于将key和quary合并起来后放入到一个隐藏大小为h输出大小为1的单隐藏层MLP

优点：key, quary, value可以是任意长！

**Scale Dot-Product Attention(缩放点积注意力)**

- $q,k_i \in R^d$(等长的query和key)，则$\alpha(q,k_i)=<q,k_i>/ \sqrt{d}$ 。
- 向量化版本
    - $Q \in R^{n \times d},K \in R^{m \times d},V \in R^{m \times v}$
    - 注意力分数：$\alpha(Q,K)=\frac{QK^T}{\sqrt{d}} \in R^{n \times m}$
    - 注意力池化：$f=softmax(\alpha(Q,K))V \in R^{n \times v}$


# Multi-head attention

