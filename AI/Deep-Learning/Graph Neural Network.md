
# My Question

> Message passing 的关系？

> Iterative Classification 中$\phi_1$的作用？

# Introduction 

ML for Graph don't need the feature engineering part. After the raw data is processed into graph data, the ML model will automacally learn the feature and do the downstream task.

Map nodes to d-dimensional embeddings such that similar nodes in the network are embedded close together.

![[Pasted image 20230707182622.png]]

Also, there are different types of tasks, there are graph level, community/subgraph level, node level and edge level.
![[Pasted image 20230707215816.png]]


To build a graph, its necessary to know: what is the node and what is the link.

Choice of the proper network representation of a given domain/problem determines our ability to use networks successfully: 
- In some cases, there is a unique, unambiguous representation 
- In other cases, the representation is by no means unique 
- The way you assign links will determine the nature of the question you can study

there different types: direct/undirect, heterogeneous, bipartite, weighted, adjacency matrix(one presentation form of graph)

A Heterogeneous Grah is defined as $G=(V,E,R,T)$ 
- Nodes with node types $v_i \in V$ 是点的集合
- Edges with relation types $v_i,r,v_j \in E$ 
- Node type $T(v_i)$ 是点的类型的集合
- Relation type $r \in R$ 


# Traditional ML Pipeline

There are two key points in traditional ML pielines:
- Design features for nodes/link/graph
- Obtain features for all training data

![[Pasted image 20230707223713.png]]


Models like logistic regression, Random forest are trained and applied to obtain features and make a prediction given a new node/link/graph.

Using effective features 𝒙 over graphs is the key to achieving good model performance. Actually, traditiional ML pipelines uses hand-designed features

## Node-level Tasks and Features
![[Pasted image 20230707221203.png]]
Goal: Characterize the structure and position of a node in the network. <u>Combine the topological information with the attribute-based information</u>.

The feature of a graph can include:
- Node degree 
	- It treats all neighboring  nodes equally
- Node centrality (How important is the node in the graph)
	- Engienvector certrality
		![[Pasted image 20230707222032.png]]
	- Betweenness centrality
		   ![[Pasted image 20230707222357.png]]
	- Closeness centrality
		![[Pasted image 20230707222643.png]]
	- PageRank 
- Clustering coefficient 
	![[Pasted image 20230707222718.png]]
- Graphlets 
	Observation: Clustering coefficient counts the #(triangles) in the ego-network
	Goal: Describe network structure around node 𝑢 
	 Graphlets are small subgraphs that describe the structure of node 𝑢’s network neighborhood
	Analogy
	- Degree counts #(edges) that a node touches 
	- Clustering coefficient counts #(triangles) that a node touches. 
	- Graphlet Degree Vector (GDV): Graphlet-base features for nodes 
		- GDV counts #(graphlets) that a node touches

## Link-level Tasks and Features

## Graph-level Features and Graph Kernels


# Node Embedding

## Introduction

Graph Representation Learning
Goal: Efficient task-independent feature learning for machine learning with graphs. No manual feature engneering is needed.

Why Embedding?
Embedding maps the nodes in a graph into an embedding space, and the <u>simarility</u> matters.
Similarity of embeddings between nodes indicates their similarity in the network. 
- Both nodes connected by an edge/Similar nodes are close to each other 
Embedding is able to encode network information, and this can be used for many downstream tasks.
One example: DeepWalk
![[Pasted image 20230709102607.png]]

## Node embedding:Encoder and Decoder

Embedding Nodes:Goal is to encode nodes so that similarity in the embedding space (e.g., dot product) approximates similarity in the graph.
![[Pasted image 20230709102945.png]]

The simarility in the initial network is needed to be defined and map nodes into the coordinates in the embedding space.

Usually, the dot product is used in the similarity of the embedding, like $Z_{v}^{T}Z_u$ . Then, how to design a simalirity function in the original network like $similarity(u,v)$ and connect the similarity in the embedding space?

1. Encoder maps from nodes to embeddings 
	$ENC(v)=z_v$ , where $v$ is a node in the graph, $z_v$ is a d-dimensional-embedding.
	One simple method is $ENC(v)=z_v=Z \cdot v$ , 
	 $\mathbf{Z} \in \mathbb{R}^{d \times|\mathcal{V}|}$ , is a matrix , each column is a node embedding. This is what we learn/optimize. 
	 $v \in \mathbb{I}^{|\mathcal{V}|}$ , this is an indicator vector, all zeroes except a one in column indicating node $v$ .
	 This simple encoder is just an embedding-lookup
	 ![[Pasted image 20230709110409.png]]
	 
2. Define a node similarity function (i.e., a measure of similarity in the original network) 
	

3. Decoder 𝐃𝐄𝐂 maps from embeddings to the similarity score 
	

4. Optimize the parameters of the encoder so that: 
$$
Similarity(u,v) \approx Z_{v}^{T}Z_u
$$ 
If I use the simple encoder then the matrix $Z$ is the parameter to be optimized. The decoder is based on node similarity, and the object is maximizing $Z_v^TZ_u$ for node pairs (u,v) that are similar.


How to Define Node Similarity?
Definition of Node Similarity can use random walks. 

## Random Walk Approaches for Node Embeddings

## Uniform Random Walk

### Overview
This is unsupervised/self-supervisedway of learning node embeddings. 
- We are not utilizing node labels 
- We are not utilizing node features 
- The goal is to directly estimate a set of coordinates (i.e., the embedding) of a node so that some aspect of the network structure (captured by DEC) is preserved. 
These embeddings are task independent 
- They are not trained for a specific task but can be used for any task.

![[Pasted image 20230709152148.png]]

The definition of Random Walk: Given a graph and a starting point, we select a neighbor of it at random, and move to this neighbor; then we select a neighbor of this point at random, and move to it, etc. The (random) sequence of points visited this way is a random walk on the graph.

One idea: I want to make the $Z_{v}^{T}Z_u \approx$ probability that $u$ and $v$ co-occur on a random walk over the graph. Then there are two steps:
- Estimate probability of visiting node 𝒗 on a random walk starting from node 𝒖 using some random walk strategy 𝑹
- Optimize embeddings to encode these random walk statistics. Similarity in ebedding space encodes random walk similarity.

Why Random Walk:
1. Expressivity: Flexible stochastic definition of node similarity that incorporates both local and higher-order neighborhood information Idea: if random walk starting from node 𝑢 visits 𝑣 with high probability, 𝑢 and 𝑣 are similar (high-order multi-hop information)
2. Efficiency: Do not need to consider all node pairs when training; only need to consider pairs that co-occur on random walks

Unsupervised Feature Learning
- Intuition: Find embedding of nodes in 𝑑-dimensional space that preserves similarity
- Idea: Learn node embedding such that nearby nodes are close together in the network.
- GIve a node $u$, I can use $N_R(u)$, the neighbourhood of $u$ obtained by some random walk strategy $R$ .

Feature Learning as Optimization
![[Pasted image 20230709153504.png]]


### Random Walk Optimization 

![[Pasted image 20230709155345.png]]
![[Pasted image 20230709155409.png]]

![[Pasted image 20230709160156.png]]

> Talk about the loss function. Here, is a for-for loop, and the normalization means a sum of all nodes. Hence the complexity is $O(n^2)$ . To simplify it, negative sampling is used. Instead of normalizing w.r.t. all nodes, just normalize against 𝑘 random “negative samples” $n_i$ .

$$
\log \left(\frac{\exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{v}\right)}{\sum_{n \in V} \exp \left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{n}\right)}\right) \\
\approx \log \left(\sigma\left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{v}\right)\right)-\sum_{i=1}^{k} \log \left(\sigma\left(\mathbf{z}_{u}^{\mathrm{T}} \mathbf{z}_{n_{i}}\right)\right), n_{i} \sim P_{V}

$$
The sample of negative nodes has a prob which is proportional to its degree. 
- Higher 𝑘 gives more robust estimates
- Higher 𝑘 corresponds to higher bias on negative events In practice 𝑘 =5-20

>Can negative sample be any node or only the nodes not on the walk? 
>
>People often use any nodes (for efficiency). However, the most “correct” way is to use nodes not on the walk.

Use SGD to optimize the loss function!

Summary
![[Pasted image 20230709161026.png]]


## Node2Vec (biased random walk)

### Overview
Goal: Embed nodes with similar network neighborhoods close in the feature space. This goal is independent to the downstream tasks.

Key observation: Flexible notion of network neighborhood $N_R(u)$ of node 𝑢 leads to rich node embeddings

Develop biased 2nd order random walk 𝑅 to generate network neighborhood 𝑁𝑅(𝑢) of node $u$ .

### Idea of node2vec
use flexible, biased random walks that can trade off between local and global views of the network.

>Explanation: DFS is global since it starts from a node and try to explore as many nodes. BFS is local since it can explore the nodes around the beginer node. So BFS offers a micro-view of neighbourhood, while DFS offers a macro-view of neighbourhood.

![[Pasted image 20230709162547.png]]
There are two parameters in node2vec
- Return parameter $p$ : Return bacl to the previous. This help to explore loacl area.
- In-out parameter $q$ : the ratio of 'BFS' and 'DFS'.


## Embedding Entire Graphs


# PageRank
I can regard static pages as nodes and the links are edges. Many web pages can be recognized as a directed graph. 
## PageRank Algorithm
Goal : Compute the importance of web pages on the web. 

Usually, in-coming links are more trustworthy. And the number of links can be related to votes.
The pagerank is a 'flow' model.
![[Pasted image 20230709165139.png]]
![[Pasted image 20230709165543.png]]

![[Pasted image 20230709171356.png]]

![[Pasted image 20230709171520.png]]


![[Pasted image 20230709171233.png]]


## Personalized PageRank(PPR)


## Random Walk with Restarts




# Message Passing and Collective Classification

## Introduction
Supposing some nodes have labels, while others doesn't have labels. I need to predict the unknown labels. Message passing can be used in this. <font color="#ff0000">Correlations</font> may be important. Nodes share the same labels tend to be connected. Hence <font color="#ff0000">collective classification</font> is used. One example is : nearby nodes have the same color.

Homophily(同质性) : Individuals with the same features may be connected
Influence : Social connection can also influence the characteristics of individuals.

Homophily and Influence above prove that correlation exists in graph.

Similar nodes are typically close together or directly connected in the network:
- Guilt-by-association : If I am connected to a node with label $X$, then I am likely to have label $X$ as well.
The classification of a node label may be effected by labels of the neighbours and the features of the neighbours.


## Collective Classification
Intuition: Simultaneous classification of interlinked nodes using correlations.
![[Pasted image 20230709202756.png]]
![[Pasted image 20230709204611.png]]


### Relational classification
![[Pasted image 20230709205459.png]]
![[Pasted image 20230709205850.png]]
An example, and the belief is passed.
![[Pasted image 20230709210928.png]]



### Iterative classification
![[Pasted image 20230709211328.png]]
![[Pasted image 20230709211607.png]]
![[Pasted image 20230709212931.png]]

Architecture of Iterative Classifiers
![[Pasted image 20230709213415.png]]

#### Example of Web Page Classification
![[Pasted image 20230709213735.png]]
![[Pasted image 20230709214043.png]]
![[Pasted image 20230709215030.png]]
![[Pasted image 20230709215055.png]]





### Belief propagation
Actually, the technique above and this belief propagation implement message passing. The message, belief are ssent over the edges of the network. Nodes can receive these messages and updating their belief. Then, in the next iteration, their neighbours are able to get this new information, and update their own belief. All of these techniques are about passing information across the neighbours, either pushing or receiving it, and update the belief of their own.

Belief Propagation is a dynamic programming approach to answering probability queries in a graph. Iterative process in which neighbor nodes 'talk' to each other, passing messages. When consensus is reached, calculate final belief.

#### Two examplex 
**Line-graph** 
![[Pasted image 20230710094549.png]]
**Tree-graph** 
![[Pasted image 20230710095000.png]]

#### Loopy Belief Propagation
What message will $i$ send to $j$ ? 
- It depends on what $i$ hears from its neighbors
- Each neighbor passes a message to $i$ its belief of the state of $i$ .
![[Pasted image 20230710095230.png]]

More formal presentation:
![[Pasted image 20230710095600.png]]

If homophily is present, the Label-label potential matrix will have high values on the diagonal. This means a node and its neighbor may have the same class. However, if there are large values off the diagonal, a node may have the opposite class to its neighbor.

![[Pasted image 20230710100519.png]]
Node $i$ aggregates all messages sent by neighbors and multiply its own prior belief. Then label-label potential is applied to send a message to node $j$ about how $i$'s label influence $j$'s label.

After convergence, final belief can be calculated. $b_i(Y_i)=$ node $i$'s belief of being in class $Y_i$ . 
![[Pasted image 20230710101517.png]]

<font color="#ff0000">A cycled example</font>
Messages from different subgraphs are no longer independent. Belief propagation can still be run. But it will pass messages in loops. In the example below, the belief 'T is true' is amplified in the loop.   
![[Pasted image 20230710103328.png]]

#### Evaluation of Belief Propagation
- Advantages:
	- Easy to program  & parallelize
	- General : can applu to any graph model with any form of potentials. Also, this potential function can be higher order. BP can not only learn homophily, but also learn more complex patterns
- Challenges
	- Convergence is not guaranteed, especially if many closed loops.
- Potential functions(parameters)
	- require training to estimate

# Graph Neural Network

cora地址： https://juejin.cn/post/7180924961790853181

Reference meterial
https://medium.com/data-reply-it-datatech/an-introduction-to-graph-neural-networks-b2d756c6b448

https://pypi.org/project/torch-geometric/

## Introduction
The target is appling deep neural network to graph. A naive approach is joining adjacency matrix and features, and feed them into a deep neural net. However, there are some problems: 1. $O(n)$ parameters 2. Not applicable to graphs of different sizes 3. Sensitive to node ordering

Goal : generalize convolutions beyond simple lattices. Leverage node features/attributes(like text, images)

In real-world, there is no fixed notion of locality or sliding window on the graph. Graph is permutation invariant. Sliding window may be strange. 

Conv in CNN actually transform information at the neighbors, combine it and create a new pixel. This is similar to that in message passing.

The idea of GCN : Node's neighborhood defines a computation graph. 
There are two steps : 
- Determine nodecomputation graph
- Propagate and transform information
In this way, the model can learn how to propagate information across the graph to compute node features.

Key idea of aggregating neighbors : Generage node embeddings based on local network neighborhoods. Supposing **A** is the target node, it gets information from B,C and D. Also, B gets information from A and C, and so on. This model needs to learn the message transformation operators along the edges as well as the aggregation operator. 
![[Pasted image 20230711215511.png]]

Difference to classical neural network is that every node gets to define its own neural network architecture based on its neighborhood. Since every node has its own network, we need to learn multi models.
![[Pasted image 20230711223152.png]]
The same node may have different embeddings at different layers. The model above will be run for limited steps. One hop means one layer. No notion of covergence.

Neighborhood Aggregation : Key distinctions are in how different approaches aggregate information across layers. Here aggregation operators must be order invariant or permutation invariant. One basic approach is averaging information from neighbors and aply a neural network. Agerage/Sum is order invariant. 
![[Pasted image 20230711224350.png]]
![[Pasted image 20230711224513.png]]

These weight matrices are shared across different nodes!

The Matrix Formulation of the aggregations
![[Pasted image 20230712102038.png]]
![[Pasted image 20230712102053.png]]





How to train a GNN ? 
![[Pasted image 20230712102426.png]]


## A single Layer of a GNN

GNN layer = Message + Aggregation
- Different instantiations unnder this perspective
- GCN, GraphSAGE, GAT

A single GNN layer : Compress a set of vectors into a single vector
This is a two-step process
- Message
- Aggregation
![[Pasted image 20230712104621.png]]


### Message Computation 
$$
m_u^{(l)} = MSG^{(l)}(h_u^{(l-1)})
$$
