JAX的存在比起Numpy有4点优势：
- 能在GPU/TPU上运行
- 能自动微分
- 能并行编译
- 能多设备并行

一般来说，可以用`import jax.numpy as jnp`对numpy进行代替

## jax.grad函数

这个函数接收一个数值函数，返回其导数函数。就是进行一个求梯度的运算。只能处理标量输入。
```python
def sum_of_squares(x):
  return jnp.sum(x**2)

sum_of_squares_dx = jax.grad(sum_of_squares)

x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
print(sum_of_squares(x))
print(sum_of_squares_dx(x))

'''
30.0
[2. 4. 6. 8.]
'''
```
这里如果使用`jax.grad(f)(x)`，就是得到x的导函数值。

对于多元函数，求导时需要加上argnums进行指定。
例如有这样一个函数
```python
def sum_squared_error(x, y):
  return jnp.sum((x-y)**2)

sum_squared_error_dx = jax.grad(sum_squared_error)
y = jnp.asarray([1.1, 2.1, 3.1, 4.1])
print(sum_squared_error_dx(x, y))
```
如果想要获得对x和对y的导数，就要用`jax.grad(sum_squared_error, argnums=(0, 1))(x,y)` 
对于多个参数的函数，可以用参数表来表示。
```python
def loss_fn(params, data):
  ...

grads = jax.grad(loss_fn)(params, data_batch)
# 其中的params是一个嵌套的数组字典，返回的grads是另一个结构相同的qiantaoshuzu字典
```

其实还有一个函数：`jax.value_and_grad`，`jax.value_and_grad(f)(*xs) == (f(*xs), jax.grad(f)(*xs)) `，返回的是一个元组。

## 辅助数据 Auxiliary data

对于定义的函数，如果要其输出一个辅助数据，就要用到`has_aux`这个参数，比如对于均方差函数，我还要看x-y，那么函数该这么写：
```python
def squared_error_with_aux(x, y):
  return sum_squared_error(x, y), x-y

jax.grad(squared_error_with_aux)(x, y)#这么写就G咯
jax.grad(squared_error_with_aux, has_aux=True)(x, y)
```
`has_aux` signifies that the function returns a pair, `(out, aux)`. It makes `jax.grad` ignore `aux`, passing it through to the user, while differentiating the function as if only `out` was returned.

## 与NumPy的区别
JAX是函数式的。Don't write code wiwth side-effects. A side-effect is any effect of a function that doesn’t appear in its output.
一个side-effect sample:
```python
import numpy as np

x = np.array([1, 2, 3])

def in_place_modify(x):
  x[0] = 123
  return None

in_place_modify(x)
in_place_modify(jnp.array(x))#这里会报错！
```
这里不能对原数组进行更改，应该复制一份新的进行更改。
```python
# side-effect-free code
def jax_in_place_modify(x):
  return x.at[0].set(123)

y = jnp.array([1, 2, 3])
jax_in_place_modify(y)
```

## JAX中的JIT操作

如何用`jax.jit()` 对代码进行加速？
```python
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()

########### 用JIT进行加速 ##############
selu_jit = jax.jit(selu) # compiled version of selu

# Warm up
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
```

但是被jitted的函数值不能作为判断条件，例如：
```python
# Condition on value of x.
def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

f_jit = jax.jit(f)
f_jit(10)  # Should raise an error. 

# While loop conditioned on x and n.
def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

g_jit = jax.jit(g)
g_jit(10, 20)  # Should raise an error. 
```
解决方法是对部分进行JIT：
```python
# While loop conditioned on x and n with a jitted body.

@jax.jit    # 这里用装饰器进行jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

g_inner_jitted(10, 20)
```
也可以通过静态参数的形式避免报错，但是会带来更大的开销。
```python
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))
# 输出10

g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))
# 输出30
```
如果使用装饰器的时候还要静态参数，用`functools.partial`。
```python
from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))
```

jit一般用来加速复杂的、会重复运行的函数

<u>JIT的caching机制</u>  

Suppose I define `f = jax.jit(g)`. When I first invoke `f`, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of `f` will reuse the cached code. This is how `jax.jit` makes up for the up-front cost of compilation.

If I specify `static_argnums`, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs. If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one.

Avoid calling `jax.jit` inside loops. For most cases, JAX will be able to use the compiled, cached function in subsequent calls to `jax.jit`. However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined. This will cause unnecessary compilation each time in the loop:
```python
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()

'''
jit called in a loop with partials:
1 loop, best of 5: 192 ms per loop
jit called in a loop with lambdas:
10 loops, best of 5: 199 ms per loop
jit called in a loop with caching:
10 loops, best of 5: 21.6 ms per loop
'''
```


## JAX中的自动向量化

向量化的主要函数：`jax.vmap` 

举例说明：这里有一个处理两个一维向量的卷积的函数
```python
import jax
import jax.numpy as jnp

x = jnp.arange(5)
w = jnp.array([2., 3., 4.])

def convolve(x, w):
  output = []
  for i in range(1, len(x)-1):
    output.append(jnp.dot(x[i-1:i+2], w))
  return jnp.array(output)

convolve(x, w)
```

对于一个batch的w向量与x向量：
```
xs = jnp.stack([x, x])
ws = jnp.stack([w, w])
```


可以用手动的方法通过循环实现卷积

```python
def manually_batched_convolve(xs, ws):
  output = []
  for i in range(xs.shape[0]):
    output.append(convolve(xs[i], ws[i]))
  return jnp.stack(output)

manually_batched_convolve(xs, ws)

# 手动向量化
def manually_vectorized_convolve(xs, ws):
  output = []
  for i in range(1, xs.shape[-1] -1):
    output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
  return jnp.stack(output, axis=1)

manually_vectorized_convolve(xs, ws)
```
也可以使用vmap
```python
auto_batch_convolve = jax.vmap(convolve)
auto_batch_convolve(xs, ws)
```

If the batch dimension is not the first, you may use the `in_axes` and `out_axes` arguments to specify the location of the batch dimension in inputs and outputs. These may be an integer if the batch axis is the same for all inputs and outputs, or lists, otherwise
```python
auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

xst = jnp.transpose(xs)
wst = jnp.transpose(ws)

auto_batch_convolve_v2(xst, wst)
```
`jax.vmap` also supports the case where only one of the arguments is batched: for example, if you would like to convolve to a single set of weights `w` with a batch of vectors `x`; in this case the `in_axes` argument can be set to `None`:
```python
batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

batch_convolve_v3(xs, w)
```


jit和vmap可以混合使用




## JAX中的高级自动微分技术

`jax.jacfwd`和`jax.jacrev`是JAX（Just Another XLA）库中的两个函数，用于计算函数的前向和反向雅可比矩阵。JAX是一个用于高性能数值计算和自动微分的库，它基于XLA（Accelerated Linear Algebra）以及其他一些技术实现了自动微分和即时编译的功能。

- `jax.jacfwd`函数用于计算函数的前向雅可比矩阵（即梯度），它接受一个函数作为参数，并返回一个函数，该函数可以计算给定输入的梯度。前向模式自动微分是一种从输入到输出的逐个元素计算梯度的方法，通常用于输入维度较小的情况。
    
- `jax.jacrev`函数用于计算函数的反向雅可比矩阵，它接受一个函数作为参数，并返回一个函数，该函数可以计算给定输入的梯度。反向模式自动微分是一种从输出到输入的逐个元素计算梯度的方法，通常用于输出维度较小的情况。
    

这两个函数都是基于JAX中的自动微分功能实现的，它们可以帮助我们高效地计算函数的梯度，这在机器学习和优化算法中非常有用。使用这些函数，我们可以轻松地计算函数对输入的梯度，以便进行梯度下降、优化或其他涉及梯度的操作

```python
def square(x):
    return jnp.square(x)
x = jnp.array([1.0, 2.0, 3.0])
# 使用 jax.jacfwd 计算函数的前向雅可比矩阵（梯度）

grad_fwd = jax.jacfwd(square)
grad = grad_fwd(x)
print("Forward gradient:", grad)

# 使用 jax.jacrev 计算函数的反向雅可比矩阵（梯度）
grad_rev = jax.jacrev(square)
grad = grad_rev(x)
print("Reverse gradient:", grad)

'''
Forward gradient: [[2. 0. 0.] [0. 4. 0.] [0. 0. 6.]] 
Reverse gradient: [[2. 0. 0.] [0. 4. 0.] [0. 0. 6.]]
'''
```


