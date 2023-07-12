
# RNN

`torch.nn.RNN()` 
the RNN block in PyTorch is an Elman RNN
Parameters:
- **input_size** – The number of expected features in the input x
- **hidden_size** – The number of features in the hidden state h
- **num_layers** – Number of recurrent layers. E.g., setting `num_layers=2` would mean stacking two RNNs together to form a stacked RNN, with the second RNN taking in outputs of the first RNN and computing the final results. Default: 1
- **nonlinearity** – The non-linearity to use. Can be either `'tanh'` or `'relu'`. Default: `'tanh'`
- **bias** – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **batch_first** – If `True`, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: `False`
- **dropout** – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to `dropout`. Default: 0
- **bidirectional** – If `True`, becomes a bidirectional RNN. Default: `False`

```python
# Create RNN Model

class RNNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        '''
        the shape of input x is (batch_size, seq_length, input_dim)
        '''
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) # here we only use the last time step of the sequence
        print(hn.shape)
        return out
# 定义模型参数
input_dim = 10
hidden_dim = 20
layer_dim = 2 # 几个RNN块
output_dim = 5

  

# 创建一个随机输入张量
batch_size = 3
seq_length = 4
x = torch.randn(batch_size, seq_length, input_dim)
# 创建RNN模型
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)
# 进行前向传播
output = model(x)
# 打印输出张量
print(output.shape)
```



Another method is using `torch.nn.RNNcell` ， this method can build RNN more flexiblly.
Parameters:
- **input_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.11)")) – The number of expected features in the input x
- **hidden_size** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.11)")) – The number of features in the hidden state h
- **bias** ([_bool_](https://docs.python.org/3/library/functions.html#bool "(in Python v3.11)")) – If `False`, then the layer does not use bias weights b_ih and b_hh. Default: `True`
- **nonlinearity** ([_str_](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.11)")) – The non-linearity to use. Can be either `'tanh'` or `'relu'`. Default: `'tanh'` 

Here is the method building a multi-layer elman RNN with RNNcell
```python
import torch
import torch.nn as nn

class MultiLayerRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MultiLayerRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn_cells = nn.ModuleList([nn.RNNCell(input_dim, hidden_dim)] + 
                                       [nn.RNNCell(hidden_dim, hidden_dim) for _ in range(num_layers-1)])

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        hidden_states = [torch.zeros(batch_size, self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        for seq in range(seq_length):
            xt = x[:, seq, :]

            for layer in range(self.num_layers):
                hidden_states[layer] = self.rnn_cells[layer](xt, hidden_states[layer])
                xt = hidden_states[layer]

        out = self.fc(hidden_states[-1])

        return out

```



# LSTM

LSTM中有3个门：forget gate, input gate, output gate

the input of current time step and the hidden state of the previous time step are dealed by the three gates, which are fully connect layer with sigmoid. 

![[Pasted image 20230707113153.png]]
where the $\bigodot$ is 'Hadamard product', which is an element-wise multiplication

Input gate controls the usage proportion of $\tilde{\mathbf{C}}_{t}$ , forget gate controls the usage of the memory $C_{t-1}$ 
The output gate regulates the information to be outputted from the cell state to the hidden state.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
# 定义模型参数
input_dim = 10
hidden_dim = 20
layer_dim = 2
output_dim = 5
# 创建一个随机输入张量
batch_size = 3
seq_length = 4
x = torch.randn(batch_size, seq_length, input_dim)

# 创建RNN模型
model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
# 进行前向传播
output = model(x)
# 打印输出张量
print(output.shape)
```





# GRU

GRU has less gates than LSTM. GRU has a reset gate and a update gate, while LSTM has three gates.

![[Pasted image 20230707113812.png]]




```python
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
# 定义模型参数
input_dim = 10
hidden_dim = 20
layer_dim = 2
output_dim = 5

# 创建一个随机输入张量
batch_size = 3
seq_length = 4
x = torch.randn(batch_size, seq_length, input_dim)
# 创建RNN模型
model = GRUModel(input_dim, hidden_dim, layer_dim, output_dim)
# 进行前向传播
output = model(x)
# 打印输出张量
print(output.shape)
```
