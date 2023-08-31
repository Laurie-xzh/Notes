
# 2023.7.10~7.14

This week I read the code in `DeepModelZoo` 
my task was trainging RNN on the dataset given in `DeepModelZoo/Dataset/generate_dataset1_1.py` 
![[signal.png]]
On Tuesday and Wednesday, I took one period as a time step, the 20 points sampled in a period as the feature to train the model. 

| Param     | Val-Loss-Teacher | Val-Loss-AR |
| --------- | ---------------- | ----------- |
| Epoch=100 |                  |             |
| lr=0.001  | 0.0021           | 0.0036      |
| lr=0.01   | 4.375e-06        | 6.458e-06   |
| lr=0.05   | 1.345e-05        | 5.465e-06   |

When lr = 0.001, the difference is visibly noticeable. When lr-0.01 or 0.05, the original data is totally covered by predicted data
![[Pasted image 20230713202755.png]]



But this method cannot deal with non-periodic data. 

Later, I change the shape of input data, every point means a step(e.g., 120 points, the input shape is [1,120,1], [batch, time_seq, input_dim]) 





# 2023.7.17~2023.7.21


During the experiment, positional Encoding is found to make a big influence to the performance of Transformer. 

Two aspects are considered, one is how to generate the positional encode, the other is how to combine it with the input data.
| Number | Schedluer|Generate Method | Combine Method | Final test loss |
| ------ | --------------- | -------------- | --------------- |-----|
| 1       | |    sin2$\pi$             |       `x+PE`         |                 |



```python
def get_positional_encoding(d_model: int, max_len: int = 20):
    encodings = torch.zeros(max_len, d_model)
    position = torch.linspace(0, 2*torch.pi,max_len+1, dtype=torch.float32)[:-1,None]
    encodings = torch.sin(position)
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings
   
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int = 20):
        super().__init__()
        """ https://github.com/labmlai/annotated_deep_learning_paper_implementations/
        """
        self.dropout = nn.Dropout(dropout_prob)
        self.register_buffer('positional_encodings', get_positional_encoding(d_model, max_len), False)
    def forward(self, x: torch.Tensor):
        pe = self.positional_encodings[:x.shape[0]].detach().requires_grad_(False)
        x = torch.concat((x, pe), dim=-1)
        x = x + pe
        return x
```


$$\text{Relative MAE} = \frac{\text{MAE}}{\text{MAD}} = \frac{\sum_{i=1}^{n}|y_i - \hat{y}i|}{\sum_{i=1}^{n}|y_i - \bar{y}|}$$