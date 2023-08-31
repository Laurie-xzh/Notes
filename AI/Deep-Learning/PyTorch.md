这里记录一些常用的代码集锦

# 基本的配置


下面是一些包的导入和查询指令
```python
import torch  
import torch.nn as nn  
import torchvision  
print(torch.__version__)  
print(torch.version.cuda)  
print(torch.backends.cudnn.version())  
print(torch.cuda.get_device_name(0))
```

下面是一些设置
```python
np.random.seed(0)  
torch.manual_seed(0)  
torch.cuda.manual_seed_all(0)  
  
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False
```
随机性源于随机种子：在机器学习和深度学习中，许多操作（如参数初始化、数据划分、采样方法等）会涉及到随机性，这些随机性通常依赖于伪随机数生成器（Pseudorandom Number Generator，PRNG）。PRNG是一种基于随机种子的算法，通过种子生成一个序列的数字，看起来像是随机的。
随机种子的作用：通过固定随机种子，可以确保每次运行时生成的随机数序列是相同的。也就是说，相同的随机种子将产生相同的随机数序列，这样可以保证在同一设备上的不同运行实例具有相同的随机性。
PyTorch和NumPy的随机种子：PyTorch和NumPy库都使用随机数生成器来执行一些随机操作，如参数初始化、数据采样等。这些库提供了设置随机种子的功能，通过固定这些库的随机种子，可以确保这些库生成的随机数序列在同一设备上的多次运行中是相同的。


GPU设置
```python
# Device configuration  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# use multi GPU
import osos.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 也可以在命令行运行代码时设置显卡：
CUDA_VISIBLE_DEVICES=0,1 python train.py
# 清楚显存
torch.cuda.empty_cache()
# 在命令行重置GPU指令
nvidia-smi --gpu-reset -i [gpu_id]
```


# 张量(Tensor)

张量类型表格
![[Pasted image 20230706165205.png]]

命名张量
张量命名是一个非常有用的方法，这样可以方便地使用维度的名字来做索引或其他操作，大大提高了可读性、易用性，防止出错。
```python
# 在PyTorch 1.3之前，需要使用注释  
# Tensor[N, C, H, W]  
images = torch.randn(32, 3, 56, 56)  
images.sum(dim=1)  
images.select(dim=1, index=0)  
  
# PyTorch 1.3之后  
NCHW = [‘N’, ‘C’, ‘H’, ‘W’]  
images = torch.randn(32, 3, 56, 56, names=NCHW)  
images.sum('C')  
images.select('C', index=0)  
# 也可以这么设置  
tensor = torch.rand(3,4,1,2,names=('C', 'N', 'H', 'W'))  
# 使用align_to可以对维度方便地排序  
tensor = tensor.align_to('N', 'C', 'H', 'W')
```


类型转换
```python
# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor  
torch.set_default_tensor_type(torch.FloatTensor)  
  
# 类型转换  
tensor = tensor.cuda()  
tensor = tensor.cpu()  
tensor = tensor.float()  
tensor = tensor.long()

# torch.Tensor与np.ndarray转换
ndarray = tensor.cpu().numpy()  
tensor = torch.from_numpy(ndarray).float()  
tensor = torch.from_numpy(ndarray.copy()).float() # If ndarray has negative stride.

# torch.tensor与PIL.Image转换
# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化  
# torch.Tensor -> PIL.Image  
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())  
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way  
  
# PIL.Image -> torch.Tensor  
path = r'./figure.jpg'  
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equivalently way

# np.ndarray与PIL.Image的转换
image = PIL.Image.fromarray(ndarray.astype(np.uint8))  
ndarray = np.asarray(PIL.Image.open(path))

# 从只包含一个元素的张量中提取值
value = torch.rand(1).item()

# 张量形变
# 在将卷积层输入全连接层的情况下通常需要对张量做形变处理，  
# 相比torch.view，torch.reshape可以自动处理输入张量不连续的情况  
  
tensor = torch.rand(2,3,4)  
shape = (6, 4)  
tensor = torch.reshape(tensor, shape)

# 打乱顺序
tensor = tensor[torch.randperm(tensor.size(0))]  # 打乱第一个维度

# 水平翻转
# pytorch不支持tensor[::-1]这样的负步长操作，水平翻转可以通过张量索引实现  
# 假设张量的维度为[N, D, H, W].  
  
tensor = tensor[:,:,:,torch.arange(tensor.size(3) - 1, -1, -1).long()]


# 复制张量
# Operation                 |  New/Shared memory | Still in computation graph |  
tensor.clone()            # |        New         |          Yes               |  
tensor.detach()           # |      Shared        |          No                |  
tensor.detach.clone()()   # |        New         |          No                |

# 张量拼接
'''  
注意torch.cat和torch.stack的区别在于torch.cat沿着给定的维度拼接，  
而torch.stack会新增一维。例如当参数是3个10x5的张量，torch.cat的结果是30x5的张量，  
而torch.stack的结果是3x10x5的张量。  
'''  
tensor = torch.cat(list_of_tensors, dim=0)  
tensor = torch.stack(list_of_tensors, dim=0)

# 将整数标签转换为one-hot编码
# pytorch的标记默认从0开始  
tensor = torch.tensor([0, 2, 1, 3])  
N = tensor.size(0)  
num_classes = 4  
one_hot = torch.zeros(N, num_classes).long()  
one_hot.scatter_(dim=1, index=torch.unsqueeze(tensor, dim=1), src=torch.ones(N, num_classes).long())

# 得到非零元素
torch.nonzero(tensor)               # index of non-zero elements  
torch.nonzero(tensor==0)            # index of zero elements  
torch.nonzero(tensor).size(0)       # number of non-zero elements  
torch.nonzero(tensor == 0).size(0)  # number of zero elements

# 判断两个张量相等
torch.allclose(tensor1, tensor2)  # float tensor  
torch.equal(tensor1, tensor2)     # int tensor

# 张量扩展
# Expand tensor of shape 64*512 to shape 64*512*7*7.  
tensor = torch.rand(64,512)  
torch.reshape(tensor, (64, 512, 1, 1)).expand(64, 512, 7, 7)

# 矩阵乘法
# Matrix multiplcation: (m*n) * (n*p) * -> (m*p).  
result = torch.mm(tensor1, tensor2)  
  
# Batch matrix multiplication: (b*m*n) * (b*n*p) -> (b*m*p)  
result = torch.bmm(tensor1, tensor2)  
  
# Element-wise multiplication.  
result = tensor1 * tensor2


```

```python
assert tensor.size() == (N, D, H, W) 作为调试手段，确保张量维度和你设想中一致。
```



```python
torch.chunk(_input_, _chunks_, _dim=0_)
```
- **input** ([_Tensor_](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – the tensor to split
- **chunks** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.11)")) – number of chunks to return
- **dim** ([_int_](https://docs.python.org/3/library/functions.html#int "(in Python v3.11)")) – dimension along which to split the tensor
Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
# 模型

```python
class BN(torch.nn.Module)  
    def __init__(self):  
        ...  
        self.register_buffer('running_mean', torch.zeros(num_features))  
  
    def forward(self, X):  
        ...  
        self.running_mean += momentum * (current - self.running_mean)
```


关于基本的模型的定义：
```python
# 两层卷积
class ConvNet(nn.Module):  
    def __init__(self, num_classes=10):  
        super(ConvNet, self).__init__()  
        self.layer1 = nn.Sequential(  
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(16),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2))  
        self.layer2 = nn.Sequential(  
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),  
            nn.BatchNorm2d(32),  
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2))  
        self.fc = nn.Linear(7*7*32, num_classes)  
  
    def forward(self, x):  
        out = self.layer1(x)  
        out = self.layer2(out)  
        out = out.reshape(out.size(0), -1)  
        out = self.fc(out)  
        return out  
  
model = ConvNet(num_classes).to(device)
```

训练
```python
# Loss and optimizer  
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
  
# Train the model  
total_step = len(train_loader)  
for epoch in range(num_epochs):  
    for i ,(images, labels) in enumerate(train_loader):  
        images = images.to(device)  
        labels = labels.to(device)  
  
        # Forward pass  
        outputs = model(images)  
        loss = criterion(outputs, labels)  
  
        # Backward and optimizer  
        optimizer.zero_grad()  
        loss.backward()  
        optimizer.step()  
  
        if (i+1) % 100 == 0:  
            print('Epoch: [{}/{}], Step: [{}/{}], Loss: {}'  
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
```




测试
```python
# Test the model  
model.eval()  # eval mode(batch norm uses moving mean/variance   
              #instead of mini-batch mean/variance)  
with torch.no_grad():  
    correct = 0  
    total = 0  
    for images, labels in test_loader:  
        images = images.to(device)  
        labels = labels.to(device)  
        outputs = model(images)  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()  
  
    print('Test accuracy of the model on the 10000 test images: {} %'  
          .format(100 * correct / total))
```

自定义loss
继承torch.nn.Module类写自己的loss
```python
class MyLoss(torch.nn.Moudle):  
    def __init__(self):  
        super(MyLoss, self).__init__()  
  
    def forward(self, x, y):  
        loss = torch.mean((x - y) ** 2)  
        return loss
```

# 数据

预处理
```python
train_transform = torchvision.transforms.Compose([  
    torchvision.transforms.RandomResizedCrop(size=224,  
                                             scale=(0.08, 1.0)),  
    torchvision.transforms.RandomHorizontalFlip(),  
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),  
                                     std=(0.229, 0.224, 0.225)),  
 ])  
 val_transform = torchvision.transforms.Compose([  
    torchvision.transforms.Resize(256),  
    torchvision.transforms.CenterCrop(224),  
    torchvision.transforms.ToTensor(),  
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),  
                                     std=(0.229, 0.224, 0.225)),  
])
```
