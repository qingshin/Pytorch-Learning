# 总

我做的不多，只对以下参数经行调试

- 学习率
- LSTM层的隐藏单元
- epochs

最好结果

RMSE：10.37
LOSS：102

**我的内容参考意义不大:thumbsdown:**

**请移步至 `lfq_best.ipynb` 进行学习:thumbsup:**



## 更改学习率

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 64)
        self.lstm2 = nn.LSTM(64,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 30

LOSS 基本上都在110左右

| 学习率 | RMSE  |
| :----: | :---: |
| 0.001  | 10.76 |
| 0.002  | 10.68 |
| 0.003  | 10.62 |
| 0.004  | 10.70 |
| 0.0045 | 10.64 |
| 0.005  | 10.62 |
| 0.0055 | 10.77 |
| 0.006  | 10.95 |
| 0.007  | 11.05 |

## 更改LSTM层的隐藏单元

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 128)
        self.lstm2 = nn.LSTM(128,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 20

RMSE：11.07
LOSS：120



**目前最好**

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 32)
        self.lstm2 = nn.LSTM(32,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 20

RMSE：10.48
LOSS：109

epochs = 30

RMSE：10.45
LOSS：109

epochs = 63

RMSE：10.37
LOSS：102





```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 32)
        self.lstm2 = nn.LSTM(32,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

epochs = 30

RMSE：10.52
LOSS：110



```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 32)  # 9个输入特征
        self.lstm1 = nn.LSTM(32, 16)
        self.lstm2 = nn.LSTM(16,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 20

RMSE：10.83
LOSS：114

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 32)
        self.lstm2 = nn.LSTM(32,16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 30

RMSE：10.49
LOSS：110



```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 64)  # 9个输入特征
        self.lstm1 = nn.LSTM(64, 24)
        self.lstm2 = nn.LSTM(24,64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # 输出PM2.5的预测值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x, _ = self.lstm1(x.unsqueeze(1))  # 将输入形状调整为 (batch_size, seq_len, input_size)
        x, _ = self.lstm2(x)  # 不需要再次增加维度
        x = torch.relu(x.squeeze(1))  # 去除seq_len维度
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
model = Net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
```

epochs = 30

RMSE：10.59
LOSS：112











