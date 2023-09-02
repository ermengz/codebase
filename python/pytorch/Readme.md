
# notes

## 语法

1. 模块

```python
    import torch                # 导入模块
    import torch.nn as nn       # 模块重命名
    from torchvision import datasets # 导入模块内的类、函数、变量
```

2. 梯度更新
    
    梯度的更新和清零由优化器完成，反向传播时损失的反向传播。

    + optimizer.zero_grad()   #  清除累积梯度
    + loss.backward()         #  反向传播
    + optimizer.step()        #  梯度更新
