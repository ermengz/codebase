"""torch深度学习框架的基本模块

"""
# 网络相关模块
import torch
import torch.nn as nn
# 数据读取模块
from torch.utils.data import DataLoader
# 数据集模块
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np

# torch         模块
# nn            网络模块
# torchvision   视觉模块
# 
trainbatch = 64
train_data = datasets.FashionMNIST("../datasets/",
                                   train=True,
                                   download=True,
                                   transform=ToTensor())
test_data = datasets.FashionMNIST("../datasets/",
                                  train=False,
                                  download=True,
                                  transform=ToTensor())
trainloader = DataLoader(train_data,batch_size=trainbatch,shuffle=True)
testloader = DataLoader(test_data,batch_size=trainbatch,shuffle=False)


# 可视化
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
if 0:
    print(len(train_data))
    dataiter = iter(trainloader)
    images, labels = dataiter._next_data()
    images = images.numpy()
    labels = labels.numpy()
    print("target shape: {}".format(images.shape))
    print("label shape: {}".format(labels.shape))
    fig = plt.figure(figsize=(10, 4))
    for idx in np.arange(trainbatch):
        ax = fig.add_subplot(2, trainbatch//2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(classes[labels[idx]])
    fig.savefig("FashionMNIST_data.jpg")

# 构建网络

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

class fistNN(nn.Module):
    def __init__(self):
        super(fistNN,self).__init__()
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        output = self.backbone(x)
        return output

model = fistNN().to(device=device)
print(model)
# 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

def train_setp(trainloader,model,loss_fn,optimizer):
    size = len(trainloader)
    model.train()
    for batch, (x,y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        output = model(x)
        loss = loss_fn(output,y)

        # backward
        optimizer.zero_grad()   #  清除累积梯度
        loss.backward()         #  反向传播
        optimizer.step()        #  梯度更新

        # visiable
        if batch % 100 ==0:
            loss, current = loss.item(), batch*len(x)
            print(f"loss: {loss:>7f}, [{current:>7d} / {size*len(x):>7d}]")


def test_setp(dataloader, model, loss_fn):
    size = len(dataloader)
    datasize = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0.,0.
    with torch.no_grad():
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pre = model(x)
            test_loss += loss_fn(pre,y).item() 
            correct += (pre.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= size
    correct /= datasize
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epoch = 10
for e in range(epoch):
    print(f"Epoch {e+1} \n-----------------------")
    train_setp(trainloader, model, loss_fn, optimizer)
    test_setp(testloader, model, loss_fn)
print("Done!")

## save model
torch.save(model.state_dict(),"../checkpoints/firstNN.pth")
print(f"save model")

## test load
model = fistNN()
model.load_state_dict(torch.load("../checkpoints/firstNN.pth"))
model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted} \nActual: {actual}")
