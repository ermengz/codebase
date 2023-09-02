"""
    python以及python接口api相关库的测试实验
    pytorch
    transform
    cuda
    tensorrt
    onnx
"""

def check_cuda():
    """torch 的gpu接口是否可用"""
    import torch
    is_avaliable = torch.cuda.is_available()
    print(f"gpu is avaliable: {is_avaliable}")
    print(f"gpu numbers: {torch.cuda.device_count()}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dumy_data = torch.randn(size=(2,3)).to(device)

    print(dumy_data)

def huggingface_test():
    """huggingface中的hello world"""
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis")
    print(f"init done")
    sentence_test = [
            "I like yuo.",
            "I hate you.",
            "爱我中华",
            "犯我中华者虽远必诛",
            "Love you love my dog."
        ]
    ret_dic = classifier(sentence_test)    

    for index, sen in enumerate(sentence_test):
        print(f"{sentence_test[index]}\t{ret_dic[index]['label']}\t{ret_dic[index]['score']}")
# output
# I like yuo.     POSITIVE        0.9996523857116699
# I hate you.     NEGATIVE        0.9992952346801758
# 爱我中华        NEGATIVE        0.8648549914360046
# 犯我中华者虽远必诛      NEGATIVE        0.9343968033790588   
# Love you love my dog.   POSITIVE        0.9998409748077393

# 可以看出，中文的识别效果不太理想 
    
    print(f"infer done")
    

def net_demo():
    """
        构建网络的三种基本方式, 模型容器：
        # nn.Sequential() 
        # nn.ModuleList()
        # nn.ModuleDict()
        相同点: 都在nn模块下
        不同点: sequential函数, 内置了forward功能
                modulelist和moduledict需要写forward函数
        他们之间的转换关系为：modulelist --> sequential
                            moduledict --> sequential
    """
    import torch
    import torch.nn as nn

    dumy_data = torch.rand([3,224,224])
    dumy_data = dumy_data.unsqueeze(0)
    print(f'dumy_data = {dumy_data.shape}')
    net1 = nn.Sequential(nn.Conv2d(3,32,3,3),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                        )
    print(f"net1:{net1}")
    output1 = net1(dumy_data)
    print(f"nn.Sequential, output shape:{output1.shape}")

    class netM2(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super(netM2,self).__init__(*args, **kwargs)

            self.block = nn.ModuleList([nn.Conv2d(3,32,3,3),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU()])
        def forward(self,x):
            for layer in self.block:
                x = layer(x)
            return x
    net2 = netM2()
    print(f"net2:{net2}")
    output2 = net2(dumy_data)
    print(f"nn.ModuleList, output shape:{output2.shape}")

    class netM3(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super(netM3,self).__init__(*args, **kwargs)

            self.block = nn.ModuleDict({"conv":nn.Conv2d(3,32,3,3),
                                        "bn":nn.BatchNorm2d(32),
                                        "relu":nn.ReLU()})
        def forward(self,x):
            for layer in self.block.values() :
                x = layer(x)
            return x
    net3 = netM3()
    print(f"net3:{net3}")
    output3 = net3(dumy_data)
    print(f"nn.ModuleDict, output shape:{output3.shape}")

    # 转换关系
    module_Dict = nn.ModuleDict({
                                'conv': nn.Conv2d(10, 10, 3),
                                'pool': nn.MaxPool2d(3)
                                })
    net4 = nn.Sequential(*module_Dict.values())
    module_Dict=nn.ModuleList([nn.Linear(32,64),nn.ReLU()])
    net5 = nn.Sequential(*module_Dict)
    print(f"net4:{net4}")
    print(f"net5:{net5}")

if __name__ =="__main__":
    # check_cuda()
    # huggingface_test()
    print(f"start {'*'*20}")
    net_demo()
    print(f"end {'*'*20}")
