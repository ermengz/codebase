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
    

if __name__ =="__main__":
    # check_cuda()
    huggingface_test()
