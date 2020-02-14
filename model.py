import torch
import torchvision
import torch.nn as nn

class TinyVgg(nn.Module):
    def __init__(self):
        super(TinyVgg,self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.ReLU(),
            nn.Conv2d(64,64,3,stride=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(64,128,3),
            nn.ReLU(),
            nn.Conv2d(128,128,3,stride=2,dilation=2),
            nn.ReLU(),
            nn.Conv2d(128,256,3),
            nn.ReLU(),
            nn.Conv2d(256,256,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.Softmax()
        )
    def forward(self, x):
        return self.seq(x)
if __name__ =='__main__':
    vgg = TinyVgg()
    print(vgg)
    output = vgg(torch.randn(3,32,32).unsqueeze(0))
    print(output.squeeze(), output.type())