import torch
import torchvision
from dataset import MyDataSet
from model import TinyVgg
import torch.nn as nn
import torch.optim as optim

# make dataset
ds = MyDataSet('cifar-10-batches-py', 'train')
# data loader
dl = torch.utils.data.DataLoader(dataset=ds,batch_size= 8, shuffle=True,num_workers=0,drop_last=True)
# model network( collect of module)
vgg = TinyVgg()
criterion = nn.MSELoss()
optimizer = optim.Adam(
            params=vgg.parameters(),
            lr=float(0.0001),
            betas=(0.0, 0.9)
        )
def one_hot(x,num_class):
    size = (x.size()[0],num_class)
    t = torch.zeros(size)
    for i,v in enumerate(t):
        v[x[i].long()] = 1
    return t
import matplotlib.pyplot as plt
for i , batch in enumerate(dl):
    print(batch[0][0])
    plt.imshow(batch[0][0])
    plt.show()
    break
    # optimizer.zero_grad()
    #
    # output = vgg(batch[0])
    # label = one_hot(batch[1], 10)
    # loss = criterion(output, label)
    # print('loss')
    # print(loss)
    # loss.backward()
    # optimizer.step()

