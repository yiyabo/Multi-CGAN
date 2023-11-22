import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
BATCHSIZE=256
def to_onehot(file_path):
    
    pep = np.load(file_path)
    # print(pep)
    pep = torch.LongTensor(pep)
    # print(pep[0])
    one_hot_pep = F.one_hot(pep, 21).reshape(-1, 30, 21)
    return one_hot_pep
def return_index(one_hot_coding):
    # one_hot = one_hot_coding.numpy()
    index = np.argwhere(one_hot_coding == 1)
    return index[:, -1].reshape(-1, 30)

def load_array(data_arrays, batch_size, is_train=True):

    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.emb = self._emb(11, 20)
        self.linear = nn.Sequential(
            nn.Linear(30*21+20, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 30 * 21),
        )
        self.disc = nn.Sequential(
           
            nn.Conv2d(channels_img, features_d, kernel_size=(10,1), stride=1, padding=0),
            nn.LeakyReLU(0.2),
           
            self._block(features_d, features_d * 2, 6, 1, 0),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
           
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def _emb(self, label, dim):
        return nn.Embedding(label, dim)
    def forward(self, x,label):
        label = label.squeeze().squeeze()
        label = self.emb(label)
        # label = label.unsqueeze(2)
        # label = label.unsqueeze(3)
        x=x.view(-1,30*21)
        # label=self.emb(label)
        x = torch.cat([x,label],1)
        x= self.linear(x)
        x = x.reshape(-1, 1, 30, 21)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.emb = self._emb(11,20)
        self.net = nn.Sequential(
            # N x channels_noise x 1 x 1

            self._block(channels_noise+20, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  #  8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  #  16x16
            self._block(features_g * 4, features_g * 2, 6, 1, 0),  #  21*21
            self._block(features_g * 2, channels_img, (10,1), 1, 0),  # 30*21
            # nn.ConvTranspose2d(
            #     features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            # ),

            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def _emb(self,label,dim):
        return nn.Embedding(label,dim)

    def forward(self, x,label):
        label=label.squeeze().squeeze()
        label = self.emb(label)
        if len(label) == 20:
                label = label.unsqueeze(0)
        # print(label.shape)
        label=label.unsqueeze(2)
        label = label.unsqueeze(3)
        # print(label.shape)
        # print(x.shape)
        x = torch.cat([x,label],1)
        return self.net(x)


def initialize_weights(model):

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic,label, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)


    mixed_scores = critic(interpolated_images,label)


    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


def save_checkpoint(state, filename="celeba_wgan_gp.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, gen, disc):
    print("=> Loading checkpoint")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])

def props_to_onehot(props):
    if isinstance(props, list):
        props = np.array(props)
    props=props.cpu().detach().numpy()
    a = np.argmax(props, axis=1)
    b = np.zeros((len(a), props.shape[1]))
    b[np.arange(len(a)), a] = 1
    return torch.FloatTensor(b)




import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np
import json

device = "cuda:2" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
IMAGE_SIZE = (30,21)
CHANNELS_IMG = 1
Z_DIM = 90
NUM_EPOCHS = 200
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
torch.cuda.set_device(2)
train1=to_onehot('data/struct_pos.npy')
train_data=to_onehot('data/struct_pos.npy')
train2=to_onehot('data/toxicity_pos.npy')

train_data=torch.cat([train_data,train2])
train3=to_onehot('data/amp_pos.npy')

train_data=torch.cat([train_data,train3])
train4=to_onehot('data/struct_neg.npy')

train_data=torch.cat([train_data,train4])
train5=to_onehot('data/toxicity_neg.npy')

train_data=torch.cat([train_data,train5])
train6=to_onehot('data/amp_neg.npy')

train_data=torch.cat([train_data,train6])
train7=to_onehot('data/toxicity_amp.npy')

train_data=torch.cat([train_data,train7])
train8=to_onehot('data/struct_amp.npy')

train_data=torch.cat([train_data,train8])
train9=to_onehot('data/struct_toxicity.npy')

train_data=torch.cat([train_data,train9])
train10=to_onehot('data/struct_toxicity_amp.npy')

train_data=torch.cat([train_data,train10])
train11=to_onehot('data/ntoxicity_namp.npy')

train_data=torch.cat([train_data,train11])
train_data=np.array(train_data)
train_data=torch.Tensor(train_data)
labels=np.empty((train1.shape[0],1))
labels.fill(0)
labels2=np.empty((train2.shape[0],1))
labels2.fill(1)
labels3=np.empty((train3.shape[0],1))
labels3.fill(2)
labels4=np.empty((train4.shape[0],1))
labels4.fill(3)
labels5=np.empty((train5.shape[0],1))
labels5.fill(4)
labels6=np.empty((train6.shape[0],1))
labels6.fill(5)
labels7=np.empty((train7.shape[0],1))
labels7.fill(6)
labels8=np.empty((train8.shape[0],1))
labels8.fill(7)
labels9=np.empty((train9.shape[0],1))
labels9.fill(8)
labels10=np.empty((train10.shape[0],1))
labels10.fill(9)
labels11=np.empty((train11.shape[0],1))
labels11.fill(10)
labels=np.concatenate((labels,labels2,labels3,labels4,labels5,labels6,labels7,labels8,labels9,labels10,labels11))

labels=torch.LongTensor(labels)
loader=load_array((train_data,labels),BATCH_SIZE)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)


opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

step = 0

gen.train()
critic.train()

lossg=[]
lossd=[]
num = 0
for epoch in range(NUM_EPOCHS):
   

    for batch_idx, (real,label) in enumerate(loader):
        real=real.unsqueeze(1)
        real = real.to(device)
        cur_batch_size = real.shape[0]
        label = label.unsqueeze(1).to(device)
       
        num += 1
       
        for _ in range(CRITIC_ITERATIONS):

            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise,label)
           
            critic_real = critic(real,label).reshape(-1)
            critic_fake = critic(fake,label).reshape(-1)
            gp = gradient_penalty(critic,label, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        gen_fake = critic(fake,label).reshape(-1)


        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
       
        if batch_idx % 20 == 0 and batch_idx > 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

           

            step += 1


    if (epoch + 1) % 50 == 0:
        torch.save({'model': gen.state_dict()}, '_' + (str)(epoch + 1) + '.pth')

