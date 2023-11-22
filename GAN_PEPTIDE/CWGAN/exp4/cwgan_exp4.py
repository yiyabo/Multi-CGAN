import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
BATCHSIZE=64
def to_onehot(file_path):
    # file_path:保存index的npy文件
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
        self.disc = nn.Sequential(
            # input: N x channels_img x 30*21
            nn.Conv2d(channels_img, features_d, kernel_size=(10,1), stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 6, 1, 0),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
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

    def forward(self, x):
        x=x.reshape(-1,1,30,21)
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 6, 1, 0),  # img: 21*21
            self._block(features_g * 2, channels_img, (10,1), 1, 0),  # img: 30*21
            # nn.ConvTranspose2d(
            #     features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            # ),
            # Output: N x channels_img x 30*21
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

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
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



# train_data=np.load('../data/train_c.npy')[0:1529,:]
# np.save('../data/pos_c.npy',train_data)
# train_data=to_onehot('../data/pos_c.npy')
# train_data=np.array(train_data)
# train_data=torch.Tensor(train_data)
# # print(train_data.shape)
# # print(train_data.shape[0])
# labels=np.ones((train_data.shape[0],1))
# # labels=[i for i in range(train_data.shape[0])]
# # labels=np.array(labels)
# # print(labels)
# labels=torch.Tensor(labels)
# train_iter=load_array((train_data,labels),BATCHSIZE)

# N, in_channels, H, W = 8, 1, 30, 21
# noise_dim = 100
# x = torch.randn((N, in_channels, H, W))
# # print(x.shape)
# disc = Discriminator(in_channels, 8)
# # print(disc(x).shape)
# assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
# gen = Generator(noise_dim, in_channels, 8)
# z = torch.randn((N, noise_dim, 1, 1))
# assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
# print(gen(z).shape)
# def main():
    # N, in_channels, H, W = 8, 3, 64, 64
    # noise_dim = 100
    # x = torch.randn((N, in_channels, H, W))
    # disc = Discriminator(in_channels, 8)
    # assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    # gen = Generator(noise_dim, in_channels, 8)
    # z = torch.randn((N, noise_dim, 1, 1))
    # assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"

import torch
import torch.nn as nn
import torch.optim as optim

# from WGAN.utils import gradient_penalty, save_checkpoint, load_checkpoint
# from WGAN.WGAN import Discriminator, Generator, initialize_weights,to_onehot,return_index,load_array
import numpy as np
import json
# Hyperparameters etc.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = (30,21)
CHANNELS_IMG = 1
Z_DIM = 90
NUM_EPOCHS = 100
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 1
LAMBDA_GP = 10
train_data=to_onehot('newdata/neg.npy')
train_data=np.array(train_data)
train_data=torch.Tensor(train_data)
# print(train_data.shape)
# print(train_data.shape[0])
labels=np.ones((train_data.shape[0],1))
# labels=[i for i in range(train_data.shape[0])]
# labels=np.array(labels)
# print(labels)
labels=torch.Tensor(labels)
loader=load_array((train_data,labels),BATCH_SIZE)
# transforms = transforms.Compose(
#     [
#         transforms.Resize(IMAGE_SIZE),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
#     ]
# )
#
# dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
# comment mnist above and uncomment below for training on CelebA
#dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
# )

# initialize gen and disc, note: discriminator should be called critic,
# according to WGAN paper (since it no longer outputs between [0, 1])
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

# for tensorboard plotting
fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
# writer_real = SummaryWriter(f"logs/GAN_MNIST/real")
# writer_fake = SummaryWriter(f"logs/GAN_MNIST/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    # Target labels not needed! <3 unsupervised
    # print(11111)
    for batch_idx, (real, _) in enumerate(loader):
        real=real.unsqueeze(1)
        real = real.to(device)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, real, fake, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        # print(
        #     f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
        #                   Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        # )
        # Print losses occasionally and print to tensorboard
        # if batch_idx % 20 == 0 and batch_idx > 0:
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] \
              Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
        )

        # with torch.no_grad():
        #     fake = gen(fixed_noise)
        #     # take out (up to) 32 examples
        #     img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        #     img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
        #
        #     writer_real.add_image("Real", img_grid_real, global_step=step)
        #     writer_fake.add_image("Fake", img_grid_fake, global_step=step)

        # step += 1
    if (epoch + 1) % 50 == 0:
        torch.save({'model': gen.state_dict()}, 'gan_exp4_neg' + (str)(epoch) + '.pth')
# noise2 = torch.randn(20000, Z_DIM, 1, 1).to(device)
# fake2 = gen(noise2)
# fake2=fake2.clone().detach()
# fake2=fake2.squeeze(1)
# for i in range(fake2.shape[0]):
#     fake2[i]=props_to_onehot(fake2[i])
# fake2=fake2.cpu().detach().numpy()
# # fake2 = np.array(fake2)
# # # np.save('../data/gen_wgan_data.npy',fake2)
# # coding = return_index(fake2)
# laten = {}
# seq = []
# laten_code = []
# np.save('../data/gen_wgan_data.npy', fake2)
# wgan_index = np.load('../data/gen_wgan_data.npy')
# np.save('../data/gen_wgan_data.npy',return_index(wgan_index))
# coding = np.load('../data/gen_wgan_data.npy')
# noise2 = noise2.squeeze()
# noise2 = noise2.squeeze()
# def create_dict(words):
#     word_dict={}
#     word_index=0
#     for word in words:
#         word_dict[word]=word_index
#         word_index+=1
#     return word_dict
# def creat_dict_word(word_dict,words):
#     word_num_dict={}
#     for word in words:
#         word_num_dict[word_dict[word]]=word
#     return word_num_dict
# words=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']
#
# word_dict=create_dict(words)
# num_dict = creat_dict_word(word_dict,words)
# # w = []
# for code in coding :
#     strs = ""
#     for c in code:
#         if c != ' ':
#             strs = strs + num_dict[c]
#     seq.append(strs)
# # for i in range(len(noise2 )):
# #     laten[w[i]] = noise2[i].cpu().detach()
# np.save('seq.npy',coding)
# np.save('laten_codes.npy',noise2.cpu().detach())
