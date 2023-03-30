# -*- coding: utf-8 -*-
# @Time    : 2023/3/16 22:35
# @Author  : LIU YI

import copy
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.manifold import TSNE
from model import get_model
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import math

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    result = out.reshape(b, *((1,) * (len(x_shape) - 1)))
    return result

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):

        x = x.to(device)
        h = model(x)
        h = h.detach()
        # similarity = torch.matmul(h, h.t())
        # similarity /= torch.norm(h, dim=1)[:, None]
        # similarity /= torch.norm(h, dim=1)[None, :]
        #
        # # Plot heatmap
        # fig, ax = plt.subplots()
        # heatmap = ax.imshow(similarity.cpu().numpy(), cmap='Blues')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_title('Cosine Similarity')
        # cbar = fig.colorbar(heatmap, ax=ax)
        # plt.savefig('random_input.png')
        # plt.show()
        h = h.squeeze()
        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps = 1000,
        beta_schedule = 'cosine',
    ):
        super().__init__()

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps)


        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas.to('cuda'))
        register_buffer('alphas_cumprod', alphas_cumprod.to('cuda'))
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to('cuda'))

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to('cuda'))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to('cuda'))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to('cuda'))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to('cuda'))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance.to('cuda'))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)).to('cuda'))
        register_buffer('posterior_mean_coef1', (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to('cuda'))
        register_buffer('posterior_mean_coef2', ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).to('cuda'))

        # calculate p2 reweighting

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )


class MiniImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_label = None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if target_label == None:
                for image_name in os.listdir(class_dir):
                    image_path = os.path.join(class_dir, image_name)
                    self.images.append(image_path)
                    self.labels.append(self.class_to_idx[class_name])
            else:
                if self.class_to_idx[class_name] == target_label:
                    for image_name in os.listdir(class_dir):
                        image_path = os.path.join(class_dir, image_name)
                        self.images.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def tSNE(h, label, name):
    # color_dict = {'blue': 0, 'green': 1, 'yellow': 2, 'red': 3, 'brown': 4, 'gray': 5, 'gold': 6, 'pink': 7,
    #               'purple': 8, 'orange': 9, 'black': 10}
    # color_dict = {'red': 0,  'black': 1}
    color_dict = {}

    for i in range(5):
        # Define the color in RGB format
        red = int(255 - (i * 255/4))  # gradually decrease red channel from 255 to 0
        green = 0  # no green channel
        blue = 0
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i] = hex_color

    for i in range(5):
        # Define the color in RGB format
        red =  0# gradually decrease red channel from 255 to 0
        green = int(255 - (i * 255/4))   # no green channel
        blue = 0
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i+5] = hex_color

    for i in range(5):
        # Define the color in RGB format
        red =  0# gradually decrease red channel from 255 to 0
        green = 0   # no green channel
        blue = int(255 - (i * 255/4))
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i+10] = hex_color


    for i in range(5):
        # Define the color in RGB format
        red =  0# gradually decrease red channel from 255 to 0
        green = int(255 - (i * 255/4)) # no green channel
        blue = int(255 - (i * 255/4))
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i+15] = hex_color

    for i in range(5):
        # Define the color in RGB format
        red =  int(255 - (i * 255/4))# gradually decrease red channel from 255 to 0
        green = 0 # no green channel
        blue = int(255 - (i * 255/4))
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i+20] = hex_color

    for i in range(5):
        # Define the color in RGB format
        red =  int(255 - (i * 255/4)) # gradually decrease red channel from 255 to 0
        green = int(255 - (i * 255/4)) # no green channel
        blue = 0
        color = (red, green, blue)
        # Convert the color from RGB to hex format
        hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
        # Add the color to the dictionary
        color_dict[i+25] = hex_color


    data = np.array(h)
    target = np.array(label)
    tsne = TSNE(n_components=2, n_iter=500)

    data_tsne = tsne.fit_transform(data)
    x, y = data_tsne[:, 0], data_tsne[:, 1]

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    # current_axes = plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)

    color_target = [color_dict[c] for c in target]
    plt.scatter(x, y, c=color_target, s=10)
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    plt.savefig(name, dpi=1000)
    plt.show()


# Create the data loader

def test_result(test_loader, logreg, device):
    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    for k, v in metrics.items():
        print(f"{k}: {np.array(v).mean():.4f}")
    return np.array(metrics["Accuracy/test"]).mean()


def test_down_stream_acc(clip_symble = False):


    round = 199

    _, preprocess = clip.load("ViT-B/32", device='cuda')

    if clip_symble:
        model = _
    else:
        resnet = get_model('byol', 'resnet18', '2_layer', False, 'mini_imagenet')
        resnet = resnet.online_encoder
        resnet.load_state_dict(
            torch.load('./saved_models/simclr_weights_agg__1/simclr_weights_agg__1_global_model_r_{}.pth'.format(round),
                       map_location='cuda'))
        resnet = resnet.to('cuda')



    # resnet = get_model('byol', 'resnet18', '2_layer', False, 'mini_imagenet')
    # resnet = resnet.online_encoder
    # resnet.load_state_dict(torch.load('./saved_models/fedema_weights_agg__1/fedema_weights_agg__1_global_model_r_199.pth', map_location='cuda'))
    # resnet = resnet.to('cuda')

    _, preprocess = clip.load("ViT-B/32", device='cuda')


    # model, preprocess = clip.load("ViT-B/32", device='cuda')
    mini_imagenet_dataset = MiniImageNetDataset(root_dir='./mini_imagenet', transform=preprocess)
    mini_imagenet_dataloader = DataLoader(mini_imagenet_dataset, batch_size=500, shuffle=True, num_workers=0)
    features, labels = inference(mini_imagenet_dataloader, resnet, 'cuda')
    # features = np.load('feature_clip_MI.npy')
    # labels = np.load('label_clip_MI.npy')

    test_size = int(len(features)/6)
    test_images, test_labels = features[:test_size], labels[:test_size]
    train_images, train_labels = features[test_size:], labels[test_size:]
    train_images = torch.from_numpy(train_images).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_images = torch.from_numpy(test_images).float()
    test_labels = torch.from_numpy(test_labels).long()
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size= 256, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    logreg = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 100))
    logreg = logreg.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=3e-3)

    logreg.train()
    for epoch in range(200):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_dataloader):

            h = h.to('cuda')
            y = y.to('cuda')

            outputs = logreg(h)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        print(f"Epoch [{epoch}/{200}]: " + "\t".join(
            [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % 100 == 0:
            print("======epoch {}======".format(epoch))
            test_result(test_dataloader, logreg, 'cuda')

    result = test_result(test_dataloader, logreg, 'cuda')
    print(result)



def draw_heat_map(clip_symble = False):

    round = 139
    _, preprocess = clip.load("ViT-B/32", device='cuda')
    # state_dict =_.state_dict()

    # for param in state_dict:
    #     tep = torch.nn.init.xavier_uniform_(state_dict[param].unsqueeze(0).unsqueeze(0))
    #     state_dict[param] = tep.squeeze().squeeze()
    #
    # _.load_state_dict(state_dict)

    if clip_symble:
        model = _
    else:
        resnet = get_model('byol', 'resnet18', '2_layer', False, 'mini_imagenet')
        resnet = resnet.online_encoder
        resnet.load_state_dict(
            torch.load('./saved_models/fedema_weights_agg__1/fedema_weights_agg__1_global_model_r_{}.pth'.format(round),
                       map_location='cuda'))
        model = resnet.to('cuda')


    mini_imagenet_dataset = MiniImageNetDataset(root_dir='./mini_imagenet', transform=preprocess)
    mini_imagenet_dataloader = DataLoader(mini_imagenet_dataset, batch_size=50, shuffle=True, num_workers=0)
    (x, y) = iter(mini_imagenet_dataloader).next()

    x = torch.randn_like(x).to('cuda')

    with torch.no_grad():
        if clip_symble:
            h = model.encode_image(x)
        else:
            h = model(x)


    h = h.detach()

    similarity = torch.matmul(h, h.t())
    similarity /= torch.norm(h, dim=1)[:, None]
    similarity /= torch.norm(h, dim=1)[None, :]

    fig, ax = plt.subplots()
    heatmap = ax.imshow(similarity.cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    title = 'Cosine Similarity at round {}'.format(round)
    fontdict = {'fontsize': 17, 'color': 'red'}
    # ax.set_title(title, fontdict = fontdict)
    cbar = fig.colorbar(heatmap, ax=ax)
    if clip_symble:
        plt.savefig('random_input_clip_output.png')
    else:
        plt.savefig('random_input_BYOL_output.png')
    plt.tight_layout()
    plt.show()


def draw_test_plot(clip_symble = True):
    _, preprocess = clip.load("ViT-B/32", device='cuda')

    if clip_symble:
        model = _
    else:
        resnet = get_model('byol', 'resnet18', '2_layer', False, 'mini_imagenet')
        resnet = resnet.online_encoder
        resnet.load_state_dict(
            torch.load('./saved_models/simclr_weights_agg__1/simclr_weights_agg__1_global_model_r_199.pth',
                       map_location='cuda'))
        model = resnet.to('cuda')

    images = []
    labels = []
    steps = 7
    diffusion_process = GaussianDiffusion(timesteps=steps)
    for target_label in range(6):

        mini_imagenet_dataset = MiniImageNetDataset(root_dir='./mini_imagenet', transform=preprocess, target_label = random.randint(0,100))
        mini_imagenet_dataloader = DataLoader(mini_imagenet_dataset, batch_size=50, shuffle=True, num_workers=0)
        (x, y) = iter(mini_imagenet_dataloader).next()

        x = x.to('cuda')

        with torch.no_grad():
            if clip_symble:
                h = model.encode_image(x)
            else:
                h = model(x)


        images += [img for img in h]
        labels += [torch.tensor(target_label*5) for lbl in y]
        #
        # for step in range(steps-3):
        #     t = torch.ones((x.shape[0]))
        #     t *= (step)
        #     t = t.to(torch.int64)
        #     t = t.to('cuda')
        #     noise = torch.randn_like(x)
        #     noise.to('cuda')
        #     x = diffusion_process.q_sample(x_start=x, t=t, noise=noise)
        #
        #     with torch.no_grad():
        #         if clip_symble:
        #             h = model.encode_image(x)
        #         else:
        #             h = model(x)
        #
        #     images += [img for img in h]
        #     labels += [torch.tensor(target_label*5+step+1).to(torch.int32) for i in y]

        #
        # for i in range(50):
        #     images.append(torch.randn(images[0].shape))
        #     labels.append(torch.tensor(1))

    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)

    tSNE(images.detach().cpu(), labels, 'tep.png')
    center = torch.mean(images, dim=0)
    distances = torch.norm(images - center, dim=1)
    radius = torch.max(distances)
    print(radius)

    # h = h.detach()
    #
    # similarity = torch.matmul(h, h.t())
    # similarity /= torch.norm(h, dim=1)[:, None]
    # similarity /= torch.norm(h, dim=1)[None, :]
    #
    # fig, ax = plt.subplots()
    # heatmap = ax.imshow(similarity.cpu().numpy(), cmap='Blues')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title('Cosine Similarity')
    # cbar = fig.colorbar(heatmap, ax=ax)
    # if clip_symble:
    #     plt.savefig('random_input_clip_output.png')
    # else:
    #     plt.savefig('random_input_BYOL_output.png')
    # plt.show()


if __name__ == '__main__':
    seed = 61

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    test_down_stream_acc()
    # draw_test_plot(False)
    # draw_heat_map(True)