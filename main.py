import numpy as np
from sklearn.manifold import TSNE
from data_loader import *
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import os
import shutil

"""Feature Network"""


class FeatureNetwork(nn.Module):
    def __init__(self, pretrain=True):
        super(FeatureNetwork, self).__init__()
        model = models.resnet18(pretrained=pretrain)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


"""Obtain Feature Vectors"""


def obtain_f(max_point=5000):
    net.eval()
    data_loader = get_data_loader(batch_size=32)
    start = True
    count = 0
    with torch.no_grad():
        for data, _ in data_loader:
            if count >= max_point:
                break
            count += 1
            f = net(data.cuda())
            if start:
                f_all = f.cpu().data
                start = False
            else:
                f_all = torch.cat((f_all, f.cpu().data), dim=0)
    return f_all


"""Brushing The Network BatchNorm"""


def brush_bn(max_epoch=10):
    net.train()
    data_loader = get_data_loader(batch_size=32, shuffle=True)
    for _ in range(max_epoch):
        for data, _ in data_loader:
            _ = net(data.cuda())


def tsne_embed(data):
    """T-SNE"""
    ppl = 100
    tsne_model = TSNE(n_components=2, random_state=None, perplexity=ppl, verbose=2, learning_rate=100)
    embedded = tsne_model.fit_transform(data)
    return embedded


"""Clustering"""


def kmeans(data, num_classes=5):
    kmeans_labels = KMeans(n_clusters=num_classes, random_state=0).fit_predict(data)
    return kmeans_labels


num_cls = 5
data = np.load('./log/features_imgnet.npy')
labels = kmeans(data)


"""2D plot"""

pca = PCA(2)
data_simple = pca.fit_transform(data)
(fig, ax) = plt.subplots(1, 1, figsize=(30, 30))
colors = ['coral', 'orange', 'royalblue', 'purple', 'forestgreen']
for cls in range(num_cls):
    filtered_data = data_simple[labels == cls]
    ax.scatter(filtered_data[:, 0], filtered_data[:, 1], color=colors[cls], s=10, alpha=0.5)
fig.savefig('./tmp/kmeans_clustered_%d_2d.png' % num_cls)

"""3D plot"""

pca = PCA(3)
data_simple = pca.fit_transform(data)
fig = plt.figure(figsize=(30, 30))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
colors = ['coral', 'orange', 'royalblue', 'purple', 'forestgreen']
for cls in range(num_cls):
    filtered_data = data_simple[labels == cls]
    ax.scatter3D(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], color=colors[cls], s=5, alpha=0.7)
fig.savefig('./tmp/kmeans_clustered_%d_3d.png' % num_cls)


"""2D image plot"""


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


pca = PCA(2)
data_simple = pca.fit_transform(data)
(fig, ax) = plt.subplots(1, 1, figsize=(60, 60))
for cls in range(num_cls):
    filtered_data = data_simple[labels == cls]
    ax.scatter(filtered_data[:, 0], filtered_data[:, 1], color=colors[cls], s=10, alpha=0.8)

