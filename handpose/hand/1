import caffe

import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import time

def load_image(imgfile):
    import caffe
    image = caffe.io.load_image(imgfile)
    transformer = caffe.io.Transformer({'data': (1, 3, 512, 512)})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))
    transformer.set_raw_scale('data', 512)
    transformer.set_channel_swap('data', (2, 1, 0))

    image = transformer.preprocess('data', image)
    image = image.reshape(1, 3, 512, 512)
    return image


class CaffeDataLoader:
    def __init__(self, protofile):
        caffe.set_mode_cpu()
        self.net = caffe.Net(protofile, 'aaa', caffe.TRAIN)

    def next(self):
        output = self.net.forward()
        data = self.net.blobs['data'].data
        label = self.net.blobs['label'].data
        return data, label

def create_network(protofile, weightfile):
    net = caffe.Net(protofile)
    #if args.cuda:
        #net.cuda()
    print(net)
    net.load_weights(weightfile)
    net.train()
    return net

def forward_network(net, data, label):
    data = torch.from_numpy(data)
    label = torch.from_numpy(label)
    if args.cuda:
        data = Variable(data.cuda())
        label = Variable(label.cuda())
    else:
        data = Variable(data)
        label = Variable(label)
    blobs = net(data, label)
    return blobs
#data_protofile = ''
#imgfile = '/home/SENSETIME/duanyiqun/Downloads/pytorch-caffe-master/data/cat.jpg'
net_protofile = 'pose_deploy.prototxt'
weightfile = 'pose_iter_102000.caffemodel'

    #data_loader = CaffeDataLoader(data_protofile)
net = create_network(net_protofile, weightfile)
net.set_verbose(False)

print(net.models)


#img = load_image(imgfile)
net.eval()
#blobs = net(img)

torch.save(net, 'model.pkl')

newnet = torch.load('model.pkl')

