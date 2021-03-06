#!/usr/bin/python
# encoding: utf-8

import os
import random
import pandas as pd
import torch
import json
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_truths_args, read_truths,read_truths_count,read_truths_count_flir
from image import *
import pathlib as pl

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5,
                 transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4,condition=False):
       self.annotations = None
       self.read_dataset(root)
       if shuffle:
           random.shuffle(self.lines)

       # print('Go to list data')
       self.nSamples  = len(self.lines)
       self.transform = transform
       self.target_transform = target_transform
       self.train = train
       self.shape = shape
       self.seen = seen
       self.batch_size = batch_size
       self.num_workers = num_workers

       self.crop = crop
       self.jitter = jitter
       self.hue = hue
       self.saturation = saturation
       self.exposure = exposure

       self.condition = condition
       self.init_shape = shape
       self.cur_shape = shape

    def __len__(self):
        return self.nSamples

    def get_image(self,index):
        assert index <= len(self), 'index range error'
        return self.lines[index]

    def read_dataset(self,path):
        plpath = pl.Path(path)
        with plpath.open('r') as file:
            dirpath = plpath.parent
            if(plpath.suffix == '.json'):
                data = json.load(file)
                imgdata = pd.DataFrame(data['images'])[['id','file_name','width','height']]
                anndata = data['annotations']
                imgdata['annotations'] = imgdata['id'].apply(lambda x:[ann for ann in anndata if ann['image_id']==x])
                self.annotations = imgdata
                self.lines = [pl.Path.joinpath(dirpath,fn) for fn in imgdata['file_name']]
            else:
                self.lines = file.readlines()

    def get_different_scale(self):
        if self.seen < 50*self.batch_size:
            wh = 13*32                          # 416
        elif self.seen < 2000*self.batch_size:
            wh = (random.randint(0,3) + 13)*32  # 416, 480
        elif self.seen < 8000*self.batch_size:
            wh = (random.randint(0,5) + 12)*32  # 384, ..., 544
        elif self.seen < 10000*self.batch_size:
            wh = (random.randint(0,7) + 11)*32  # 352, ..., 576
        else: # self.seen < 20000*self.batch_size:
            wh = (random.randint(0,9) + 10)*32  # 320, ..., 608
        # print('new width and height: %d x %d ' % (wh, wh))
        return (wh, wh)

    def get_different_scale_my(self):
        init_width, init_height = self.init_shape
        init_width = init_width//32
        init_height = init_height // 32

        if self.seen < 100*self.batch_size:
            return self.init_shape
        elif self.seen < 8000*self.batch_size:
            rand_scale = random.randint(0,3)
            init_width = init_width-2 + rand_scale
            init_height = init_height-2 + rand_scale
        elif self.seen < 12000*self.batch_size:
            rand_scale = random.randint(0, 5)
            init_width = init_width - 3 + rand_scale
            init_height = init_height - 3 + rand_scale
        elif self.seen < 16000*self.batch_size:
            rand_scale = random.randint(0, 7)
            init_width = init_width - 4 + rand_scale
            init_height = init_height - 4 + rand_scale
        else:
            rand_scale = random.randint(0, 9)
            init_width = init_width - 5 + rand_scale
            init_height = init_height - 5 + rand_scale
        wid = init_width * 32
        hei = init_height * 32
        return (wid, hei)

    def __getitem__(self, index):
        # print('get item')
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index]

        img_id = os.path.basename(imgpath).split('.')[0]

        if self.train:
            # print(index)
            if (self.seen % (self.batch_size*100)) == 0: # in paper, every 10 batches, but we did every 64 images
                self.shape = self.get_different_scale_my()
                # self.shape = self.get_different_scale()
                # print('Image size: ', self.shape)
                # self.shape = self.get_different_scale()
            img, label = load_data_detection(imgpath, self.shape, self.crop, self.jitter, self.hue, self.saturation, self.exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath.rstrip()).convert('RGB')
            if self.shape:
                img, org_w, org_h = letterbox_image(img, self.shape[0], self.shape[1]), img.width, img.height
    
            # labpath = imgpath.replace('images', 'labels').replace('images', 'Annotations').replace('.jpg', '.txt').replace('.png','.txt')
            labpath = imgpath.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png','.txt').replace('.tif','.txt')
            try:
                tmp = torch.from_numpy(read_truths_args(labpath, 8.0 / img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1, 5)
            # tmp = torch.from_numpy(read_truths(labpath))
            label = torch.zeros(50*5)
            #if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        if self.train:
            if self.condition:
                #### this is for label daytime or nighttime on KAIST dataset
                set_label = 0 if int(img_id[4]) < 3 else 1
                return (img, (label, set_label))
            else:
                # print('end function get item')
                return (img, label)
        else:
            return (img, label, org_w, org_h)


class densityDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5,
                 transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4,
                 condition=False):
        self.read_dataset(root)
        if shuffle:
            random.shuffle(self.lines)

        # print('Go to list data')
        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.crop = crop
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

        self.condition = condition
        self.init_shape = shape
        self.cur_shape = shape

    def __len__(self):
        return self.nSamples

    def get_image(self, index):
        assert index <= len(self), 'index range error'
        return self.lines[index]

    def read_dataset(self, path):
        plpath = pl.Path(path)
        with plpath.open('r') as file:
            dirpath = plpath.parent
            if (plpath.suffix == '.json'):
                data = json.load(file)
                imgdata = pd.DataFrame(data['images'])[['id', 'file_name', 'width', 'height']]
                anndata = data['annotations']
                imgdata['annotations'] = imgdata['id'].apply(lambda x: [ann for ann in anndata if ann['image_id'] == x])
                self.annotations = imgdata
                self.lines = [pl.Path.joinpath(dirpath, fn) for fn in imgdata['file_name']]
            else:
                self.lines = file.readlines()

    def __getitem__(self, index):
        # print('get item')
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        img_id = os.path.basename(imgpath).split('.')[0]

        if self.train:
            # print(index)
            if (self.seen % (self.batch_size * 100)) == 0:  # in paper, every 10 batches, but we did every 64 images
                self.shape = self.get_different_scale_my()
                # self.shape = self.get_different_scale()
                # print('Image size: ', self.shape)
                # self.shape = self.get_different_scale()
            img, label = load_data_detection(imgpath, self.shape, self.crop, self.jitter, self.hue, self.saturation,
                                             self.exposure)
            label = torch.from_numpy(label)
        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img, org_w, org_h = letterbox_image(img, self.shape[0], self.shape[1]), img.width, img.height

            # labpath = imgpath.replace('images', 'labels').replace('images', 'Annotations').replace('.jpg', '.txt').replace('.png','.txt')
            labpath = imgpath.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace(
                '.png', '.txt').replace('.tif', '.txt')
            try:
                tmp = read_truths_count_flir(labpath, 8.0 / img.width)
            except Exception as e:
                tmp = np.array([0,0,0])
                # tmp = torch.from_numpy(read_truths(labpath))
            label = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        if self.train:
            if self.condition:
                #### this is for label daytime or nighttime on KAIST dataset
                set_label = 0 if int(img_id[4]) < 3 else 1
                return (img, (label, set_label))
            else:
                # print('end function get item')
                return (img, label)
        else:
            return (img, label, org_w, org_h)

class featureDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, crop=False, jitter=0.3, hue=0.1, saturation=1.5, exposure=1.5,
                target_transform=None, train=False, seen=0, batch_size=64, num_workers=4,
                 condition=False):
        with pl.Path(root).open('r') as file:
            self.lines = file.readlines()
        if shuffle:
            random.shuffle(self.lines)

        # print('Go to list data')
        self.nSamples = len(self.lines)
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.crop = crop
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

        self.condition = condition
        self.init_shape = shape
        self.cur_shape = shape

    def __len__(self):
        return self.nSamples

    def get_image(self, index):
        assert index <= len(self), 'index range error'
        return self.lines[index]

    def __getitem__(self, index):
        # print('get item')
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()
        img_id = os.path.basename(imgpath).split('.')[0]

        img = torch.from_numpy(np.load(pl.Path(imgpath))).squeeze(0)

        # labpath = imgpath.replace('images', 'labels').replace('images', 'Annotations').replace('.jpg', '.txt').replace('.png','.txt')
        labpath = imgpath.replace('images', 'labels').replace('.npy', '.txt')
        try:
            with pl.Path(labpath).open('r') as l:
                tmp = float(l.readlines()[0])
        except Exception:
            tmp = 0
            # tmp = torch.from_numpy(read_truths(labpath))
        label = tmp

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers

        if self.train:
            if self.condition:
                #### this is for label daytime or nighttime on KAIST dataset
                set_label = 0 if int(img_id[4]) < 3 else 1
                return (img, (label, set_label))
            else:
                # print('end function get item')
                return (img, label)
        else:
            return (img, label)

def json_to_txt(json_path):
    jpath = pl.Path(json_path)
    with jpath.open('r') as jfile:
        data = json.load(jfile)
        anndata = pd.DataFrame(data['annotations'])
        anndata[['bb_x','bb_y','bb_w','bb_h']] = pd.DataFrame(anndata.bbox.tolist())
        anndata['bb_x']/=640
        anndata['bb_y']/=512
        anndata['bb_w']/=640
        anndata['bb_h']/=512
        for img in data['images']:
            id = img['id']
            tpath = jpath.parent.joinpath(img['file_name'])
            tpath = pl.Path.joinpath(tpath.parent,tpath.stem+'.txt')
            lines = anndata[anndata.image_id == id]
            if len(lines) > 0:
                str = lines.to_string(header=False,index=False,columns=['category_id','bb_x','bb_y','bb_w','bb_h'])
                with tpath.open('w') as text:
                    text.write(str)
                print(img)
if __name__ == '__main__':
    json_to_txt('D:/dataset/flir_dataset/val/thermal_annotations.json')