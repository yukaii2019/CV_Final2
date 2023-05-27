import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms.functional as TF
import random
import torchvision

def get_dataloader(dataset_dir, label_data, batch_size=1, split='test'):
    dataset = mydataset(dataset_dir, label_data, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train' or split=='val' or split=='train_val'), num_workers=12, pin_memory=True)
    return dataloader


class mydataset(Dataset):
    def __init__(self, dataset_dir, label_data,  split='test'):
        super(mydataset).__init__()
        
        self.dataset_dir = dataset_dir
        self.split = split

        self.image_names = []
        self.len = 0

        k = 24
        if split == 'train':
            self.seqs = [x for x in range(k)] 
            self.subjects = ['S1', 'S2', 'S3', 'S4']
        elif split == 'val':
            self.seqs = [x for x in range(k, 26)] 
            self.subjects = ['S1', 'S2', 'S3', 'S4']
        elif split == 'train_val':
            self.seqs = [x for x in range(26)]
            self.subjects = ['S1', 'S2', 'S3', 'S4']
        elif split == 'test':
            self.seqs = [x for x in range(26)]
            self.subjects = ['S5', 'S6', 'S7', 'S8']

        for subject in self.subjects :
            for seq in self.seqs:
                image_folder = os.path.join(subject, f'{seq + 1:02d}')
                try:
                    names = [os.path.splitext(os.path.join(image_folder, name))[0] for name in os.listdir(os.path.join(self.dataset_dir, image_folder)) if name.endswith('.jpg')]
                    self.image_names.extend(names)
                    self.len += len(names)
                except:
                    print(f'Labels are not available for {image_folder}')
        
        if split != 'test':
            with open(label_data) as file:
                self.label_data = json.load(file)

        print(f'Number of {self.split} images is {self.len}')
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.split != 'test':
            fn_img = os.path.join(self.dataset_dir , self.image_names[index]+".jpg")
            fn_msk = os.path.join(self.dataset_dir , self.image_names[index]+".png")

            img = Image.open(fn_img).convert('L')
            mask = Image.open(fn_msk).convert('L')
            
            mask = (np.array(mask) != 0).astype(int)
            data = np.array(list(self.label_data[self.image_names[index]+".jpg"].values())).astype(np.float32)

            img = transforms.functional.pil_to_tensor(img)
            mask = transforms.ToTensor()(mask)           
            pos = torch.tensor(data[:-1]).contiguous()
            conf = torch.tensor(data[-1]).contiguous()

            img = transforms.Resize((48, 64))(img)
            mask = transforms.Resize((48, 64))(mask)

            ###########failure augmentations###########
            # r = transforms.RandomRotation.get_params((-10, 10))
            # img = TF.rotate(img, r)
            # mask = TF.rotate(mask, r)
            # pos[4] += r
            # pos[0] = (pos[0]-0.5) * np.cos(r*np.pi/180.) - (0.5-pos[1])*np.sin(r*np.pi/180.) + 0.5
            # pos[1] = 0.5 - ((pos[0]-0.5) * np.sin(r*np.pi/180.) + (0.5-pos[1])*np.cos(r*np.pi/180.))

            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(192, 256))
            # img = TF.crop(img, i, j, h, w)
            # mask = TF.crop(mask, i, j, h, w)

            # if random.random() > 0.5:
            #     img = TF.hflip(img)
            #     mask = TF.hflip(mask)
            #     pos[0] = 1 - pos[0] 

            # if random.random() > 0.5:
            #     img = TF.vflip(img)
            #     mask = TF.vflip(mask)
            #     pos[1] = 1 - pos[1] 

            mask = torch.squeeze(mask, 0)
            img = (img / 255).to(torch.float32)
            mask = mask.to(int)
            pos = pos.to(torch.float32)
            conf = conf.to(int)

            return {
                'images': img,
                'mask': mask,
                'pos': pos,
                'conf': conf 
            }

        else:
            pass
            # img = Image.open(self.image_names[index])
            # return {
            #     'images': img, 
            # }


def imshow(inp, fn=None, mul255 = False):
    inp = inp.numpy().transpose((1, 2, 0))
    if mul255: inp = inp*255
    im = Image.fromarray((inp).astype(np.uint8))
    im.save(fn)

if __name__ == '__main__':
    train_loader = get_dataloader('/home/ykhsieh/CV/final/dataset', '/home/ykhsieh/CV/final/dataset/data2.json', batch_size=4, split='train')
    print(train_loader.dataset.len)
    data = iter(train_loader).next()


    print(data['images'].shape, data['mask'].shape, data['pos'].shape, data['conf'].shape)

    images = data['images']
    mask = data['mask'] 
    pred_pos = data['pos']
    conf = data['conf']

    images = images.cpu().numpy()
    images = (255*images).astype(np.uint8)

    mask = mask.cpu().numpy()
    mask = (255*mask).astype(np.uint8)
    pos = pred_pos.cpu().numpy()

    h, w = images.shape[2], images.shape[3]

    for i in range(len(images)):
        pos[i] = pos[i] * np.array([w, h, h/2, w/2, 1]) # x y h w
        center_coordinates = (int(np.clip(pos[i][0], 0, w)), int(np.clip(pos[i][1], 0, h)))
        axesLength = (int(np.clip(pos[i][3], 0, w/2)), int(np.clip(pos[i][2], 0, h/2)))
        angle = pos[i][4]
        images[i][0] = cv2.ellipse(images[i][0], center_coordinates, axesLength, angle, 0, 360, 255, 1)
        cv2.imwrite(f'./view_data/test{i}.jpg', images[i][0])
 
    imshow(torchvision.utils.make_grid(torch.tensor(images)), fn="./view_data/train_set.png", mul255 = False)
    imshow(torchvision.utils.make_grid(torch.tensor(mask).unsqueeze(1)), fn="./view_data/train_mask.png", mul255 = False)
