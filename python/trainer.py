import os
import csv
import random
import glob

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tqdm


from module import *
from data import get_dataloader
from logger import logger
from utils import set_seed 

class Trainer:
    def __init__(self, args):
        '''
            args:
            ## input file
                dataset_dir="/home/ykhsieh/CV/final/dataset/open_eye"
                label_data="/home/ykhsieh/CV/final/dataset/data.json"

            ## output file
                val_imgs_dir="./log-${time}/val_imgs"
                learning_curv_dir="./log-${time}/val_imgs"
                check_point_root="./log-${time}/checkpoints"
                log_root="./log-${time}"

            ## others
                batch_size=2
                lr=0.0001
                num_epochs=150
                milestones = [50, 100, 150]
        '''

        set_seed(9527)

        self.args = args
        self.device = torch.device('cuda')

        self.model = DenseNet2D().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.train_loader = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='train')
        self.val_loader   = get_dataloader(args.dataset_dir, args.label_data, batch_size=args.batch_size, split='val')        

        self.criterion1 = nn.MSELoss() 
        self.criterion2 = nn.CrossEntropyLoss()

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=0.1)
        
        self.train_loss_list = []
        self.val_loss_list = []

    def plot_learning_curve(self, result_list, name='train'):
        plt.plot(range(len(result_list)), result_list, label=f'{name}_loss')
        plt.title(f'{name} loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.savefig(os.path.join(self.args.learning_curv_dir , f'{name}_loss.png'))
        plt.close()


    # def save_checkpoint(self):
    #     state = {'model': self.model.state_dict()}
        
    #     for pth in glob.glob(os.path.join(self.args.check_point_root, '*.pth')):
    #         os.remove(pth)

    #     checkpoint_path = (os.path.join(self.args.check_point_root, 'acc-%i.pth' % (self.acc*100000)))

    #     torch.save(state, checkpoint_path)
    #     logger.info('model saved to %s' % checkpoint_path)

    def visualize(self, pred_mask, pred_pos, images, name = 'train'):
        pred_mask = pred_mask.max(1, keepdim=False)[1].cpu().numpy()[0]
        pred_mask = (255*pred_mask).astype(np.uint8)
        cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_mask{self.epoch}.jpg'), pred_mask)

        image = images.cpu().detach().numpy()[0][0]
        image = (255*image).astype(np.uint8)

        h, w = image.shape[0], image.shape[1]

        pred_pos = pred_pos.cpu().detach().numpy()[0]
        pred_pos = pred_pos * np.array([w, h, h/2, w/2, 1]) # x y h w
        center_coordinates = (int(np.clip(pred_pos[0], 0, w)), int(np.clip(pred_pos[1], 0, h)))
        axesLength = (int(np.clip(pred_pos[3], 0, w/2)), int(np.clip(pred_pos[2], 0, h/2)))
        angle = np.arcsin(pred_pos[4]) * (180 / np.pi)

        image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, 255, 1)
        cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_ellp{self.epoch}.jpg'), image)

    def train_epoch(self):

        train_loss = 0.0
        train_loss_1 = 0.0
        train_loss_2 = 0.0
        
        self.model.train()
        for batch, data in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80, leave=False):

            images, mask, pos = data['images'].to(self.device), data['mask'].to(self.device), data['pos'].to(self.device)
            
            pred_mask, pred_pos = self.model(images)

            loss_1 = self.criterion1(pred_pos, pos)
            loss_2 = self.criterion2(pred_mask, mask) 
            loss = 0.2*loss_1 + 0.8*loss_2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_loss_1 += loss_1.item() 
            train_loss_2 += loss_2.item() 

        train_loss   /= len(self.train_loader)
        train_loss_1 /= len(self.train_loader)
        train_loss_2 /= len(self.train_loader)
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Train Loss: {train_loss:.5f} | Train Loss1: {train_loss_1:.5f} | Train Loss2: {train_loss_2:.5f}')

        self.train_loss_list.append(train_loss)
    
        self.visualize(pred_mask, pred_pos, images, name='train')

        

    def val_epoch(self):
        self.model.eval()
        
        val_loss = 0.0
        val_loss_1 = 0.0
        val_loss_2 = 0.0
        
        for batch, data in tqdm.tqdm(enumerate(self.val_loader), total = len(self.val_loader), ncols=80, leave=False):

            with torch.no_grad():
                images, mask, pos = data['images'].to(self.device), data['mask'].to(self.device), data['pos'].to(self.device)
            

            pred_mask, pred_pos = self.model(images)

            loss_1 = self.criterion1(pred_pos, pos)
            loss_2 = self.criterion2(pred_mask, mask) 
            loss = 0.2*loss_1 + 0.8*loss_2

            val_loss += loss.item()
            val_loss_1 += loss_1.item()
            val_loss_2 += loss_2.item()

        val_loss /=   len(self.val_loader)
        val_loss_1 /= len(self.val_loader)
        val_loss_2 /= len(self.val_loader)
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Val Loss: {val_loss:.5f} | Val Loss1: {val_loss_1:.5f} | Val Loss2: {val_loss_2:.5f}')
        
        self.val_loss_list.append(val_loss)
        
        self.visualize(pred_mask, pred_pos, images, name='val')

        

    def train(self):
        self.epoch = 0
        self.best_loss = None

        for self.epoch in range(self.args.num_epochs):

            self.train_epoch()
            self.plot_learning_curve(self.train_loss_list, name='train')

            self.val_epoch()
            self.plot_learning_curve(self.val_loss_list, name='val')

            self.scheduler.step()

            if self.best_loss == None or self.val_loss_list[-1] <= self.best_loss:
                self.best_loss = self.val_loss_list[-1]
                logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Save best model to {self.args.check_point_root} ...')
                torch.save(self.model.state_dict(), os.path.join(self.args.check_point_root, 'model_best.pth'))
                self.best_loss = self.val_loss_list[-1]



    
            


    

