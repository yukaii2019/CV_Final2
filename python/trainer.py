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
from utils import *
from loss import get_segLoss, get_seg2ptLoss, normPts 

class Trainer:
    def __init__(self, args):
        '''
            args:
            ## input file
                dataset_dir="/home/ykhsieh/CV/final/dataset/"
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
        
        self.train_loss_list = {'total': [], 'seg2pt': [], 'seg': [], 'exist': []}
        self.val_loss_list = {'total': [], 'seg2pt': [], 'seg': [], 'exist': []}
        self.val_score_list = {'wiou': [], 'atnr': [], 'score': []}

    def plot_learning_curve(self, result_list, name='train'):
        for (type_, list_) in result_list.items():
            plt.plot(range(len(list_)), list_, label=f'{name}_{type_}_value')
            plt.title(f'{name} {type_}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend(loc='best')
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            plt.savefig(os.path.join(self.args.learning_curv_dir , f'{name}_{type_}.png'))
            plt.close()


    def save_checkpoint(self):
        for pth in glob.glob(os.path.join(self.args.check_point_root, '*.pth')):
            os.remove(pth)
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Save best model to {self.args.check_point_root} ...')
        torch.save(self.model.state_dict(), os.path.join(self.args.check_point_root, f'model_best_{int(self.best_score*1000000)}.pth')) 



    def visualize(self, pred_mask, pred_pos, images, name = 'train'):
        pred_mask = pred_mask.max(1, keepdim=False)[1].cpu().numpy()[0]
        pred_mask = (255*pred_mask).astype(np.uint8)
        cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_mask{self.epoch}.jpg'), pred_mask)

        '''
            since the regression module is removed, cannot draw
        '''
        # image = images.cpu().detach().numpy()[0][0]
        # image = (255*image).astype(np.uint8)

        # h, w = pred_mask.shape[0], pred_mask.shape[1]
        # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC) 

        # pred_pos = pred_pos.cpu().detach().numpy()[0]
        # pred_pos = pred_pos * np.array([w, h, h/2, w/2, 1]) # x y h w
        # center_coordinates = (int(np.clip(pred_pos[0], 0, w)), int(np.clip(pred_pos[1], 0, h)))
        # axesLength = (int(np.clip(pred_pos[3], 0, w/2)), int(np.clip(pred_pos[2], 0, h/2)))
        # angle = np.arcsin(pred_pos[4]) * (180 / np.pi)

        # image = cv2.ellipse(image, center_coordinates, axesLength, angle, 0, 360, 255, 1)
        # cv2.imwrite(os.path.join(self.args.val_imgs_dir, f'{name}_ellp{self.epoch}.jpg'), image)

    def train_epoch(self):

        total_loss = 0.0
        total_seg2pt_pup_loss = 0.0
        total_seg_loss = 0.0
        total_exist_loss = 0.0
        
        self.model.train()
        for batch, data in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=80, leave=False):

            images, mask, pos, conf = data['images'].to(self.device), data['mask'].to(self.device), data['pos'].to(self.device), data['conf'].to(self.device)
            
            pred_mask, pred_exist = self.model(images)

            l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(pred_mask[:, 1, ...], normPts(pos[:,0:2]), temperature=4)
            l_seg2pt_pup = torch.mean(l_seg2pt_pup)
            l_seg = get_segLoss(pred_mask, mask, conf, self.beta)
            l_exist = self.criterion2(pred_exist, conf)

            loss = l_seg2pt_pup + 20*l_seg + 5*l_exist
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_seg2pt_pup_loss += l_seg2pt_pup.item()
            total_seg_loss += l_seg.item()
            total_exist_loss += l_exist.item() 
            

        total_loss   /= len(self.train_loader)
        total_seg2pt_pup_loss /= len(self.train_loader)
        total_seg_loss /= len(self.train_loader)
        total_exist_loss /= len(self.train_loader)

        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Train Loss: {total_loss:.5f} | l_seg2pt_pup: {total_seg2pt_pup_loss:.5f} | l_seg: {total_seg_loss:.5f} | l_exist: {total_exist_loss:.5f}')

        self.train_loss_list['total'].append(total_loss)
        self.train_loss_list['seg2pt'].append(total_seg2pt_pup_loss)
        self.train_loss_list['seg'].append(total_seg_loss)
        self.train_loss_list['exist'].append(total_exist_loss)
    
        self.visualize(pred_mask, None, images, name='train')

    def benchmark(self, pred_mask, pred_conf, mask, conf):
        pred_mask = torch.clone(pred_mask.max(1, keepdim=False)[1]).cpu().detach().numpy()
        pred_conf = torch.clone(pred_conf.max(1, keepdim=False)[1]).cpu().detach().numpy() 
        mask = torch.clone(mask).cpu().detach().numpy() 
        conf = torch.clone(conf).cpu().detach().numpy() 

        for i in range(len(conf)):
            if conf[i] == 1:
                self.label_validity.append(1.0)
                iou = mask_iou(pred_mask[i], mask[i])
                self.iou_meter.update(pred_conf[i] * iou)
            else:
                self.label_validity.append(0.0)

        self.output_conf.extend(pred_conf)


    def val_epoch(self):
        self.model.eval()
        
        total_loss = 0.0
        total_seg2pt_pup_loss = 0.0
        total_seg_loss = 0.0
        total_exist_loss = 0.0

        self.output_conf = []
        self.label_validity = []
        self.iou_meter = AverageMeter()

        with torch.no_grad():

            for batch, data in tqdm.tqdm(enumerate(self.val_loader), total = len(self.val_loader), ncols=80, leave=False):


                images, mask, pos, conf = data['images'].to(self.device), data['mask'].to(self.device), data['pos'].to(self.device), data['conf'].to(self.device)

                pred_mask, pred_exist = self.model(images)

                l_seg2pt_pup, pred_c_seg_pup = get_seg2ptLoss(pred_mask[:, 1, ...], normPts(pos[:,0:2]), temperature=4)
                l_seg2pt_pup = torch.mean(l_seg2pt_pup)
                l_seg = get_segLoss(pred_mask, mask, conf, self.beta)
                l_exist = self.criterion2(pred_exist, conf)

                loss = l_seg2pt_pup + 20*l_seg + 5*l_exist


                total_loss += loss.item()
                total_seg2pt_pup_loss += l_seg2pt_pup.item()
                total_seg_loss += l_seg.item()
                total_exist_loss += l_exist.item() 

                self.benchmark(pred_mask, pred_exist , mask, conf)


        total_loss /=   len(self.val_loader)
        total_seg2pt_pup_loss /= len(self.val_loader)
        total_seg_loss /= len(self.val_loader)
        total_exist_loss /= len(self.val_loader)
        
        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Val  Loss: {total_loss:.5f} | l_seg2pt_pup: {total_seg2pt_pup_loss:.5f} | l_seg: {total_seg_loss:.5f} | l_exist: {total_exist_loss:.5f}')

        self.val_loss_list['total'].append(total_loss)
        self.val_loss_list['seg2pt'].append(total_seg2pt_pup_loss)
        self.val_loss_list['seg'].append(total_seg_loss)
        self.val_loss_list['exist'].append(total_exist_loss)

        tn_rates = true_negative_curve(np.array(self.output_conf), np.array(self.label_validity))
        wiou = self.iou_meter.avg()
        atnr = np.mean(tn_rates)
        self.score = 0.7 * wiou + 0.3 * atnr
        
        self.val_score_list['wiou'].append(wiou)
        self.val_score_list['atnr'].append(atnr)
        self.val_score_list['score'].append(self.score)


        logger.info(f'[{self.epoch + 1}/{self.args.num_epochs}] Val  wiou: {wiou:.4f} | atnr: {atnr:.4f} | score: {self.score:.4f}')

        self.visualize(pred_mask, None, images, name='val')

        

    def train(self):
        self.epoch = 0
        self.best_score = None

        for self.epoch in range(self.args.num_epochs):
            self.alpha = linVal(self.epoch, (0, self.args.num_epochs), (0, 1), 0)
            self.beta = 1-self.alpha

            self.train_epoch()
            self.plot_learning_curve(self.train_loss_list, name='train')

            self.val_epoch()
            self.plot_learning_curve(self.val_loss_list, name='val')
            self.plot_learning_curve(self.val_score_list, name='val')

            self.scheduler.step()

            if self.best_score == None or self.score > self.best_score:
                self.best_score = self.score 
                self.save_checkpoint()
                



    
            


    

