import argparse
import os
from os.path import join
from tqdm import tqdm
import numpy as np
import time
import random
import torch
import json
import pickle
from torch import nn

from GPG.util import utils
from GPG.data.data_helper import DataHelper
from GPG.optims import Optim
import GPG.lr_scheduler as L
from GPG.data.data_util import *
# from GPG.data.vocab import Vocab
from GPG.models.model import GraphPointerGenerator
from GPG.models.model_utils import compute_loss, compute_loss_2
from GPG.data.feature import InputFeaturesQG, Example


class Trainer(object):
    def __init__(self, args):
        self.args = args
        torch.cuda.manual_seed(args.seed)
        self.logging, self.logging_csv, self.log_path = self.set_up_logging()
        if self.args.restore:  # 存储已有模型的路径
            print('loading checkpoint...\n')
            checkpoints = torch.load(self.args.restore)

        start_time = time.time()
        self.dataloader = DataHelper(gz=True, config=self.args)
        self.args.n_type = self.dataloader.n_type
        torch.backends.cudnn.benchmark = True
        print('loading data...\n')
        print('loading time cost: %.3f' % (time.time() - start_time))

        with open(self.args.embedding, "rb") as f:
            embedding = pickle.load(f)
            embedding = torch.Tensor(embedding)

        self.model = GraphPointerGenerator(config=self.args, embeddings = embedding)

        self.gpus = [int(i) for i in self.args.gpus.split(',')]
        self.model = self.model.cuda()
        self.print_model_info()

        if self.args.restore:
            self.model.load_state_dict(checkpoints['model'], strict=False)
            self.updates = checkpoints['updates']
            self.optim = checkpoints['optim']
        else:
            self.updates = 0
            self.optim = Optim(self.args.optim, self.args.learning_rate, self.args.max_grad_norm,
                        lr_decay=self.args.learning_rate_decay, start_decay_at=self.args.start_decay_at)
            # self.optim.set_parameters(self.model.parameters())
            self.optim.set_parameters(filter(lambda p: p.requires_grad, self.model.parameters()))
        
        self.scheduler = L.CosineAnnealingLR(self.optim.optimizer, T_max=self.args.epoch) if self.args.schedule else None
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpus)


    def save_model(self, path):
        model_state_dict = self.model.module.state_dict() if len(self.gpus) > 1 else self.model.state_dict()
        checkpoints = {
            'model': model_state_dict,
            'config': self.args,
            'optim': self.optim,
            'updates': self.updates}
        torch.save(checkpoints, path)
    
    def set_up_logging(self):
        log_dir = self.args.log + "GraphPointerGenerator"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_path = log_dir + '/' + utils.format_time(time.localtime()) + '/'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logging = utils.logging(log_path + 'log.txt') 
        logging_csv = utils.logging_csv(log_path + 'record.csv')
        return logging, logging_csv, log_path

    def save_settings(self):
        os.makedirs(self.log_path, exist_ok=True)
        json.dump(self.args.__dict__, open(join(self.log_path, "run_settings.json"), 'w'))

    def eval_loss(self, epoch):
        self.model.eval()
        eval_dataloader = self.dataloader.dev_loader
        total_loss = 0.
        i = 0
        start_time = time.time()
        for batch in tqdm(eval_dataloader):
            # if i > 2:
            #     break
            with torch.no_grad():
                eos_trg = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
                eos_trg = eos_trg[:, 1:].cuda()
                logits, softmasks = self.model(batch)
                batch_size, nsteps, _ = logits.size()
                preds = logits.view(batch_size * nsteps, -1)
                targets = eos_trg.contiguous().view(-1)
                targets = targets.cuda()
                loss = self.criterion(preds, targets)
                # outputs, attns, coverages, softmasks = self.model(batch)
                # target = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
                # bfs_mask = batch['bfs_mask']  
                # if self.args.use_cuda:
                #     target = target.cuda() 
                #     bfs_mask = bfs_mask.cuda()
                # loss, acc = compute_loss(outputs.transpose(0, 1), target.transpose(0, 1)[1:], attns, coverages, bfs_mask, softmasks,self.args.num_reason_layers, self.args.is_coverage, self.args.bfs_clf, self.args.bfs_lambda)
                total_loss += loss.data.item()
            i = i + 1
        val_loss = total_loss / i
        self.logging("time: %6.3f, epoch: %3d, val loss: %6.3f\n"
                    % (time.time() - start_time, epoch, val_loss
                    ))
        print("epoch: {}, val loss: {}\n".format(epoch, val_loss))
        return val_loss

    def train(self):
        best_loss = 1e10
        for epoch in range(1, self.args.epoch + 1):
            total_loss = 0.
            updates = 0
            start_time = time.time()
            train_dataloader = self.dataloader.train_loader
            if self.args.schedule:
                self.scheduler.step()
                print("Decaying learning rate to %g" % self.scheduler.get_lr()[0])
            
            self.model.train()
            for batch in tqdm(train_dataloader):
                self.model.zero_grad()
                if True:
                    eos_trg = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']) 
                    eos_trg = eos_trg[:, 1:]
                    eos_trg = eos_trg.cuda()
                logits, softmasks = self.model(batch)
                batch_size, nsteps, _ = logits.size()
                preds = logits.view(batch_size * nsteps, -1)
                targets = eos_trg.contiguous().view(-1)
                targets = targets.cuda()
                loss = self.criterion(preds, targets)
                # outputs, attns, coverages, softmasks = self.model(batch)
                # target = batch['tgt_idxs_extend'] * batch['tgt_mask'].type_as(batch['tgt_idxs_extend']).cuda()
                # bfs_mask = batch['bfs_mask'].cuda()   
                # loss, acc = compute_loss_2(logits, eos_trg, bfs_mask, softmasks, self.args.num_reason_layers, self.args.bfs_clf, self.args.bfs_lambda)
                loss.backward()
                total_loss += loss.data.item()
                self.optim.step()
                updates += 1 

            self.logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f\n"
                    % (time.time() - start_time, epoch, updates, total_loss / updates))
            print('evaluating after %d updates...\r' % updates)
            val_loss = self.eval_loss(epoch)
            # if val_loss <= best_loss:
            #     best_loss = val_loss
            self.save_model(self.log_path + str(val_loss) + '_checkpoint.pt')
            self.save_settings()

            print("Epoch {}  - final loss : {:.4f} - val loss :{:.4f}"
                  .format(epoch, total_loss / updates, val_loss))
            self.updates = self.updates + updates
            self.model.train()

    def print_model_info(self):
        self.logging(repr(self.model) + "\n\n")  
        param_count = 0
        for param in self.model.parameters():
            param_count += param.view(-1).size()[0]
        self.logging('total number of parameters: %d\n\n' % param_count)

        param_count = 0
        for param in self.model.parameters():
            if param.requires_grad:
                param_count += param.view(-1).size()[0]
        self.logging('total number of parameters need to be optimized: %d\n\n' % param_count)

        print("===========model structure ======")
        print(self.model)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

            # print("before back prop")
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            #         print(name, param.grad)
            # print("after back prop")
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            #         print(name, param.grad)
