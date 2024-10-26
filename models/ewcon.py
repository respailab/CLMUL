# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.buffer import Buffer
from copy import deepcopy
from models.utils.continual_model import ContinualModel
from utils.mul.unlearn import UnlearnerLoss,UnlearnerLossOnlyBadTeacher
from utils.args import *

class EwcOn(ContinualModel):
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='Continual learning via online EWC.')
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for EWC online')
        return parser

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.teacher_model = deepcopy(self.net).to(self.device)
        self.logsoft = nn.LogSoftmax(dim=1)
        self.ER_weight = args.ER_weight
        self.model_iterations=0
        self.checkpoint = None
        self.fish = None
        self.Bernoulli_probability = args.Bernoulli_probability

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.args.gamma
            self.fish += fish

        self.checkpoint = self.net.get_params().data.clone()

    ''''def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()'''

    
    def observe(self, inputs, labels, not_aug_inputs, ulabel, unlearning_teacher=None, full_trained_teacher=None, task_label = None):

        torch.autograd.set_detect_anomaly(True)
        outputs, _, bat_inv= self.net(inputs, return_features=True)
        self.opt.zero_grad()
        # Unlearning step
        loss = 0

        # TODO - Have no reduction for unlearning losses and then memrge the individual losses to get the batch loss at the end.
        if (ulabel ==1).sum() > 0:
            mask = (ulabel == 1)
            with torch.no_grad():
                #full_teacher_logits = full_trained_teacher(inputs[mask])
                unlearn_teacher_logits = unlearning_teacher(inputs[mask])
            # unlearning_loss = UnlearnerLoss(output=outputs[mask], labels=labels[mask], full_teacher_logits=full_teacher_logits[mask], 
            #                                 unlearn_teacher_logits=unlearn_teacher_logits[mask], KL_temperature=1)
            unlearning_loss = UnlearnerLossOnlyBadTeacher(output = outputs[mask], 
                                            unlearn_teacher_logits = unlearn_teacher_logits, KL_temperature=1)
            loss += unlearning_loss
        retain_mask = (ulabel == 0)
        if retain_mask.sum() > 0:
            penalty=self.penalty()
            learning_loss = self.loss(outputs[retain_mask], labels[retain_mask].long()) + self.args.e_lambda * penalty
            loss += learning_loss
        if not self.buffer.is_empty():
            #print("In buffer")
            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            teacher_logits, _, tea_inv= self.teacher_model(buf_inputs, return_features=True)
            student_outputs, _, buf_inv = self.net(buf_inputs, return_features=True)
            with torch.no_grad():
                teacher_model_prob = F.softmax(teacher_logits, 1)
                label_mask = F.one_hot(buf_labels, num_classes=teacher_logits.shape[-1]) > 0
                adaptive_weight = teacher_model_prob[label_mask]
                # print(teacher_logits)
            squared_losses = adaptive_weight * torch.mean((student_outputs - teacher_logits.detach()) ** 2 , dim=1)

            loss += 0.1 *  squared_losses.mean()
            loss += self.ER_weight * self.loss(student_outputs, buf_labels)

            inputs = torch.cat((inputs[retain_mask], buf_inputs))
            labels = torch.cat((labels[retain_mask], buf_labels))
            bat_inv = torch.cat((bat_inv[retain_mask], buf_inv))

        # print("This worked")
        loss.backward(retain_graph=True)
        # print("This also worked")
        self.opt.step()
        self.model_iterations += 1
        if retain_mask.sum() > 0:
            self.buffer.add_data(examples=not_aug_inputs[retain_mask], labels=labels[:not_aug_inputs[retain_mask].shape[0]], logits = None, 
                                task_labels = task_label[retain_mask])

        #Updating the teacher model
        if torch.rand(1) < self.Bernoulli_probability:
            # Momentum coefficient m
            m = min(1 - 1 / (self.model_iterations + 1), 0.999)
            for teacher_param, param in zip(self.teacher_model.parameters(), self.net.parameters()):
                teacher_param.data.mul_(m).add_(alpha=1 - m, other=param.data)
        return loss.item()
       
    
    
    
