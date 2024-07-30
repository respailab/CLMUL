# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys
from utils.mul.model import ResNet18
from backbone.ResNet18 import*
from tqdm.notebook import tqdm


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model, dataset,last=False,eval_tea=False):
    current_model = model.net
    if eval_tea:
        current_model = model.teacher_model
    status = current_model.training
    current_model.eval()

    num_classes = 10  # Replace with the actual number of classes
    class_accuracies = [0] * num_classes
    class_totals = [0] * num_classes

    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = current_model(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            # Calculate accuracy for each class
            for i in range(num_classes):
                class_indices = (labels == i).nonzero(as_tuple=True)[0]
                if len(class_indices) > 0:
                    class_accuracies[i] += (pred[class_indices] == labels[class_indices]).float().sum().item()
                    class_totals[i] += len(class_indices)

            #Task incremntal learnning results
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    # Calculate final class accuracies
    for i in range(num_classes):
        if class_totals[i] > 0:
            class_accuracies[i] /= class_totals[i]

    current_model.train(status)

    return accs, accs_mask_classes, class_accuracies



def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    model_stash = create_stash(model, args, dataset)

    tea_loggers = {}
    tea_results = {}
    tea_results_mask_classes = {}

    unlearning_teacher=ResNet18(num_classes = 10, pretrained = False).to(model.device).eval() 

    if hasattr(model, 'teacher_model'):
        tea_results['teacher_model'], tea_results_mask_classes['teacher_model'] = [], []

    if args.csv_log:
        #csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
        if hasattr(model, 'teacher_model'):
            #print(f'Creating Logger for the teacher model')
            tea_loggers['teacher_model'] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    #random_results_class, random_results_task = evaluate(model, dataset_copy)
    print(file=sys.stderr)

    # learn initial 4 tasks. 
    for t in range(0, 4):
        model.net.train()
        dataset.i = t*dataset.N_CLASSES_PER_TASK
        train_loader, test_loader = dataset.get_data_loaders()
        # if t:
        #     accs = evaluate(model, dataset, last=True)
        #     results[t-1] = results[t-1] + accs[0]
        #     if dataset.SETTING == 'class-il':
        #         results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
     
        for epoch in range(args.n_epochs):
            print('Epoch:',epoch)
            for i, data in enumerate(tqdm(train_loader)):
                #print(i)
                inputs, labels, not_aug_inputs,clabel = data
                #print('Class label:',clabel)
                #print('Label:',labels)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                # Learning step
                #print(inputs.shape)
                cl_loss=model.observe(inputs, labels, not_aug_inputs, ulabel=torch.zeros(inputs.shape[0]), unlearning_teacher=unlearning_teacher, full_trained_teacher=model, task_label=torch.tensor(t).repeat(inputs.shape[0]))
                loss=cl_loss
                progress_bar(i, len(train_loader), epoch, t, loss)
                     

                '''# Unlearning step
                
                unlearning_indices = (ulabel == 1).nonzero(as_tuple=True)[0]
                if len(unlearning_indices) > 0:
                     ul_loss=model.observe(inputs[unlearning_indices], labels[unlearning_indices], not_aug_inputs[unlearning_indices], ulabel=1,unlearning_teacher=unlearning_teacher, full_trained_teacher=model)
                     loss=ul_loss
                     progress_bar(i, len(train_loader), epoch, t, loss)'''
            print('Epoch ended')
        
        #Evaluate class accuracies after each epoch
    print('Continual Learning Process ended:')
    accs, _, class_accuracies = evaluate(model, dataset)
    for i, class_acc in enumerate(class_accuracies):
        print('Class', i, 'Accuracy:', class_acc)


        # Unlearning second task (Task: 1)

        # remove task 1 from buffer

    #model.buffer.remove_task(1)
    #model.buffer.remove_task(3)
    #model.buffer.remove_task(4)
    
    for t in range(1, 5):
        model.net.train()
        dataset.i = t*dataset.N_CLASSES_PER_TASK
        train_loader, test_loader = dataset.get_data_loaders()
        # if t:
        #     accs = evaluate(model, dataset, last=True)
        #     results[t-1] = results[t-1] + accs[0]
        #     if dataset.SETTING == 'class-il':
        #         results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        if t == 1:
               model.buffer.remove_task(1)
        elif t == 3:
               model.buffer.remove_task(3)
        #elif t == 4:
               #model.buffer.remove_task(4)
     
        for epoch in range(args.n_epochs):
            print('Epoch:',epoch)
            for i, data in enumerate(tqdm(train_loader)):
                #print(i)
                inputs, labels, not_aug_inputs,clabel = data
                #print('Class label:',clabel)
                #print('Label:',labels)
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                ulabels = []
                for i in range(len(labels)):
                    if labels[i] == 2 or labels[i] == 3 or labels[i] == 6 or labels[i] == 7:
                        ulabels.append(1)
                    else:
                        ulabels.append(0)
                ulabels = torch.tensor(ulabels)
                # Learning step
                cl_loss=model.observe(inputs, labels, not_aug_inputs, ulabel=ulabels, unlearning_teacher=unlearning_teacher, full_trained_teacher=model, task_label = torch.tensor(t).repeat(inputs.shape[0]))
                loss=cl_loss
                progress_bar(i, len(train_loader), epoch, t, loss)
                     

                '''# Unlearning step
                
                unlearning_indices = (ulabel == 1).nonzero(as_tuple=True)[0]
                if len(unlearning_indices) > 0:
                     ul_loss=model.observe(inputs[unlearning_indices], labels[unlearning_indices], not_aug_inputs[unlearning_indices], ulabel=1,unlearning_teacher=unlearning_teacher, full_trained_teacher=model)
                     loss=ul_loss
                     progress_bar(i, len(train_loader), epoch, t, loss)'''
            print('Epoch ended')
        
        #Evaluate class accuracies after each epoch
        print('Continual unLearning Process ended:')
        accs, _, class_accuracies = evaluate(model, dataset)
        for i, class_acc in enumerate(class_accuracies):
            print('Class', i, 'Accuracy:', class_acc)
    

