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
from backbone.ResNet18 import *
from tqdm.notebook import tqdm
import random


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, inputs, labels):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((inputs.cpu(), labels.cpu()))

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return None, None
        sampled = random.sample(self.buffer, min(len(self.buffer), batch_size))
        inputs, labels = zip(*sampled)
        return torch.stack(inputs), torch.tensor(labels)


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model, dataset, last=False, eval_tea=False):
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

            for i in range(num_classes):
                class_indices = (labels == i).nonzero(as_tuple=True)[0]
                if len(class_indices) > 0:
                    class_accuracies[i] += (pred[class_indices] == labels[class_indices]).float().sum().item()
                    class_totals[i] += len(class_indices)

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    for i in range(num_classes):
        if class_totals[i] > 0:
            class_accuracies[i] /= class_totals[i]

    current_model.train(status)

    return accs, accs_mask_classes, class_accuracies


def train(model: ContinualModel, dataset: ContinualDataset, args: Namespace) -> None:
    model.net.to(model.device)
    results, results_mask_classes = [], []
    model_stash = create_stash(model, args, dataset)

    tea_loggers = {}
    tea_results = {}
    tea_results_mask_classes = {}

    unlearning_teacher = ResNet18(num_classes=10, pretrained=False).to(model.device).eval()

    if hasattr(model, 'teacher_model'):
        tea_results['teacher_model'], tea_results_mask_classes['teacher_model'] = [], []

    if args.csv_log:
        if hasattr(model, 'teacher_model'):
            tea_loggers['teacher_model'] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    print(file=sys.stderr)

    replay_buffer = ReplayBuffer(capacity=1000)  # Adjust the buffer size as needed

    for t in range(0, 4):
        model.net.train()
        dataset.i = t * dataset.N_CLASSES_PER_TASK
        train_loader, test_loader = dataset.get_data_loaders()

        for epoch in range(args.n_epochs):
            print('Epoch:', epoch)
            for i, data in enumerate(tqdm(train_loader)):
                inputs, labels, not_aug_inputs, clabel = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)

                replay_inputs, replay_labels = replay_buffer.sample(batch_size=len(inputs))
                if replay_inputs is not None:
                    replay_inputs, replay_labels = replay_inputs.to(model.device), replay_labels.to(model.device)
                    combined_inputs = torch.cat([inputs, replay_inputs])
                    combined_labels = torch.cat([labels, replay_labels])
                    combined_not_aug_inputs = torch.cat([not_aug_inputs, replay_inputs])
                else:
                    combined_inputs, combined_labels = inputs, labels
                    combined_not_aug_inputs = not_aug_inputs

                cl_loss = model.observe(combined_inputs, combined_labels, combined_not_aug_inputs, ulabel=torch.zeros(combined_inputs.shape[0]).to(model.device), unlearning_teacher=unlearning_teacher, full_trained_teacher=model, task_label=torch.tensor(t).repeat(combined_inputs.shape[0]).to(model.device))
                loss = cl_loss
                progress_bar(i, len(train_loader), epoch, t, loss)

                for input, label in zip(inputs, labels):
                    replay_buffer.add(input, label)

            print('Epoch ended')

    print('Continual Learning Process ended:')
    accs, _, class_accuracies = evaluate(model, dataset)
    for i, class_acc in enumerate(class_accuracies):
        print('Class', i, 'Accuracy:', class_acc)

    for t in range(1, 5):
        model.net.train()
        dataset.i = t * dataset.N_CLASSES_PER_TASK
        train_loader, test_loader = dataset.get_data_loaders()
        if t == 1:
            model.buffer.remove_task(1)
        elif t == 3:
            model.buffer.remove_task(3)

        for epoch in range(args.n_epochs):
            print('Epoch:', epoch)
            for i, data in enumerate(tqdm(train_loader)):
                inputs, labels, not_aug_inputs, clabel = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                ulabels = []
                for i in range(len(labels)):
                    if labels[i] == 2 or labels[i] == 3 or labels[i] == 6 or labels[i] == 7:
                        ulabels.append(1)
                    else:
                        ulabels.append(0)
                ulabels = torch.tensor(ulabels).to(model.device)

                replay_inputs, replay_labels = replay_buffer.sample(batch_size=len(inputs))
                if replay_inputs is not None:
                    replay_inputs, replay_labels = replay_inputs.to(model.device), replay_labels.to(model.device)
                    combined_inputs = torch.cat([inputs, replay_inputs])
                    combined_labels = torch.cat([labels, replay_labels])
                    combined_not_aug_inputs = torch.cat([not_aug_inputs, replay_inputs])
                    combined_ulabels = torch.cat([ulabels, torch.zeros(len(replay_inputs)).to(model.device)])
                else:
                    combined_inputs, combined_labels = inputs, labels
                    combined_not_aug_inputs = not_aug_inputs
                    combined_ulabels = ulabels

                cl_loss = model.observe(combined_inputs, combined_labels, combined_not_aug_inputs, ulabel=combined_ulabels, unlearning_teacher=unlearning_teacher, full_trained_teacher=model, task_label=torch.tensor(t).repeat(combined_inputs.shape[0]).to(model.device))
                loss = cl_loss
                progress_bar(i, len(train_loader), epoch, t, loss)

                for input, label in zip(inputs, labels):
                    replay_buffer.add(input, label)

            print('Epoch ended')

    print('Continual unLearning Process ended:')
    accs, _, class_accuracies = evaluate(model, dataset)
    for i, class_acc in enumerate(class_accuracies):
        print('Class', i, 'Accuracy:', class_acc)
        