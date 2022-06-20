import os

import utils
import networks
from grad_cam import GradCAM
from plot_utils import plot_confmat, plot_gradcam

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # base_model = torch.load('model_train/resnet50', map_location=torch.device('cpu'))
# train_set = datasets.ImageFolder(root=utils.dirs['train'], transform=utils.transform['eval'])
# test_set = datasets.ImageFolder(root=utils.dirs['test'], transform=utils.transform['eval'])
# train_dl = DataLoader(train_set, batch_size=128)
# test_dl = DataLoader(test_set, batch_size=120)
#
# resnet18 = torch.load('model_train/resnet50', map_location=torch.device('cpu'))
# train_preds = utils.get_all_preds(resnet18, train_dl)
# test_preds = utils.get_all_preds(resnet18, test_dl)
#
# # train_preds.shape, test_preds.shape
# train_correct = utils.get_num_correct(train_preds, torch.as_tensor(train_set.targets, device=device))
# test_correct = utils.get_num_correct(test_preds, torch.as_tensor(test_set.targets, device=device))
#
# print(f'Train Correct: {train_correct:5}\tTrain Accuracy: {(100*train_correct/len(train_set)):5.2f}%')
# print(f'Test Correct: {test_correct:6}\tTest Accuracy: {(100*test_correct/len(test_set)):6.2f}%')
#
# train_confmat = utils.get_confmat(train_set.targets, train_preds)
# test_confmat = utils.get_confmat(test_set.targets, test_preds)
# plot_confmat(train_confmat, test_confmat, train_set.classes, f'{type(resnet18).__name__.lower()}')
#
# print(train_set.classes)
img_test = os.listdir('data/test/crack_new')

