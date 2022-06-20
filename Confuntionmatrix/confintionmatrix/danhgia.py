import os
import cv2 as cv
import torch
from PIL import Image
from matplotlib import pyplot as plt
from predict import predict
import numpy as np
matrix_test = np.concatenate([np.ones((1,2000)), np.zeros((1,2000))], axis=0)
matrix_test = np.reshape(matrix_test, (1,4000))
matrix_test = matrix_test[0]
print(matrix_test)

resnet50 = torch.load('model_train/model_resnet50_resize64', map_location=torch.device('cpu'))
img_test_crack = os.listdir('data/test/crack_new')
img_test_no_crack = os.listdir('data/test/no_crack_new')
dir_crack = 'data/test/crack_new/'
dir_no_crack = 'data/test/no_crack_new/'
class_name_1 = []
class_name_0 = []
# print(dir + img_test[0])
# img = cv.imread(dir+img_test[1])
# a = predict(resnet50,Image.fromarray(img))
# print(a)

for i in img_test_crack:
    img = cv.imread(dir_crack+i)
    cl = predict(resnet50, Image.fromarray(img))
    class_name_1.append(int(cl))
for i in img_test_no_crack:
    img = cv.imread(dir_no_crack+i)
    cl = predict(resnet50, Image.fromarray(img))
    class_name_0.append(int(cl))

class_name = class_name_1 + class_name_0
print(class_name)

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

labels = [1,0]
cm = confusion_matrix(matrix_test, class_name)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)

plt.savefig('outputs/results_resnet50_64.png')
plt.show()