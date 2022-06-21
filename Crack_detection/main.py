#resnet50
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from resnet.predict_on_crops import predict_on_crops
import argparse
from predict_new import predict_new
plt.figure(figsize=(10,10))

img_dir = 'test_img/1-4.jpg'
image_name = img_dir.split('/')[-1]
# img_shape = cv2.imread()

hight = 64
weight = hight
output_image, matrix = predict_on_crops(img_dir, hight, weight)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite(f'output_predict/resnet50_64/predict/{image_name}.png', output_image)

matrix = np.array(matrix)
matrix = matrix.reshape((math.ceil(output_image.shape[0]/hight), math.ceil(output_image.shape[1]/weight)))
print(matrix)
cv2.imwrite(f'matrix/{image_name}.png',matrix)


ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4, help="connectivity for connected component analysis")
args = vars(ap.parse_args())

thresh = cv2.imread(f'matrix/{image_name}.png',0)
print(thresh)
#apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(thresh, args["connectivity"], cv2.CV_32S)
# output = cv2.connectedComponentsWithStats(img)
(numLabels, labels, stats, centroids) = output

print(stats)

s = stats[:, 4]
if hight > 100:
    p = 4
else:
    p = 7

index = np.where(s < p)
print(index[0])

for i in index[0]:
    thresh[stats[i,1]:stats[i,1]+stats[i,3], stats[i,0]:stats[i,0]+stats[i,2]] = 0
print(thresh)


out_img = predict_new(img_dir, thresh, hight, weight)
plt.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
plt.show()
cv2.imwrite(f'output_predict/resnet50_64/predict_cc/{image_name}.png',out_img)