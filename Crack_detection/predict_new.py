
import cv2
import numpy as np

def predict_new(input_image, thresh, height=64, width=64):
    im = cv2.imread(input_image)
    imgheight, imgwidth, channels = im.shape
    k=0
    output_image = np.zeros_like(im)
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            a = im[i:i+height, j:j+width]
            if thresh[int(i/height)][int(j/width)] == 255:
                text ='P'
                color = (0,0, 255)

            else:
                text ='N'
                color = (0, 255, 0)

            cv2.putText(a, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX , 0.7, color, 1, cv2.LINE_AA)
            b = np.zeros_like(a, dtype=np.uint8)
            b[:] = color
            add_img = cv2.addWeighted(a, 0.9, b, 0.1, 0)
            output_image[i:i+height, j:j+width,:] = add_img
            k+=1
    return output_image