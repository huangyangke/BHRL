#from mmdet.datasets.pipelines import AutoAugment,Rotate,Translate
import cv2
import numpy as np
import random
import mmcv
# _MAX_LEVEL = 10
# def level_to_value(level, max_value):
#     """Map from level to values based on max_value."""
#     return (level / _MAX_LEVEL) * max_value
# def random_negative(value, random_negative_prob):
#     """Randomly negate value based on random_negative_prob."""
#     return -value if np.random.rand() < random_negative_prob else value
# offset = int(level_to_value(5, 250))
# translate = Translate(level=5)

# rotate = Rotate(level=5)
# result = {}
# result['img'] = cv2.imread('/home/mgtv/pictures/lq4749.png')
# #rotate._rotate_img(result, angle=15)
# offset = random_negative(offset, 0.5)
# translate._translate_img(result, offset, 'vertical')#vertical horizontal
image = cv2.imread('/home/mgtv/pictures/lq4749.png')
image = cv2.resize(image, [224,224])
image = mmcv.impad(image, shape=(250,250))
# print('>>', image.shape[:2])
# image = cv2.rotate(image, 3)
# print('>>', image.shape[:2])
cv2.imwrite('/home/mgtv/enhance.jpg', image)




