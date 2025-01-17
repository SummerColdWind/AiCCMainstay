import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fa5]+')

def show_image(image):
    """ 展示一张图片 """
    image_show = image.copy().astype('uint8')
    if np.amax(image_show) == 1:
        image_show = image_show * 255
    cv2.imshow('Show', image_show)
    cv2.waitKey(0)


def show_image_plt(image):
    """ 展示一张图片，通过matplotlib """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

def load_image(path):
    """ 加载一张图片 """
    with open(path, 'rb') as file:
        file_data = file.read()
        image_array = np.frombuffer(file_data, np.uint8)
        image_raw = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image_raw


def save_image(image, path='result.png'):
    """ 保存一张图片 """
    extend = '.' + path.split('.')[-1]
    retval, buffer = cv2.imencode(extend, image.astype('uint8'))
    with open(path, 'wb') as f:
        f.write(buffer)



def extract_Chinese(string, join=False):
    """ 提取字符串的中文 """
    chinese_string = CHINESE_PATTERN.findall(string)
    if chinese_string:
        return ''.join(chinese_string) if join else chinese_string[0]
    return ''


