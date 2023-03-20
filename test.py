import numpy as np
from PIL import Image


# def calculate_average_color(image, x, y, width, height):
#     box = (x, y, x + width, y + height)
#     region = image.crop(box)
#     avg_color = region.convert('RGB').resize((1, 1)).getpixel((0, 0))
#     return avg_color
#
#
# img = Image.open("./images/image1.jpg")
# print(calculate_average_color(img, 0, 0, *img.size))
# img.close()

# image = Image.open("./images/image1.jpg")
# data = np.asarray(image)
# rr = np.sum(data[:, 0])
# gg = np.sum(data[:, 1])
# bb = np.sum(data[:, 2])
# w, h = image.size
# cnt = w * h
# print(rr, gg, bb, cnt)
# print(rr // cnt, gg // cnt, bb // cnt)


# a = np.array([20, 20, 13])
# b = np.array([50, 16, 13])
# print(type(abs(a - b)))
# diff = np.array([10, 10, 10]) < a
# print(all(diff))
while i != 3:
    print('ghj')