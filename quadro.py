import sys
from math import ceil
from pprint import pprint
from typing import List
import cProfile
import cv2
from threading import Thread

import numpy as np
from PIL import Image, ImageStat
import pygame as pg

DIFF = np.array([10, 10, 10])
MAX_DEPTH = 1
IS_BORDERS = True


class QuadTree:
    class QuadNode:
        def __init__(self, img: Image, x, y, width, height, color=None):
            self.x = x
            self.y = y

            self.w = width
            self.h = height

            self.image = img.crop((x, y, width + x, height + y))
            self.color = color
            self.children = []

        def is_leaf(self):
            return not self.children

        def split_to_four(self):
            child_w, child_h = ceil(self.w / 2), ceil(self.h / 2)
            self.children.append(
                QuadTree.QuadNode(self.image, self.x, self.y,
                                  child_w, child_h))
            self.children.append(
                QuadTree.QuadNode(self.image, self.x + child_w, self.y,
                                  child_w, child_h))
            self.children.append(
                QuadTree.QuadNode(self.image, self.x, self.y + child_h,
                                  child_w, child_h))
            self.children.append(
                QuadTree.QuadNode(self.image, self.x + child_w, self.y +
                                  child_h, child_w, child_h))

        def render(self, screen, is_border):  # сделать границы
            box_rect = pg.Rect(self.x, self.y, self.w, self.h)
            pg.draw.rect(screen, self.color, box_rect)
            if IS_BORDERS:
                pg.draw.rect(screen, (0, 0, 0), box_rect, 1)

        def get_all_nodes(self):
            all_nodes = [self]
            for child_node in self.children:
                all_nodes.extend(child_node.get_all_nodes())
            return all_nodes

        def set_color(self, color):
            self.color = color

        def __repr__(self):
            return f"Node({self.x}, {self.y}, {self.w}, {self.h}, " \
                   f"color={self.color}"

    def __init__(self, img):
        self.img = img
        self.root_node = QuadTree.QuadNode(img, 0, 0, *self.img.size)

    def build(self):
        def build_inner(cur_node, cur_color, depth=0):
            if depth >= MAX_DEPTH:
                cur_node.set_color(cur_color)
                return
            if cur_node.w == 1 or cur_node.h == 1:
                return

            resp, cols = should_divide(self.img, cur_color,
                                       cur_node.x, cur_node.y, cur_node.w,
                                       cur_node.h)
            if resp is False:
                cur_node.set_color(cur_color)
                return

            cur_node.split_to_four()
            for node, col in zip(cur_node.children, cols):
                build_inner(node, col, depth + 1)

        self.root_node.children.clear()
        build_inner(self.root_node,
                    get_average_color(self.img,
                                      self.root_node.x,
                                      self.root_node.y,
                                      self.root_node.w,
                                      self.root_node.h))

    def print_nodes(self):
        def _print_nodes(node: "QuadTree.QuadNode", depth=0):
            print("\t" * depth, node)
            for child in node.children:
                _print_nodes(child, depth + 1)

        _print_nodes(self.root_node)
        print("#" * 20)

    def render(self, screen, is_border):
        leaves = self.find_leaves()
        # pprint(leaves)
        for leaf in leaves:
            leaf.render(screen, is_border)

    def find_leaves(self) -> List["QuadTree.QuadNode"]:
        def find_leaves_inner(cur_node: "QuadTree.QuadNode"):
            if cur_node.is_leaf():
                return [cur_node]
            else:
                leaves = []
                for child in cur_node.children:
                    leaves.extend(find_leaves_inner(child))
                return leaves

        return find_leaves_inner(self.root_node)


def get_average_color(img, *box):
    region = img.crop(box)
    rg_array = np.array(region)
    rg_mean = cv2.mean(rg_array)[:-1]
    return np.array(rg_mean)


def should_divide(img, cur_color, x, y, width, height):
    avg_all = cur_color
    half_w, half_h = width / 2, height / 2
    avg_lt = get_average_color(img, x1 := x, y1 := y, x1 + half_w, y1 + half_h)
    avg_lb = get_average_color(img, x1 := x, y1 := y + half_h, x1 + half_w,
                               y1 + half_h)
    avg_rb = get_average_color(img, x1 := x + half_w, y1 := y + half_h,
                               x1 + half_w, y1 + half_h)
    avg_rt = get_average_color(img, x1 := x + half_w, y1 := y, x1 + half_w,
                               y1 + half_h)
    avgs = [avg_rt, avg_lt, avg_rb, avg_lb]
    # print(cur_color, all([all(abs(avg_all - avg) < DIFF) for avg in avgs]))
    res = not all([all(abs(avg_all - avg) < DIFF) for avg in avgs])
    if res:
        return res, avgs
    return res, avg_all


def build_and_render(pic, screen):
    pic.build()
    pic.render(screen, False)
    pg.display.flip()


# def saving(img, size, rgb):  # пока только для самого изображения
#     result = Image.new("RGB", (img.width, img.height))
#     r, g, b = rgb
#     for x in range(size[0]):
#         for y in range(size[1]):
#             result.putpixel((x, y), (r, g, b))
#     result.save("./images/compressed.jpg")


def main():
    global MAX_DEPTH
    image_path = './images/image2.jpg'
    image = Image.open(image_path)
    image = image.convert("RGBA")

    pg.init()
    screen = pg.display.set_mode(image.size)
    pic = QuadTree(image)
    build_and_render(pic, screen)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    MAX_DEPTH += 1
                    build_and_render(pic, screen)
                elif event.key == pg.K_DOWN:
                    MAX_DEPTH -= 1
                    build_and_render(pic, screen)


if __name__ == '__main__':
    main()
    # cProfile.run("main()", sort="tottime")
