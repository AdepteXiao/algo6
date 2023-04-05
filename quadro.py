import argparse
import os
import sys
import threading
from math import ceil
from pprint import pprint
from typing import List
import cProfile
import cv2
from threading import Thread
from queue import Queue

import numpy as np
from PIL import Image
import pygame as pg

DIFF = np.array([5, 5, 5])
MAX_DEPTH = 1
IS_BORDERS = False


class QuadTree:
    class QuadNode:
        def __init__(self, img: Image, x, y, width, height, color=None):
            """
            Класс элемента квадродерева
            :param img: исходное изображение
            :param x: координата x
            :param y: координата y
            :param width: ширина элемента
            :param height: высота элемента
            :param color: цвет элемента
            """
            self.x = x
            self.y = y

            self.w = width
            self.h = height

            self.image = img.crop((x, y, width + x, height + y))
            self.color = color
            self.children = []

        def is_leaf(self):
            """
            Метод проверки элемента на наличие детей
            :return: True, если элемент является листом, иначе False
            """
            return not self.children

        def split_to_four(self):
            """
            Метод разделения элемента на 4 ребенка
            """
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

        def render(self, screen):
            """
            Метод отрисовки элемента
            :param screen: окно pygame
            """
            box_rect = pg.Rect(self.x, self.y, self.w, self.h)
            pg.draw.rect(screen, self.color, box_rect)
            if IS_BORDERS:
                pg.draw.rect(screen, (0, 0, 0), box_rect, 1)

        def get_all_nodes(self):
            """
            Метод получения всех элементов после текущего
            :return: массив из элементов
            """
            all_nodes = [self]
            for child_node in self.children:
                all_nodes.extend(child_node.get_all_nodes())
            return all_nodes

        def set_color(self, color):
            """
            Сеттер цвета элемента
            :param color: цвет в формате RGB
            """
            self.color = color

        def __repr__(self):
            """
            Метод для наглядного представления элементов
            """
            return f"Node({self.x}, {self.y}, {self.w}, {self.h}, " \
                   f"color={self.color}"

    def __init__(self, img):
        """
        Класс квадро дерева
        :param img: исходное изображение
        """
        self.img = img
        self.root_node = QuadTree.QuadNode(img, 0, 0, *self.img.size)

    def build(self):
        """
        Метод построения дерева
        """
        def build_inner(cur_node, cur_color, depth=0):
            """
            Построение отдельного элемента
            :param cur_node: элемент
            :param cur_color: цвет элемента
            :param depth: текущая глубина
            """
            lock = threading.Lock()
            if depth >= MAX_DEPTH:
                with lock:
                    cur_node.set_color(cur_color)
                return
            if cur_node.w == 1 or cur_node.h == 1:
                with lock:
                    cur_node.set_color(cur_color)
                return

            resp, cols = should_divide(self.img, cur_color,
                                       cur_node.x, cur_node.y, cur_node.w,
                                       cur_node.h)
            if resp is False:
                with lock:
                    cur_node.set_color(cur_color)
                return
            with lock:
                cur_node.split_to_four()
            threads = []
            for node, col in zip(cur_node.children, cols):
                thread = threading.Thread(target=build_inner,
                                          args=(node, col, depth + 1))
                thread.start()
                threads.append(thread)

            for process in threads:
                process.join()

        self.root_node.children.clear()
        build_inner(self.root_node,
                    get_average_color(self.img,
                                      self.root_node.x,
                                      self.root_node.y,
                                      self.root_node.w,
                                      self.root_node.h))

    def print_nodes(self):
        """
        Метод для вывода всех элементов дерева
        """
        def _print_nodes(node: "QuadTree.QuadNode", depth=0):
            print("\t" * depth, node)
            for child in node.children:
                _print_nodes(child, depth + 1)

        _print_nodes(self.root_node)
        print("#" * 20)

    def render(self, screen):
        """
        Метод отрисовки листьев дерева
        :param screen: окно pygame
        """
        leaves = self.find_leaves()
        for leaf in leaves:
            leaf.render(screen)

    def find_leaves(self) -> List["QuadTree.QuadNode"]:
        """
        Метод для поиска листьев дерева
        :return: все листья дерева
        """
        def find_leaves_inner(cur_node: "QuadTree.QuadNode"):
            """
            Метод для нахождения листьев для переданного элемента
            :param cur_node: элемент
            :return: все листья
            """
            if cur_node.is_leaf():
                return [cur_node]
            else:
                leaves = []
                for child in cur_node.children:
                    leaves.extend(find_leaves_inner(child))
                return leaves

        return find_leaves_inner(self.root_node)


def get_average_color(img, *box):
    """
    Метод получения среднего цвета элемента
    :param img: исходное изображение
    :param box: координаты элемента
    :return: средний цвет
    """
    region = img.crop(box)
    rg_array = np.array(region)
    rg_mean = cv2.mean(rg_array)[:-1]
    return np.array(rg_mean)


def should_divide(img, cur_color, x, y, width, height):
    """
    Метод проверки необходимости делить элемент на детей
    :param img: исходное изображение
    :param cur_color: средний цвет элемента
    :param x: координата х
    :param y: координата у
    :param width: ширина
    :param height: высота
    :return: True и средние цвета детей, если нужно и
    False и средний цвет элемента, в ином случае
    """
    avg_all = cur_color
    half_w, half_h = width / 2, height / 2
    avg_lt = get_average_color(img, x1 := x, y1 := y, x1 + half_w, y1 + half_h)
    avg_lb = get_average_color(img, x1 := x, y1 := y + half_h, x1 + half_w,
                               y1 + half_h)
    avg_rb = get_average_color(img, x1 := x + half_w, y1 := y + half_h,
                               x1 + half_w, y1 + half_h)
    avg_rt = get_average_color(img, x1 := x + half_w, y1 := y, x1 + half_w,
                               y1 + half_h)
    avgs = [avg_lt, avg_rt, avg_lb, avg_rb]
    res = not all([all(abs(avg_all - avg) < DIFF) for avg in avgs])
    if res:
        return res, avgs
    return res, avg_all


def build_and_render(pic, screen):
    """
    Метод построения и отрисовки дерева
    :param pic: исходное изображение
    :param screen: окно pygame
    """
    pic.build()
    pic.render(screen)
    pg.display.flip()


def main():
    """
    Точка входа
    """
    parser = argparse.ArgumentParser(description='QuadTree image processing')
    parser.add_argument('input_path', metavar='input_path', type=str,
                        help='Path to input image file')
    parser.add_argument('-o', '--output_path', type=str,
                        help='Path to output image file')
    args = parser.parse_args()
    if not os.path.isfile(args.input_path):
        print(f"File not found: {args.input_path}")
        sys.exit()
    global MAX_DEPTH
    # image_path = './images/image4.jpg'
    image = Image.open(args.input_path)
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
                if event.key == pg.K_UP and MAX_DEPTH <= 7:
                    MAX_DEPTH += 1
                    build_and_render(pic, screen)
                elif event.key == pg.K_DOWN and MAX_DEPTH >= 0:
                    MAX_DEPTH -= 1
                    build_and_render(pic, screen)
                elif event.key == pg.K_RETURN:
                    # pg.image.save(screen, './images/image_new.jpg')
                    output_path = args.output_path or './images/image_new.jpg'
                    pg.image.save(screen, output_path)


if __name__ == '__main__':
    main()
    # cProfile.run("main()", sort="tottime")
