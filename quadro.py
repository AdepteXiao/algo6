import numpy as np
from PIL import Image

image = Image.open('./images/image1.jpg')
max_depth = 8
diff = np.array([10, 10, 10])


class QuadTree:
    class QuadNode:
        def __init__(self, x, y, width, height, img, color=None):
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
            child_size = (self.w // 2, self.h // 2)
            self.children.append(
                QuadTree.QuadNode(self.x, self.y, *child_size, image))
            self.children.append(
                QuadTree.QuadNode(self.x + child_size[0], self.y, *child_size,
                                  image))
            self.children.append(
                QuadTree.QuadNode(self.x, self.y + child_size[1], *child_size,
                                  image))
            self.children.append(
                QuadTree.QuadNode(self.x + child_size[0], self.y +
                                  child_size[1], *child_size, image))

        def get_all_nodes(self):
            all_nodes = [self]
            for child_node in self.children:
                all_nodes.extend(child_node.get_all_nodes())
            return all_nodes

        def set_color(self, color):
            self.color = color

    def __init__(self, img):
        self.img = img
        self.root_node = QuadTree.QuadNode(0, 0, *image.size, img)

    def build(self):
        def build_inner(cur_node, cur_color, depth=0):
            if depth >= max_depth:
                cur_node.set_color(cur_color)
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


def get_average_color(img, x, y, width, height):
    box = (x, y, x + width, y + height)
    region = img.crop(box)
    avg_color = region.convert('RGB').resize((1, 1)).getpixel((0, 0))
    return np.array(avg_color)


def should_divide(img, cur_color, x, y, width, height):
    avg_all = cur_color
    avg_lt = get_average_color(img, x, y, width // 2, height // 2)
    avg_lb = get_average_color(img, x, y + height // 2, width // 2,
                               height // 2)
    avg_rb = get_average_color(img, x + width // 2, y + height // 2,
                               width // 2, height // 2)
    avg_rt = get_average_color(img, x + width // 2, y, width // 2,
                               height // 2)
    avgs = [avg_rt, avg_lt, avg_rb, avg_lb]
    res = any([all(abs(avg_all - avg) > diff) for avg in avgs])
    if res:
        return res, avgs
    return res, avg_all


def saving(img, size, rgb):  # пока только для самого изображения
    result = Image.new("RGB", (img.width, img.height))
    r, g, b = rgb
    for x in range(size[0]):
        for y in range(size[1]):
            result.putpixel((x, y), (r, g, b))
    result.save("./images/compressed.jpg")


if __name__ == '__main__':
    pic = QuadTree(image)
    pic.build()
    pic.print_nodes()
