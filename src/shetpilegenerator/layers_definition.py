import math
import matplotlib.pyplot as plt


def find_x_coordinate(point_c, opposite_length, y_coordinate_b):
    # Extract the coordinates of point C
    x_c, y_c, z_c = point_c
    # Calculate the length of BC
    bc_length = abs(y_c - y_coordinate_b)
    # Calculate the angle theta in radians
    theta = math.atan(bc_length / opposite_length)
    # Calculate the x-coordinate of point C
    x_c = opposite_length / math.tan(theta)
    return x_c


def define_layers_with_dike_geometry(height, length, angle):
    # last point of the top 1 layer is the first point of the top 2 layer
    first_point = (-20, 0, 0)
    radians = math.radians(angle)
    distance = math.tan(radians) * height
    second_point = (first_point[0] + distance * math.cos(radians), height, 0)
    third_point = (second_point[0] + length, height, 0)
    fourth_point = (find_x_coordinate(third_point, height, 2), 2, 0)
    TOP_1 = [
        (-80, -2, 0),
        (-20, -2, 0),
        first_point,
        (-80, 0, 0),
    ]
    TOP_2 = [
        (-20, -2, 0),
        (25, -2, 0),
        fourth_point,
        third_point,
        second_point,
        first_point,
    ]
    TOP_3 = [
        (25, -2, 0),
        (80, -2, 0),
        (80, 2, 0),
        fourth_point,
    ]
    MIDDLE = [
        (-80, -8, 0),
        (-20, -8, 0),
        (25, -8, 0),
        (80, -8, 0),
        (80, -2, 0),
        (25, -2, 0),
        (-20, -2, 0),
        (-80, -2, 0)
    ]
    BOTTOM = [
        (-80, -15, 0),
        (-20, -15, 0),
        (25, -15, 0),
        (80, -15, 0),
        (80, -8, 0),
        (25, -8, 0),
        (-20, -8, 0),
        (-80, -8, 0),
    ]
    return [TOP_1, TOP_2, TOP_3, MIDDLE, BOTTOM]


def test_define_layers_with_dike_geometry(height, length, angle):
    # last point of the top 1 layer is the first point of the top 2 layer
    first_point = (-20, 0, 0)
    radians = math.radians(angle)
    distance = math.tan(radians) * height
    second_point = (first_point[0] + distance * math.cos(radians), height, 0)
    third_point = (second_point[0] + length, height, 0)
    fourth_point = (find_x_coordinate(third_point, height, 2), 2, 0)
    return [first_point, second_point, third_point, fourth_point ]



def define_layers():
    TOP_1 = [
        (-80, -2, 0),
        (-20, -2, 0),
        (-20, 0, 0),
        (-80, 0, 0),
    ]
    TOP_2 = [
        (-20, -2, 0),
        (25, -2, 0),
        (25, 2, 0),
        (8, 8, 0),
        (0, 8, 0),
        (-20, 0, 0),
    ]
    TOP_3 = [
        (25, -2, 0),
        (80, -2, 0),
        (80, 2, 0),
        (25, 2, 0),
    ]
    MIDDLE = [
        (-80, -8, 0),
        (-20, -8, 0),
        (25, -8, 0),
        (80, -8, 0),
        (80, -2, 0),
        (25, -2, 0),
        (-20, -2, 0),
        (-80, -2, 0)
    ]
    BOTTOM = [
        (-80, -15, 0),
        (-20, -15, 0),
        (25, -15, 0),
        (80, -15, 0),
        (80, -8, 0),
        (25, -8, 0),
        (-20, -8, 0),
        (-80, -8, 0),
    ]
    return [TOP_1, TOP_2, TOP_3, MIDDLE, BOTTOM]


def define_layers_with_sheetpile(x_location, length):
    """
    Define the layers including the sheetpile

    :param x_location: x location between the sheetpile and right side the model
    :param length: length of the sheetpile
    :return:
    """
    TOP_1 = [
        (-80, -2, 0),
        (-20, -2, 0),
        (-20, 0, 0),
        (-80, 0, 0),
    ]
    TOP_2_left = [
        (-20, -2, 0),
        (25, -2, 0),
        (25, 2, 0),
        (8, 8, 0),
        (0, 8, 0),
        (-20, 0, 0),
    ]
    TOP_2_right = [
        (-20, -2, 0),
        (25, -2, 0),
        (25, 2, 0),
        (8, 8, 0),
        (0, 8, 0),
        (-20, 0, 0),
    ]
    TOP_3 = [
        (25, -2, 0),
        (80, -2, 0),
        (80, 2, 0),
        (25, 2, 0),
    ]
    MIDDLE = [
        (-80, -8, 0),
        (-20, -8, 0),
        (25, -8, 0),
        (80, -8, 0),
        (80, -2, 0),
        (25, -2, 0),
        (-20, -2, 0),
        (-80, -2, 0)
    ]
    BOTTOM = [
        (-80, -15, 0),
        (-20, -15, 0),
        (25, -15, 0),
        (80, -15, 0),
        (80, -8, 0),
        (25, -8, 0),
        (-20, -8, 0),
        (-80, -8, 0),
    ]
    return [TOP_1, TOP_2, TOP_3, MIDDLE, BOTTOM]