import math


def convert_to_number(name):
    return int.from_bytes(name.encode(), 'little')


def convert_from_number(number):
    return number.to_bytes(math.ceil(number.bit_length() / 8), 'little').decode()
