import math
from math import *
from tkinter import S
import numpy as np
import matplotlib.pyplot as plt

E = 0.01
dlt = E / 100


def fooA(x1, x2):
    a = 6
    b = -1
    c = 3
    d = 4
    alf = 70
    return (pow(((x1 - a) * cos(alf) + (x2 - b) * sin(alf)), 2)) / (c * c) + \
           (pow(((x2 - b) * cos(alf) - (x1 - a) * sin(alf)), 2)) / (d * d)


def fooB(x1, x2):
    return 100 * pow((x2 - pow(x1, 2)), 2) + pow((1 - x1), 2)


def foo(x1, x2, n):
    if n == 0:
        return fooA(x1, x2)
    else:
        return fooB(x1, x2)


def Gradient():
    return 0


print("Start:")
print("Градиент: ",Gradient())