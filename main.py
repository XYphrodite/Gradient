from math import *
import numpy as np
import matplotlib.pyplot as plt
import threading

# threading._DummyThread._Thread__stop = lambda x: 42
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


def foo(x1, x2, n=0):
    if n == 0:
        return fooA(x1, x2)
    else:
        return fooB(x1, x2)


# вектор градиента
def grad(x, n):
    # h — приращение
    h = 0.0001
    x1 = x[0]
    x2 = x[1]
    # поиск производной
    y1 = (foo(x1 + h, x2, n) - foo(x1 - h, x2, n)) / (h * 2)
    y2 = (foo(x1, x2 + h, n) - foo(x1, x2 - h, n)) / (h * 2)
    return [y1, y2]


def gradient(e, x, n):
    Iterations = 0
    h = 0.00005
    gradient_vector = grad(x, n)
    while (pow(gradient_vector[0] + gradient_vector[1], 2) > e):
        x = [x[0] - h * gradient_vector[0], x[1] - h * gradient_vector[1]]
        gradient_vector = grad(x, n)
        Iterations += 1
        # print(Iterations)
    return x, Iterations


def down(e, x, n):
    # поиск приращения для x1
    def Find_x1h(x, gradientVector1):
        y = foo(x[0], x[1], n)
        h = 0
        while True:
            h += e
            yh = foo(x[0] - h * gradientVector1, x[1], n)
            if (yh < y):
                y = yh
                h += e
            else:
                h -= e
                break
        return h

    # поиск приращения для x2
    def Find_x2h(x, gradientVector2):
        y = foo(x[0], x[1], n)
        h = 0
        while True:
            h += e
            yh = foo(x[0], x[1] - h * gradientVector2, n)
            if yh < y:
                y = yh
                h += e
            else:
                h -= e
                break
        return h

    Iterations = 0
    gradient_vector = grad(x, n)
    while (pow(gradient_vector[0] + gradient_vector[1], 2) > e):
        x1h = Find_x1h(x, gradient_vector[0])
        x2h = Find_x2h(x, gradient_vector[1])
        x = [x[0] - x1h * gradient_vector[0], x[1] - x2h * gradient_vector[1]]
        gradient_vector = grad(x, n)
        Iterations += 1
    return x, Iterations


x = [10, 10]
xG, N1 = gradient(dlt / 5, x, 0)
xD, N2 = down(dlt / 5, x, 0)
xG1, N11 = gradient(dlt, x, 1)
# xD1, N21 = down(dlt, x, 1)
print("Градиент: ", xG)
print("Количество шагов: ", N1)
print("Метод наискорейшего спуска: ", xD)
print("Количество шагов: ", N2)

print("Gradient solution: ", xG1)
print("Iterations: ", N11)
# print("Steepest gradient solution: ", xD1)
# print("Iterations: ", N21)

fig, (ax2, ax1) = plt.subplots(1, 2)

levels = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10, 0.20, 0.30,
          0.4,
          0.5, 0.6, 0.7, 0.8, 0.9, 1]
xa1 = np.arange(2, 10, 0.125)
xa2 = np.arange(-5, 3, 0.125)
xa1, xa2 = np.meshgrid(xa1, xa2)
f2 = np.vectorize(fooA)
ya = f2(xa1, xa2)
conta = ax2.contour(xa1, xa2, ya, levels=levels)
ax2.plot(xG[0], xG[1], color="red", marker=".")
ax2.plot(xD[0], xD[1], color="blue", marker=".")

levels = [0, 0.5,0.75, 1, 1.5]
xb1 = np.arange(-1, 3, 0.01)
xb2 = np.arange(-1, 3, 0.01)
xb1, xb2 = np.meshgrid(xb1, xb2)
f2b = np.vectorize(fooB)
yb = f2b(xb1, xb2)
contb = ax1.contour(xb1, xb2, yb, levels=levels)
ax1.plot(xG1[0], xG1[1], color="pink", marker=".")
# ax1.plot(xD1[0], xD1[1], color="blue", marker=".")

plt.show()
