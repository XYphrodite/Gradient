from math import *
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


def foo(x1, x2, n=0):
    if n == 0:
        return fooA(x1, x2)
    else:
        return fooB(x[0], x[1])


def grad(x, n):
    # gradient vector
    h = 0.0001
    x1 = x[0]
    x2 = x[1]
    return [(foo(x1 + h, x2, n) - foo(x1 - h, x2, n)) / (h * 2), (foo(x1, x2 + h, n) - foo(x1, x2 - h, n)) / (h * 2)]


def conj_grad(e, x, n):
    xt = x
    k = 0
    h = 1
    gradvect = grad(xt, n)
    while (pow(gradvect[0] + gradvect[1], 2) > e):
        xt = [xt[0] - h * gradvect[0], xt[1] - h * gradvect[1]]
        gradvect = grad(xt, n)
        k += 1
    return xt, k


def gradient_steepest(e, xt, n):
    x = xt

    def f_x1h(x1, x2, gradx1):
        y = foo(x1, x2, n)
        h = 0
        while (True):
            yh = foo(x1 - (h + e) * gradx1, x2, n)
            if (yh < y):
                y = yh
                h += e
            else:
                break
        return h

    def f_x2h(x1, x2, gradx2):
        y = foo(x1, x2, n)
        h = 0
        while (True):
            yh = foo(x1, x2 - (h + e) * gradx2, n)
            if (yh < y):
                y = yh
                h += e
            else:
                break
        return h

    N = 0
    gradvect = grad(x, n)
    while (pow(gradvect[0] + gradvect[1], 2) > e):
        x1h = f_x1h(x[0], x[1], gradvect[0])
        x2h = f_x2h(x[0], x[1], gradvect[1])
        x = [x[0] - x1h * gradvect[0], x[1] - x2h * gradvect[1]]
        gradvect = grad(x,n)
        N+=1
    return x, N


x = [10, 10]
xG, N1 = conj_grad(dlt / 500, x, 0)
xD, N2 = gradient_steepest(dlt/5, x, 0)
# xG1, N11 = conj_grad(dlt / 42, x, 1)
# xD1, N21 = gradient_steepest(dlt / 42, x, 1)
print("Gradient solution: ", xG)
print("Iterations: ", N1)
print("Steepest gradient solution: ", xD)
print("Iterations: ", N2)

# print("Gradient solution: ", xG1)
# print("Iterations: ", N11)
# print("Steepest gradient solution: ", xD1)
# print("Iterations: ", N21)

fig, ( ax2) = plt.subplots(1, 1)

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

#xb1 = np.arange(-27, 28, 0.125)
#xb2 = np.arange(-24, 26, 0.125)
#xb1, xb2 = np.meshgrid(xb1, xb2)
#f2b = np.vectorize(fooB)
#yb = f2b(xb1, xb2)
# contb = ax1.contour(xb1, xb2, yb, levels=10)
# ax1.plot(xG1[0], xG1[1], color="pink", marker=".")
# ax1.plot(xD1[0], xD1[1], color="blue", marker=".")
plt.show()
