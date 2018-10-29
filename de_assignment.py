import numpy as np
from matplotlib import pyplot as plt

'''Differential Equations Course Assignment
    Student: Elena Lukyanchikova
    Group: B17-02
    Variant: #12'''

'''To run the code:
    compile and run "de_assignment.py"
    Input :
    IVP information to the console(x0,y0), 
    upper-bound of [x0,xf] interval - xf,
    number of steps - n (affects precision)'''

'''method returns the value of y' = y *y * x + 3 * y * x'''


def f(x, y):
    try:
        return y * y * x + 3 * y * x
    except OverflowError:
        return float('inf')


'''method for computing the y value at x_current  corresponding to IVP (according to the Exact solution)
    y(x_current) = 1 / (const * exp(-3*x*x/2) - (1/3))  '''


def exact_solution(x_current, const):
    y = 1 / (const * np.exp((-3 * x_current * x_current) / 2) - (1 / 3))
    return y


'''method for computing the constant corresponding to IVP (according to the Exact solution)
    const = (3 + x)* exp(3*x*x/2) / (3 * y)  '''


def calculate_const(x0, y0):
    const = np.exp(3 * x0 * x0 / 2) * (3 + y0) / (3 * y0)
    return const


'''method for computing the x value in which the solution doesn't exist (according to feasible region of the Exact solution).
    x != sqrt(  (2/3)  * ln(|3*const|))'''


def calculate_asymptote(c):
    b = np.absolute([3 * c])
    a = np.sqrt((2 / 3) * np.log(b))
    return a


'''method for computing Approximate values using Euler method'''


def euler_method(x, y, h):
    y_n = h * f(x, y) + y
    return y_n


'''method for computing Approximate values using Euler improved method'''


def euler_improved_method(x, y, h):
    temp = y + h * f(x, y)
    y_n = y + (h / 2) * (f(x, y) + f(x + h, temp))
    return y_n


'''method for computing Approximate values using Runge-Kutta method of 4rd order'''


def runge_kutta_method(x, y, h):
    k1 = f(x, y)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(x + h, y + k3 * h)

    y_n = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_n




''' Variant 12:
    ( x0, y0) = ( 1, 3)
     xf = 5.5
     n = 1000'''
# Setting (x0,y0) for IVP,
# xf - upper-bound of x interval
x0 = float(input("Enter x0: "))  # 1
y0 = float(input("Enter y0: "))  # 3
xf = float(input("Enter max x: "))  # 5.5
n = int(input("Enter number of steps: "))  #
# compute the step
h = (xf - x0) / n
# fill array of x values (in [x0,xf]) with step h
x = np.linspace(x0, xf, n)

y_1 = np.zeros([n])
y_2 = np.zeros([n])
y_3 = np.zeros([n])
y_e = np.zeros([n])
y_1[0] = y0
y_2[0] = y0
y_3[0] = y0
y_e[0] = y0

error_1 = np.zeros([n])
error_2 = np.zeros([n])
error_3 = np.zeros([n])
error_1[0] = 0
error_2[0] = 0
error_3[0] = 0

# compute constant according to IVP - (x0,y0)
const = calculate_const(x0, y0)
# compute asymptote
asymptote = calculate_asymptote(const)

for i in range(1, n):
    # in all x points except those are asymptotes
    if x[i] != asymptote:
        # compute y values using numerical methods
        y_1[i] = euler_method(x[i - 1], y_1[i - 1], h)
        y_2[i] = euler_improved_method(x[i - 1], y_2[i - 1], h)
        y_3[i] = runge_kutta_method(x[i - 1], y_3[i - 1], h)
        # solve Initial Value Problem
        y_e[i] = exact_solution(x[i - 1], const)
    # compute Global Truncation Errors for each of approximations
    error_1[i] = y_e[i] - y_1[i]
    error_2[i] = y_e[i] - y_2[i]
    error_3[i] = y_e[i] - y_3[i]

print(asymptote)
print(const)
# plot Analytical Solution
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([y0 - 0.01, y0 + 40])
plt.plot(x, y_e, 'rx', label="Exact solution")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Value of y")
plt.title("Analytical solution: f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()

# plot Euler Method approximation
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([y0 - 0.01, y0 + 40])
plt.plot(x, y_2, 'c+', label="Euler Method")
plt.plot(x, y_e, 'r', label="Exact solution")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Euler Method: f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()

# plot Euler Improved Method approximation
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([y0 - 0.01, y0 + 40])
plt.plot(x, y_2, 'g+', label="Euler Improved Method")
plt.plot(x, y_e, 'r', label="Exact solution")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Euler Improved Method: f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()

# plot Runge-Kutta Method approximation
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([y0 - 0.01, y0 + 40])
plt.plot(x, y_3, 'y+', label="Runge-Kutta Method")
plt.plot(x, y_e, 'r', label="Exact solution")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Runge-Kutta Method: f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()

# plot Global Truncation Error for each of Numerical Methods
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([-1, 1])
plt.plot(x, error_1, 'c', label="Euler method")
plt.plot(x, error_2, 'g.', label="Euler Improved method")
plt.plot(x, error_3, 'y', label="Runge-Kutta method")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Global Truncation Error")
plt.title("Global Truncation Error: f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()

# plot all Numerical and Analytical solutions
axes = plt.gca()
if asymptote < xf:
    axes.set_xlim([x0 - 0.01, asymptote])
else:
    axes.set_xlim([0, xf])
axes.set_ylim([y0 - 0.01, y0 + 40])
plt.plot(x, y_e, 'r', label="Exact Solution")
plt.plot(x, y_3, 'cd', label="Runge-Kutta Method")
plt.plot(x, y_2, 'g*', label="Euler Improved Method")
plt.plot(x, y_1, 'y^', label="Euler Method")
plt.legend()
plt.xlabel("Value of x ")
plt.ylabel("Value of y")
plt.title("f(" + str(x0) + ") =" + str(y0) + ", n = " + str(n))
plt.show()
