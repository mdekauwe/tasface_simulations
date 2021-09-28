import matplotlib.pyplot as plt
import numpy as np
import sys


def get_weibull_params(p12=None, p50=None, p88=None):
    """
    Calculate the Weibull sensitivity (b) and shape (c) parameters
    """

    if p12 is not None and p50 is not None:
        px1 = p12
        x1 = 12. / 100.
        px2 = p50
        x2 = 50. / 100.
    elif p12 is not None and p88 is not None:
        px1 = p12
        x1 = 12. / 100.
        px2 = p88
        x2 = 88. / 100.
    elif p50 is not None and p88 is not None:
        px1 = p50
        x1 = 50. / 100.
        px2 = p88
        x2 = 88. / 100.

    print(x1, x2)
    num = np.log(np.log(1. - x1) / np.log(1. - x2))
    den = np.log(px1) - np.log(px2)
    c = num / den

    b = px1 / ((-np.log(1 - x1))**(1. / c))

    return b, c

Kmax = 2.0
p12 = 1.73
p50 = 3.0023835

(b, c) = get_weibull_params(p12=p12, p50=p50)
print(b, c)

p = np.linspace(-5, 0.0)
weibull = np.exp(-(-p / b)**c)

# Whole plant hydraulic conductance, including vulnerability to
# cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
Kplant = Kmax * weibull

plt.plot(p, Kplant/Kmax)
plt.show()

sys.exit()

b_plant = 2
c_plant = 2.0
Kmax = 1.5

p = np.linspace(-5, 0.0)

weibull = np.exp(-1.0 * (p / b_plant)**c_plant)

# Whole plant hydraulic conductance, including vulnerability to
# cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
Kplant = Kmax * weibull

plt.plot(p, Kplant/Kmax)

b_plant = 4
c_plant = 2.0
Kmax = 1.5

p = np.linspace(-5, 0.0)

weibull = np.exp(-1.0 * (p / b_plant)**c_plant)

# Whole plant hydraulic conductance, including vulnerability to
# cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
Kplant = Kmax * weibull

plt.plot(p, Kplant/Kmax)


b_plant = 1
c_plant = 2.0


p = np.linspace(-5, 0.0)

weibull = np.exp(-1.0 * (p / b_plant)**c_plant)

# Whole plant hydraulic conductance, including vulnerability to
# cavitation, kg timestep (e.g. 30 min-1)-1 MPa-1 m-2
Kplant = Kmax * weibull

plt.plot(p, Kplant/Kmax)

plt.show()
