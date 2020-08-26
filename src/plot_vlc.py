import matplotlib.pyplot as plt
import numpy as np




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
