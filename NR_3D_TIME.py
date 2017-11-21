'''
This is a code for calculating the Temperature gradient on the surface of a cantilever, irradiated with a Gaussian
laser spot at a specified position on the surface centerline.
Based on theory by James Davis - Refer to Fiber Heating 8.0.docx

Author: Anupum Pant
Date: June 13, 2017
Updated: July 17, 2017
'''

import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os
t_min = 0
t_max = 100e-6
t_total = 10

# Set variables here
def temp_time(t):
    length = 20e-6                      # Length of the NR (m)
    width = 5e-6                        # Width of the NR (m)
    height = 110e-9                      # Height of the NR (m)
    T0 = 77                             # Cryostat temperature (K)
    Tinf = 298                          # Ambient temperature (K)
    kappa = 20                          # Thermal conductivity (W/(m.K))
    n_CdS = 2.5513                      # Refractive index real part (532 nm)
    k_CdS = 0.0057657                   # Refractive index imag part (532 nm)
    #n_CdS = 2.6399                     # Refractive index real part (514 nm)
    #k_CdS = 0.055754                   # Refractive index imag part (514 nm)
    ref = 0.19081                       # Reflectivity (532 nm)
    #ref = 0.20317                      # Reflectivity (514 nm)
    spotDiameter = 2.5e-6               # Laser spot diameter (m)
    w0 = spotDiameter/2                 # Laser spot radius (m)
    sampleRate = 10                     # Spatial sample rate
    numSum = 5                          # Eigenvalue sample rate
    lambda0 = 532e-9                    # laser wavelength (m)
    laser_power = 5                     # Laser power in milliwatts
    Power = laser_power*10**-3*(1-ref)  # Laser power (W)
    spotPos = 0.75                      # Normalized position of laser spot (0: base, 1: tip)
    base_area = width*height

    rho = 4826
    heatcap = 470
    tau = (kappa*t)/(4826*heatcap*length**2)
    #print(np.exp(-2.46*tau))

    # Few calculations, using defined variables above.
    r_max = np.sqrt(0.5*np.log(100)*np.square(w0))
    kpp = (2*math.pi*k_CdS)/(lambda0)
    L0 = spotPos*length
    I0 = (Power*2)/(math.pi*np.square(w0))

    theta = np.linspace(0, 2*math.pi, sampleRate)
    r = np.linspace(0, r_max, sampleRate)
    u_max = (2*np.square(r_max))/(np.square(w0))
    #u = (2*np.square(r))/(np.square(w0))
    u = np.linspace(0, u_max, sampleRate)
    a = length/width
    b = length/height
    l = np.linspace(0, numSum, numSum+1)
    m = np.linspace(0, numSum, numSum+1)
    n = np.linspace(0, numSum, numSum+1)
    Q_gen = 0.5*I0*math.pi*np.square(w0)*(1-np.exp(-2*kpp*height))*(1-np.exp(-u_max))

    # Eigen functions. X, Y and Z
    def xEigFun(xi,l):
        if l == 0:
            return 1
        else:
            return np.sqrt(2)*np.cos(l*math.pi*xi)

    def yEigFun(eta,m):
            return np.sqrt(2)*np.sin((m+0.5)*math.pi*eta)

    def zEigFun(zeta,n):
        if n == 0:
            return 1
        else:
            return np.sqrt(2)*np.cos(n*math.pi*zeta)

    # Conversion of Eigen fuctions for accepting vector inputs
    xfunc = np.vectorize(xEigFun, otypes=[np.float])
    yfunc = np.vectorize(yEigFun, otypes=[np.float])
    zfunc = np.vectorize(zEigFun, otypes=[np.float])

    # Funtion to Calculate the double integral for respective 'l' and 'm' values.
    # Optional output: plot of first integral1 vs. u
    dbl_int = np.zeros((numSum,numSum,numSum,))
    z_ana = np.zeros((numSum,numSum,numSum))
    gammaEigSquared = np.zeros((numSum,numSum,numSum))
    for ll in range(0, numSum):
        for mm in range(0, numSum):
            for nn in range(0, numSum):
                integral_xy = np.zeros(sampleRate)
                for i in range(0,sampleRate):
                    sigma_xy = ((0.5*w0**2*length*kpp)/(width*kappa*T0))*I0*np.exp(-u[i])
                    xi = (0.5)+((np.sqrt(u[i]/2))*(w0/(width))*(np.sin(theta)))
                    eta = ((L0/length)-((w0/length)*(np.sqrt(u[i]/2))*np.cos(theta)))
                    integral_xy[i] = np.trapz((sigma_xy*xfunc(xi,ll)*yfunc(eta,mm)), x=theta)
                dbl_int[ll,mm,nn] = (np.trapz(integral_xy,x=u))          # Calculation and return of 2nd integral
                gammaEigSquared[ll,mm,nn] = (np.square(a*ll*math.pi) + np.square((mm+0.5)*math.pi) + np.square(b*nn*math.pi))
                if nn == 0:
                    z_ana[ll,mm,nn] = (1 - np.exp(-2 * kpp * height)) / (2 * kpp * height)
                else:
                    z_ana[ll,mm,nn] = ((np.sqrt(2)*2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*(-1)**nn)
                A_lmn = np.exp(-gammaEigSquared*tau)+((1 / gammaEigSquared) * dbl_int * z_ana)
                #print("Status", ((ll/(numSum-1))*100)"% complete")
    #print("A(000) = ", A_lmn[0,0,0])
    #print("A(100) = ", A_lmn[1,0,0])
    #print("A(010) = ", A_lmn[0,1,0])
    #print("A(001) = ", A_lmn[0,0,1])
    #print(gammaEigSquared)

    # Function to get A_lmn for respective values of l, m and n
    triple_sum_vals = np.zeros((numSum+1,numSum+1,numSum+1))
    def U(xi, eta, zeta):
        for ll in range(0, numSum):
            for mm in range(0, numSum):
                for nn in range(0, numSum):
                    triple_sum_vals[ll,mm,nn] = A_lmn[ll,mm,nn]*xEigFun(xi,ll)*yEigFun(eta,mm)*zEigFun(zeta,nn)
        triple_sum = np.sum(np.sum(np.sum(triple_sum_vals)))
        return ((((triple_sum))*T0))
    return(U(0.5,0.75,0))

temp = np.zeros(t_total)
t = np.linspace(t_min, t_max, t_total)
for i in range(0, t_total):
    temp[i] = temp_time(t[i])
    print(t[i], temp[i])

plt.plot(t, temp)
plt.show()
