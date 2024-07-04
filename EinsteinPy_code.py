import sympy
import numpy as np
from einsteinpy.symbolic import MetricTensor, ChristoffelSymbols, RiemannCurvatureTensor

syms = sympy.symbols('r theta phi')
#define the metric for 3-D spherical coordinates
metric =  np.zeros([3, 3], dtype = list)
metric[0,0] = 1
metric[1,1] = syms[0]**2
metric[2,2] = (syms[0]**2)*(sympy.sin(syms[1])**2)

#creating metric object
m_obj = MetricTensor(metric, syms)
print("Metric tensor for 3-d spherical coordinates:\n",m_obj.tensor())

#Calculating the christoffel symbols
ch = ChristoffelSymbols.from_metric(m_obj)
print("\nChristoffel symbols:")
print(ch.tensor())
#We are using indices of array to get value of single element at given position. Example is bellow code line.
print("\nElement of christoffel symbols at position [0,1,1]:", ch.tensor()[0,1,1])


#Calculating Riemann Tensor from Christoffel Symbols
rm1 = RiemannCurvatureTensor.from_christoffels(ch)
print("\nCalculating Riemann Tensor from Christoffel Symbols:")
print(rm1.tensor())

#Calculating Riemann Tensor from Metric Tensor
rm2 = RiemannCurvatureTensor.from_metric(m_obj)
print("\nCalculating Riemann Tensor from Metric Tensor:")
print(rm2.tensor())


"""Calculating the christoffel symbols for Schwarzschild Spacetime Metric"""
syms = sympy.symbols("t r theta phi")
G, M, c, a = sympy.symbols("G M c a")
#using metric values of schwarschild space-time
#"a" is schwarzschild radius

list2d =  np.zeros([4, 4], dtype = list)
list2d[0][0] = 1 - (a / syms[1])
list2d[1][1] = -1 / ((1 - (a / syms[1])) * (c ** 2))
list2d[2][2] = -1 * (syms[1] ** 2) / (c ** 2)
list2d[3][3] = -1 * (syms[1] ** 2) * (sympy.sin(syms[2]) ** 2) / (c ** 2)
sch = MetricTensor(list2d, syms)
print("\nSchwarzschild Spacetime Metric:\n", sch.tensor())

# single substitution
subs1 = sch.subs(a,0)
print("\nFor a=0\n",subs1.tensor())

# multiple substitution
subs2 = sch.subs([(a,0), (c,1)])
print("\nFor a=0,c=1\n",subs2.tensor())

sch_ch = ChristoffelSymbols.from_metric(sch)
print("\nChristoffel symbols for Schwarzschild Spacetime Metric:\n")
print(sch_ch.tensor())

