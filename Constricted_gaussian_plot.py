import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def atomic_orbital(alpha,radius):
    return np.exp(-alpha*np.power(radius,2))




def molecular_orbital(coeffs,atomic_orbital,alphas,radius):
    temp=0
    for i in range(len(coeffs)):
        temp+=coeffs[i]*atomic_orbital(alphas[i],radius)
    return temp


alphas=[0.3425250914E+01,0.6239137298E+00,0.1688554040E+00 ]
coeffs=[0.1543289673E+00,0.5353281423E+00,0.4446345422E+00]

x=np.linspace(-5, 5,1000)

for alpha in alphas:
    plt.plot(x,atomic_orbital(alpha, x),label=r"$\alpha=%.6f$"%alpha,linewidth=0.6)




plt.plot(x,molecular_orbital(coeffs, atomic_orbital, alphas, x),label='Molecular',c='black')

plt.legend()


plt.xlabel(r"Distance [$\AA$]")
plt.savefig("Constricted Gaussians", dpi=300)