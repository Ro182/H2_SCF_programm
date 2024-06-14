import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import erf
import pandas as pd

class atomic_orbital:
    def __init__(self,position,alpha,coeff):
        self.position=position
        self.alpha=alpha
        self.coeff=coeff
        self.A=np.power(2*self.alpha/np.pi,3/4)
    def gaussian(self,alpha,radius):
        radius=np.linalg.norm(self.position-radius)
        return self.A*self.coeff*np.exp(-self.alpha*(radius**2))


class atom:
    def __init__(self,primitive_orbitals,charge,position):
        self.primitive_orbitals=primitive_orbitals
        self.charge=charge
        self.position=position
        
        
def F0(x):
    """Special function needed for the nuclear attraction and the two-electron integrals"""
    if x == 0:
        return 1.0
    else:
        return (1/np.sqrt(x))*(np.sqrt(np.pi)/2)*erf(np.sqrt(x))
    



def overlap(molecule):
    """Calculation of the Overlap integral S"""
    size=len(molecule)
    S=np.zeros((size,size))
     
    for i in range(size):
        for j in range(size):
            for primitive_a in molecule[i].primitive_orbitals:
                for primitive_b in molecule[j].primitive_orbitals:
                    N=primitive_a.A*primitive_b.A
                    coeff_a=primitive_a.coeff
                    coeff_b=primitive_b.coeff
                    a=primitive_a.alpha
                    b=primitive_b.alpha
                    position_a=primitive_a.position
                    position_b=primitive_b.position
                    distance=np.linalg.norm(position_a-position_b)
                    
                    S[i,j]+= N*coeff_a*coeff_b*\
                        np.power(np.pi/(a+b),3/2)*np.exp(-distance**2*a*b/(a+b))
    return S



def kinetic_energy(molecule):
    """Calculation of the kinetic energy of electrons"""
    
    size=len(molecule)

    T=np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            for primitive_a in molecule[i].primitive_orbitals:
                for primitive_b in molecule[j].primitive_orbitals:
                    N=primitive_a.A*primitive_b.A
                    coeff_a=primitive_a.coeff
                    coeff_b=primitive_b.coeff
                    a=primitive_a.alpha
                    b=primitive_b.alpha
                    position_a=primitive_a.position
                    position_b=primitive_b.position
                    distance=np.linalg.norm(position_a-position_b)                   
                    

                    T[i][j]+=N*coeff_a*coeff_b*0.5*((a*b/(a+b))*(6-4*(distance**2)*a*b/(a+b)))\
                        *np.power(np.pi/(a+b),3/2)*np.exp(-distance**2*a*b/(a+b))

    return T







def electron_nucleus_potential(molecule):
    """Calculation of the electron-nuclear attraction"""
    size=len(molecule)
    V=np.zeros((size,size))
    

    for a in molecule:
        Rc=a.position
        Z=a.charge
        for i in range(size):
            for j in range(size):
                for primitive_a in molecule[i].primitive_orbitals:
                    for primitive_b in molecule[j].primitive_orbitals:
                        
                        N=primitive_a.A*primitive_b.A
                        coeff_a=primitive_a.coeff
                        coeff_b=primitive_b.coeff
                        a=primitive_a.alpha
                        b=primitive_b.alpha
                        position_a=primitive_a.position
                        position_b=primitive_b.position
                        distance=np.linalg.norm(position_a-position_b)   
                        
                        distance=np.linalg.norm(position_a-position_b)
        
                        Rp=(a*position_a+b*position_b)/(a+b)
                        temp=(a+b)*(np.linalg.norm(Rp-Rc)**2)
                        
                        V[i,j]+=N*coeff_a*coeff_b*(-2*np.pi*Z/(a+b))*np.exp(-(distance**2)*a*b/(a+b))*F0(temp)


    return V






def electron_electron_matrix(molecule):
    """Calculation of two electron integrals"""
    size=len(molecule)
    V_ee=np.zeros([size,size,size,size])
    
    
    for i in range(size):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    for primitive_a in molecule[i].primitive_orbitals:
                        for primitive_b in molecule[j].primitive_orbitals:
                            for primitive_c in molecule[k].primitive_orbitals:
                                for primitive_d in molecule[l].primitive_orbitals:
                                    
                                    N=primitive_a.A*primitive_b.A*primitive_c.A*primitive_d.A
                                    coeff_T=primitive_a.coeff*primitive_b.coeff*primitive_c.coeff*primitive_d.coeff
                                    a=primitive_a.alpha
                                    b=primitive_b.alpha
                                    c=primitive_c.alpha
                                    d=primitive_d.alpha                    
                                    
                                    position_a=primitive_a.position
                                    position_b=primitive_b.position
                                    position_c=primitive_c.position
                                    position_d=primitive_d.position
                                    
                                    
                                    Rp=(a*position_a+c*position_c)/(a+c)
                                    Rq=(b*position_b+d*position_d)/(b+d)
                                    
                                    V_ee[i,j,k,l]+=N*coeff_T*(2*np.power(np.pi,5/2)/((a+c)*(b+d)*np.power(a+b+c+d,1/2)))*\
                                                   np.exp(-(a*c/(a+c))*np.linalg.norm(position_a-position_c)**2-(b*d/(b+d))*\
                                                          np.linalg.norm(position_b-position_d)**2)*\
                                                       F0(((a+c)*(b+d)/(a+b+c+d))*np.linalg.norm(Rp-Rq)**2)
                    

    return V_ee


def electron_electron_potential(density_matrix,V_ee):
    """Calculates the Coulomb and Exchange contribution to the Fock matrix"""
    size=density_matrix.shape[0]

    G=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                for l in range(size):
                    J=V_ee[i,k,j,l]
                    K=V_ee[i,k,l,j]
                    G[i,j]+=density_matrix[k,l]*(J-0.5*K)
                    
    return G



def electronic_energy_value(density_matrix,H,G):
    """Calculates the electronic energy"""
    electronic_energy=0
    size=density_matrix.shape[0]
    for i in range(size):
        for j in range(size):
            electronic_energy+=density_matrix[i,j]*(H[i,j]+0.5*G[i,j])
    return electronic_energy
    
    



def nuclear_repulsion(molecule):
    """Calculates the nuclear repulsion between nuclei"""
    ENN=0
    for i in range(len(molecule)):
        for j in range (len(molecule)):
            if j>i:
                ENN+=molecule[i].charge*molecule[j].charge/np.linalg.norm(molecule[i].position-molecule[j].position)
    return ENN




def new_density_matrix(MOs):
    """Creates a new density matrix basis on the solutions of the Fock equation"""
    size=MOs.shape[0]
    density_matrix=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            for k in range(n_occupied_orbitals):
                C=MOs[i,k]
                C_dagger=MOs[j,k]
                density_matrix[i,j]+=2*C*C_dagger
    return density_matrix
    









def scf(molecule,P, S,T,V_ne, G,max_steps=10,tol=1e-5):
    """Self consistent field method"""
    electronic_energy=0

    density_matrix=P
    
    for step in range(max_steps):
        
        electronic_energy_previous=electronic_energy
        
        G=electron_electron_potential(density_matrix, V_ee)
        H=T+V_ne
        
        #definition of Fock matrix
        F=H+G
        
        #diagonalization of Overlap S matrix
        S_inv=linalg.inv(S)
        S_inv_sqrt=linalg.sqrtm(S_inv)
        
        #diagonalization of Fock matrix
        F_diagonalized=np.dot(S_inv_sqrt,np.dot(F,S_inv_sqrt))
        eigenvalues,eigenvectors=linalg.eigh(F_diagonalized)
        
        MOs=np.dot(S_inv_sqrt,eigenvectors)
        
        #new density matrix based on the solution of the Fock matrix
        density_matrix=new_density_matrix(MOs)
        
        #calculus of the energy
        electronic_energy=electronic_energy_value(density_matrix, H, G)
        
        #check for convergence
        if abs(electronic_energy-electronic_energy_previous)<=tol:
            return electronic_energy
        
    print("Convergance not met")
    
    return electronic_energy



#other basis sets to test

#def2-SV(P) H Basis set 
#alphas=[13.00773,1.962079,0.444529,0.1219492]
#coeffs=[ 0.19682158E-01,0.13796524, 0.47831935,1.0000000]

#STO-3G H Minimal Basis
alphas=[0.3425250914E+01,0.6239137298E+00,0.1688554040E+00 ]
coeffs=[0.1543289673E+00,0.5353281423E+00,0.4446345422E+00]

#STO-6G
#alphas=[0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01, 0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00]   
#coeffs= [0.9163596281E-02,0.4936149294E-01,0.1685383049E+00, 0.3705627997E+00, 0.4164915298E+00, 0.1303340841E+00]




#Calculation of energy for different internuclear distances
energies=[]
nuclear_energies=[]

#definition of distances from 0.1 to 8, 100 points
distances=np.linspace(0.1,8,100)



for distance in distances:

    #definition of first hydrogen atom
    H1_primitives=[]
    H1_position=np.array([0.0,0.0,0.0])
    H1_charge=1.0

    #definition of second hydrogen atom
    H2_primitives=[]
    H2_position=np.array([distance,0.0,0.0])
    H2_charge=1.0
    
    #Defining the parameter for the basis set used for each 1s orbital of each hydrogen atom
    for i in range(len(alphas)):
        H1_primitives.append(atomic_orbital(H1_position, alphas[i],coeffs[i]))
        H2_primitives.append(atomic_orbital(H2_position, alphas[i],coeffs[i]))
        
    
    #definition of molecule
    molecule=[atom(H1_primitives,H1_charge,H1_position),atom(H2_primitives, H2_charge, H2_position)]
    n_occupied_orbitals=1
    
    #initial guess for density matrix P_{pq}=0
    density_matrix=np.zeros((len(molecule),len(molecule)))
    
    #calculation of kinetic energy of electrons
    T=kinetic_energy(molecule) 
    
    #calculation of two electron integrals
    V_ee=electron_electron_matrix(molecule)
    
    #calculation of the electron-nucli atraction
    V_ne=electron_nucleus_potential(molecule)
    
    #Calculation of the Coulomb and Exchange contributions to the Fock matrix
    G=electron_electron_potential(density_matrix, V_ee)
    
    #calculation of Overlap matrix S
    S=overlap(molecule)       
    
    #calculation of Nuclei repulsion
    ENN=nuclear_repulsion(molecule)
    
    #calculation of electronic energy by Self Consistent Field proccedure
    electronic_energy=scf(molecule, density_matrix, S, T, V_ne, G)
    
    #save the energies for plorring later on
    energies.append(electronic_energy)
    nuclear_energies.append(ENN)


#Plot electronic energy and nuclear-nuclear repulsion.
a0=0.52917721090
distance_in_A=distances*a0
total_energies=np.array(nuclear_energies)+np.array(energies)
plt.plot(distance_in_A,energies,label=r'Electron energy: $T_{e}+V_{Ne}$')
plt.plot(distance_in_A,nuclear_energies,label=r'Nuclear-Nuclear repulsion: $V_{NN}$')
plt.plot(distance_in_A,total_energies,label=r'Total energy')
plt.ylim((-2.5,2.5))
plt.xlim((0,4))
plt.ylabel(r"Energy $[Hartree]$")
plt.xlabel("Distance $[\AA]$")
plt.legend()
plt.savefig("Components_vs_Distance_H2_constristed.png",dpi=300 )






#Plot energy vs Different inter atomic distances
plt.figure()



data=pd.read_fwf('H2_data_sto_3g.dat',header=None)
data.columns=["Distance","Total Energy ORCA"]

#determine the minima of the energy curve and the value of the minima
x_min=0
y_min=0
for i in range( len(total_energies)):
    if total_energies[i]<y_min:
        y_min=total_energies[i]
        x_min=distance_in_A[i]
   
print(r"The equilibrium distance is: %e [Angstrom]"%x_min )

#determine the minima for the ORCA data obtained
x_min_orca=0
y_min_orca=0
for i in range( len(data['Distance'])):
    if data['Total Energy ORCA'][i]<y_min_orca:
        y_min_orca=data['Total Energy ORCA'][i]
        x_min_orca=data['Distance'][i]
   
print(r"The ORCA equilibrium distance is: %e [Angstrom]"%x_min_orca)



data.plot('Distance',label='Hartree Fock',linestyle='-.',c='b')

plt.plot(distance_in_A,total_energies,label=r'Total Energy code',alpha=0.8,c='0')
plt.legend()

plt.ylabel(r"Energy $[Hartree]$")
plt.xlabel("Distance $[\AA]$ ")
plt.ylim((-1.2,0))
plt.xlim((0.4*a0,7*a0))



plt.axhline(y=total_energies[-1],linestyle='--')
De=total_energies[-1]-min(total_energies)

plt.annotate("", xy=(x_min, min(total_energies)), xytext=(x_min,total_energies[-1]),
            arrowprops=dict(arrowstyle="<->"))

plt.annotate(r"$D_e$: %e[Hartree] "%De, xy=(x_min, min(total_energies)), xytext=(x_min,total_energies[-1]*1.2))


plt.annotate(r"Bond Length: %f $\AA$"%x_min, xy=(x_min, min(total_energies)), xytext=(x_min,total_energies[-1]+0.02))



plt.savefig("Energy_vs_Distance_H2_contristed.png",dpi=300 )


print("The minimun of the energy is: %e"%min(total_energies))
print("The Spectroscopy dissociation energy is %e [Hartree]"%(total_energies[-1]-min(total_energies)))

plt.plot()
