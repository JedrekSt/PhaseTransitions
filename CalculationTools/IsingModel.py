import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt

class op:
    sx = np.array([
            [0,1],
            [1,0]
        ])
    sy = np.array([
            [0,-1j],
            [1j,0]
        ])
    sz = np.array([
            [1,0],
            [0,-1]
        ])
    Id = np.array([
            [1,0],
            [0,1]
        ])

    @staticmethod
    def dag(A):
        return np.conj(A.T)
    
    @staticmethod
    def n_fold_kron(tab):
        ans = 1
        dim = len(tab)
        for i in range(dim):
            ans = np.kron(ans,tab[i])
        return ans


class Ising_sim:
    def __init__(self,dim,**kwargs):
        self.dim = dim
        self.S_ = self.Spins()
        self.H_ = self.Hamiltonian(**kwargs)

    def Spins(self):
        s_ = lambda x_ : [op.n_fold_kron([op.Id for _ in range(j)] + [x_] + [op.Id for _ in range(j+1,self.dim)]) for j in range(self.dim)]
        return (s_(op.sx),s_(op.sy),s_(op.sz))
    
    def Hamiltonian(self,**kwargs):
        J = kwargs.get("J",1)
        h = kwargs.get("h",0)
        left = - h * sum(self.S_[2][i] for i in range(self.dim))
        right = J * sum(sum(self.S_[k][i] @ self.S_[k][(i+1)%self.dim] for i in range(self.dim)) for k in range(3))
        return left + right
    
    def Get_data(self,**kwargs):
        T0 = kwargs.get("T0",0.001)
        Tk = kwargs.get("Tk",10)
        T = np.arange(T0,Tk,0.01)
        E = np.real(eig(self.H_)[0])
        U_ = lambda T_ : E @ np.exp(-E/T_) / sum(np.exp(-E/T_))
        U_internal = np.vectorize(U_)(T)
        Cp = np.gradient(U_internal)
        return T,U_internal,Cp,E

    def Spectrum(self):
        return np.real(eig(self.H_)[0])
    
class Ising_sim_star(Ising_sim):
    def Hamiltonian(self,**kwargs):
        J2 = kwargs.get("J2",kwargs.get("J",1))
        h = kwargs.get("h",0)
        H0 = np.kron(super().Hamiltonian(**kwargs),np.eye(2**self.dim))
        H1 = J2 * sum(sum(np.kron(self.S_[k][i],(self.S_[k][i]+self.S_[k][(i+1)%self.dim])) for i in range(self.dim)) for k in range(3))
        H2 = - h * sum(np.kron(np.eye(2**self.dim),self.S_[2][i]) for i in range(self.dim))
        return H0 + H1 + H2 