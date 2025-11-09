import numpy as np
class Network:
    def __init__(self, structure):
        self.structure = structure
        self.num_layers = len(structure) 
        self.Bₙ = [np.random.randn(l, 1) for l in structure[1:]] 
        
        self.Wₙ = [np.random.randn(l, next_l) 
                   for l, next_l in zip(structure[:-1], structure[1:])]
        
    def backprop(self, x, y):        
        მJⳆმBₙₛ = [np.zeros(b.shape) for b in self.Bₙ]
        მJⳆმWₙₛ = [np.zeros(W.shape) for W in self.Wₙ]

        Zₙ = [] 
        Aₙ = [] 
        for b, W in zip(self.Bₙ, self.Wₙ):
            z = W.T @ a + b if Zₙ else W.T @ x + b
            a = σ(z)
            Zₙ.append(z)
            Aₙ.append(a)
        
        H = self.num_layers-2
        for L in range(H, -1, -1):
            δ =  σࠤ(Zₙ[L]) * (self.Wₙ[L+1] @ δ) if L != H else ᐁC(Aₙ[L], y) * σࠤ(Zₙ[L])
            მJⳆმBₙₛ[L] = δ
            მJⳆმWₙₛ[L] = Aₙ[L-1] @ δ.T if L != 0 else x @ δ.T
        
        return (მJⳆმBₙₛ, მJⳆმWₙₛ)
    
    def gradient_descent(self, mini_batch, λ):
        მJⳆმBₙ= [np.zeros(b.shape) for b in self.Bₙ]
        მJⳆმWₙ = [np.zeros(W.shape) for W in self.Wₙ]

        for x, y in mini_batch:
            მJⳆმBₙₛ, მJⳆმWₙₛ = self.backprop(x, y)
            მJⳆმBₙ = [მJⳆმb + მJⳆმbₛ for მJⳆმb, მJⳆმbₛ in zip(მJⳆმBₙ, მJⳆმBₙₛ)]  
            მJⳆმWₙ = [მJⳆმW + მJⳆმWₛ for მJⳆმW, მJⳆმWₛ in zip(მJⳆმWₙ, მJⳆმWₙₛ)]

        d = len(mini_batch)
        self.Wₙ = [W - λ/d * მJⳆმW for W, მJⳆმW in zip(self.Wₙ, მJⳆმWₙ)]
        self.Bₙ = [b - λ/d * მJⳆმb for b, მJⳆმb in zip(self.Bₙ, მJⳆმBₙ)]
            
    def train(self, epochs, training_data, λ):
        for j in range(epochs):
            for mini_batch in training_data:
                self.gradient_descent(mini_batch, λ)       
                


def ᐁC(aᴺ, y):    
    return (aᴺ-y)                 #so we can easily change the cost.


def σ(z):
    return 1.0/(1.0+np.exp(-z))


def σࠤ(z):
    return σ(z)*(1-σ(z))


my_net = Network([3, 2 ,2])
print("Initial Weights:")
print(my_net.Wₙ[0])
#the following generates a list of cnt vectors of length dim.
random_vectors = lambda dim, cnt: [np.random.rand(dim, 1) for i in range(cnt)]
random_batch= list(zip(random_vectors(3, 64) , random_vectors(2, 64)))
my_net.gradient_descent(random_batch, 3.0)
print("Optimized Weights:")
print(my_net.Wₙ[0])