import numpy as np
Np=5#Number of rows - number of patients
Nf=4#Number of columns - number of features
A=np.random.randn(Np,Nf) #Gauss. randommatrixA, shape (Np,Nf)
w=np.random.randn(Nf) #Gauss. randomvector wid, shape (Nf,)
y=A@w # shape (Np,)