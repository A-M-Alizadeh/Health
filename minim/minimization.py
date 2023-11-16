# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 16:33:35 2023

@author: d001834
"""

import numpy as np
import matplotlib.pyplot as plt
class SolveMinProbl:
    def __init__(self,y=np.ones((3,)),A=np.eye(3)):
        self.matr=A
        self.vect=y
        self.Np=A.shape[0]
        self.Nf=A.shape[1]
        self.sol=np.zeros((self.Nf,),dtype=float)
        return

    def plot_w_hat(self,title='Solution'):
        w_hat=self.sol
        plt.figure()
        plt.plot(w_hat)
        plt.ylabel('w_hat[n]')
        plt.xlabel('n')
        plt.title(title)
        plt.grid()
        plt.show()
        return
    
    def print_result(self,title='Estimated w'):
        print(title)
        print(self.sol)
        return
    
    def plot_y_yhat(self):
        what=self.sol
        yhat=self.matr@what
        plt.figure()
        plt.plot(self.vect,label='true')
        plt.plot(yhat,label='estim')
        plt.ylabel('y[n]')
        plt.xlabel('n')
        plt.legend()
        plt.grid()
        plt.show()
        return
        
        
        
        
        
class SolveLLS(SolveMinProbl):
    #    """This is the LLS solution for the regression problem"""
    def run(self):
        A=self.matr
        y=self.vect
        A1=np.linalg.inv(A.T@A)
        y1=A.T@y
        what=A1@y1
        self.sol=what
        return

if __name__=="__main__":
    Np=5
    Nf=3
    A=np.random.randn(Np,Nf)
    w=np.random.randn(Nf,1)
    eps=np.random.randn(Np,1)*0.1
    y=A@w+eps
    m=SolveLLS(y,A)
    m.run()
    m.print_result()
    m.plot_w_hat()
    m.plot_y_yhat()
    
        
        