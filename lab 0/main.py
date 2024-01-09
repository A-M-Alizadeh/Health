# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:24:18 2023

@author: d001834
"""
import numpy as np
import minim.minimization as mymin
Np=5
Nf=3
A=np.random.randn(Np,Nf)
w=np.random.randn(Nf,1)
eps=np.random.randn(Np,1)*0.1
y=A@w+eps
m=mymin.SolveLLS(y,A)
m.run()
m.print_result()
m.plot_w_hat()
m.plot_y_yhat()
