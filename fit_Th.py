import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import pickle
import os
import scipy as sp
from os import sys
import pyvisco as visco
from pyvisco import styles
styles.format_fig()

def spec1(tau_n):

  # !--------------------------------------------------
  # ! Calculates integral of spectral function equation 25 in McCarthy et al. 2011
  # !--------------------------------------------------
  
  if tau_n > 1.e-11:
    fac=0.39-0.28/(1.+2.6*tau_n**0.1)
    spec1=0.32*tau_n**fac
  else:
    spec1=1853.*np.sqrt(tau_n)

  return spec1
        
def spec2(fp):

# !--------------------------------------------------
# ! Calculates integral of spectral function equation 26 in McCarthy et al. 2011
# !--------------------------------------------------

  a=np.array([0.55097,0.054332,-0.0023615,-5.7175e-5,9.9473e-6,-3.4761e-7,3.9461e-9])
  spec2=1.
  if fp <= 1.e+13:
    x=np.log(fp)
    spec2=0.
    for i in range(0,7):
      spec2=spec2+a[i]*x**i
      
  return spec2
  
def normcompliance_modifiedAndrade(tau_n, T_h):

    b1 = 0.16
    b2 = 1.75
    b3 = -0.558
    b4 = 6.975
    #b1 = 0.264
    #b2 = 0.044
    #b3 = -0.115
    #b4 = -2.845

    alpha = b1 + (b2*(1 - T_h))
    A = 10**(b3 + (b4*(1 - T_h)))

    J1_norm = 1 + (A*(np.pi/2))*(tau_n**alpha)*((np.sin(alpha*(np.pi/2)))**(-1))
    J2_norm = (A*(np.pi/2))*(tau_n**alpha)*((np.cos(alpha*(np.pi/2)))**(-1)) + tau_n
    
    return J1_norm, J2_norm

def normcompliance_Mc11(tau_n):

    xp=spec1(tau_n)
    yp=spec2(1./(tau_n*2.*np.pi))

    J1_norm=1./yp
    J2_norm=(np.pi*xp)/2.+tau_n

    return J1_norm, J2_norm

def normcompliance_YT16(tau_n, T_h):

    A_b = 0.664
    alpha = 0.38
    tau_p = 6e-05
    beta = 0
    delta_poro = 0
    gamma = 5
    T_visc = 0.94

    if T_h < 0.91:
        A_p = 0.01
    elif T_h < 0.96:
        A_p = 0.01 + (0.4 * (T_h - 0.91))
    elif T_h < 1:
        A_p = 0.03
    else:
        A_p = 0.03 + beta

    if T_h < 0.92:
        sigma_p = 4
    elif T_h < 1:
        sigma_p = 4 + (37.5 * (T_h - 0.92))
    else:
        sigma_p = 7

    J1_norm = 1 + ((A_b*(tau_n**alpha))/alpha) + (((np.sqrt(2*np.pi))/2)*A_p*sigma_p*(1-sp.special.erf((np.log(tau_p/tau_n))/(np.sqrt(2)*sigma_p))))
    J2_norm = (np.pi/2)*((A_b*(tau_n**alpha))+(A_p*np.exp(-((np.log(tau_p/tau_n))**2)/(2*(sigma_p**2))))) + tau_n

    return J1_norm, J2_norm        

def peak(Ap, sigmap, taup, tau):
    
    # peak = Ap*np.exp(-(np.log(tau/taup)**2)/(2*sigmap**2))
    peak = sp.special.gamma
    
    return peak

    
def plot_resall_fit(master, GMaxw, df, df_fit, N_opt):
    

    fig, ax1 = plt.subplots(figsize=(4,0.75*4))
    color = iter(cm.rainbow(np.linspace(0, 1, 11)))
    c=next(color)
    ax1.semilogx(df['t'],master[0,:], label=['G (filter)'], color=c, ls='', marker='o', markersize=3)
    ax1.semilogx(df_fit['t'],GMaxw[0,:], label=['Prony fit'], ls='-', lw=2, color=c)
    
    x=2
    for Th in np.linspace(0.92,1.1,10):
        c=next(color)
        ax1.semilogx(df['t'],master[x,:], color=c, ls='', marker='o', markersize=3)
        ax1.semilogx(df_fit['t'], GMaxw[x,:], ls='-', lw=2, color=c)
        y=peak(0.03,5.5,6e-5,df_fit['t'])
        
        fits=sp.stats.norm.fit(master[x,:])
        print (fits)
        ax1.semilogx(df_fit['t'],-sp.stats.norm.pdf(df_fit['t'],fits[0],fits[1]), ls='', marker='x', markersize=2, color=c)
        # ax1.semilogx(df_fit['t'], y, ls='', marker='x', markersize=2, color=c)
        # I think problem here is that data is not in equal time increments (all in log increments)
        print (x,np.shape(master))
        x=x+2

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Relaxation modulus delta') 
    ax1.legend()
    plt.savefig("relaxation_modulus_difference.jpg", dpi=300)   
    
   
# Set homologous temperatures array 
Tsol=1326.
Th_input=np.linspace(0.9,1.1,21)
T_input=Th_input*Tsol    

# Make tau array   
nx=201
mintau=-12
maxtau=3
tau_input=np.logspace(mintau,maxtau,nx)
prony_residual=np.zeros((len(Th_input),len(tau_input)))
prony_residual_fit=np.zeros((len(Th_input), ((maxtau-mintau)-1)*10))
N_opt=8

for i in range(len(Th_input)):
    

    f_plot = os.path.join(".", "plot_output_"+repr(round(Th_input[i],2)))

    #Load user master curve in frequency domain
    with open(os.path.join(f_plot, f"data_output.pkl"), 'rb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        df_original=pickle.load(f)
    with open(os.path.join(f_plot, f"fit_output.pkl"), 'rb') as f:
        df_fit=pickle.load(f)
    
    
    prony_residual[i,:]=df_original['G_relax_delta']
    prony_residual_fit[i,:]=df_fit['G_relax_delta']
    

plot_resall_fit(prony_residual,prony_residual_fit,df_original,df_fit,N_opt) # N=8 seems to work

