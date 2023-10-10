import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy as sp
from os import sys
import pyvisco as visco
from pyvisco import styles
styles.format_fig()


Tlo=800
T_input=np.array([Tlo])

### 

f_data = os.path.join(".", "data")
# f_plot = os.path.join(".", "plot_output_"+repr(Tlo))
# f_plot = os.path.join(".", "plot_output_Mc11")
f_plot = os.path.join(".", "plot_output_Modified_Andrade_"+repr(Tlo))

isExist = os.path.exists(f_plot)
if not isExist:
    os.mkdir(f_plot)
###

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

    
nx=201
ny=1
tau_input=np.logspace(-12,3,nx)
output_loT=np.zeros((nx*ny,2))
output_loTAn=np.zeros((nx*ny,2))
output_Mc11=np.zeros((nx*ny,2))
Ts=1326.
Th_input_loT=(T_input+273)/(Ts+273.)

k=0
for i in range(len(tau_input)):
    for j in range(len(Th_input_loT)):
        J1_YT16, J2_YT16 = normcompliance_YT16(tau_input[i], Th_input_loT[j])
        J1_Mc11, J2_Mc11 = normcompliance_Mc11(tau_input[i])
        J1_An, J2_An = normcompliance_modifiedAndrade(tau_input[i], Th_input_loT[j])
        output_loT[k,0]=tau_input[i]
        output_loT[k,1]=1./np.sqrt((J1_YT16**2.)+(J2_YT16**2.))
        output_loTAn[k,0]=tau_input[i]
        output_loTAn[k,1]=1./np.sqrt((J1_An**2.)+(J2_An**2.))
        output_Mc11[k,0]=tau_input[i]
        output_Mc11[k,1]=1./np.sqrt((J1_Mc11**2.)+(J2_Mc11**2.))
        k=k+1

T_input=np.array([1400])
output_hiT=np.zeros((nx*ny,2))
output_hiTAn=np.zeros((nx*ny,2))
Ts=1326.
Th_input_hiT=(T_input+273)/(Ts+273.)

k=0
for i in range(len(tau_input)):
    for j in range(len(Th_input_hiT)):
        J1_YT16, J2_YT16 = normcompliance_YT16(tau_input[i], Th_input_hiT[j])
        J1_An, J2_An = normcompliance_modifiedAndrade(tau_input[i], Th_input_hiT[j])
        output_hiT[k,0]=tau_input[i]
        output_hiT[k,1]=1./np.sqrt((J1_YT16**2.)+(J2_YT16**2.))
        output_hiTAn[k,0]=tau_input[i]
        output_hiTAn[k,1]=1./np.sqrt((J1_An**2.)+(J2_An**2.))
        k=k+1


header = '{0:^7s},{1:^7s}\n{2:^7s},{3:^7s}'.format('t', 'G_relax', 's', 'GPa')
np.savetxt('output_YT16.csv',output_loT, fmt=('%5.4e','%5.4e'), comments='', header=header,delimiter=",")
np.savetxt('output_Mc11.csv',output_Mc11, fmt=('%5.4e','%5.4e'), comments='', header=header,delimiter=",")
np.savetxt('output_Modified_Anrade.csv',output_loTAn, fmt=('%5.4e','%5.4e'), comments='', header=header,delimiter=",")
fig, ax = plt.subplots(1,1)
colour_plot1='k'
colour_plot2='r'
colour_plot3='b'
colour_plot4='g'
ax.plot(output_loT[:,0],output_loT[:,1],linestyle='-.',color=colour_plot1,label=f'YT16 Th={Th_input_loT[0]:.3f}')
ax.plot(output_loTAn[:,0],output_loTAn[:,1],linestyle='-.',color=colour_plot4,label=f'Modified_Andrade Th={Th_input_loT[0]:.3f}')
ax.plot(output_hiT[:,0],output_hiT[:,1],linestyle='-.',color=colour_plot2,label=f'YT16 Th={Th_input_hiT[0]:.3f}')
ax.plot(output_Mc11[:,0],output_Mc11[:,1],linestyle='-.',color=colour_plot3,label=f'Mc11')
ax.set_xscale('log', base=10)
ax.set_xlabel(r'Normalised timescale $\tau/\tau_M$')
ax.set_ylabel(r'Normalised modulus $J_U/\sqrt{J_1^2+J_2^2}$')
#ax.invert_xaxis()
ax.legend()
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(f_plot, f"YT16_modulus.jpg"), dpi=300)
plt.close()

#Load user master curve in frequency domain
# data = visco.load.file('output_YT16.csv')
# data = visco.load.file('output_Mc11.csv')
data = visco.load.file('output_Modified_Anrade.csv')
RefT=float(T_input)
domain = 'time'
modul = 'G'
df_master, units = visco.load.user_master(data, domain, RefT, modul)

#Smooth
win=1
df_master = visco.master.smooth(df_master, win=1)
# fig_smooth = visco.master.plot_smooth(df_master, units)
def plot_smooth(df_master, units):
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    stor_filt = '{}_stor_filt'.format(modul)
    loss_filt = '{}_loss_filt'.format(modul)
    relax_filt = '{}_relax_filt'.format(modul)
    if df_master.domain == 'freq':
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        df_master.plot(x='f', y=[stor], label=["{}'(raw)".format(modul)], 
            ax=ax1, logx=True, logy=False, color=['C0'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=[stor_filt], label=["{}'(filter)".format(modul)], 
            ax=ax1, logx=True, logy=False, color=['C0'])
        df_master.plot(x='f', y=[loss], label=["{}''(raw)".format(modul)], 
            ax=ax2, logx=True, logy=False, color=['C1'], marker='o', ls='', alpha=0.5)
        df_master.plot(x='f', y=[loss_filt], label=["{}''(filter)".format(modul)], 
            ax=ax2, logx=True, logy=False, color=['C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus ({})'.format(units[stor]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus ({})'.format(units[stor])) 
        ax1.legend()
        ax2.legend()
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots(figsize=(4,0.75*4))
        df_master.plot(x='t', y=[relax], label = [relax], 
            ax=ax1, logx=True, logy=False, ls='', marker='o', color=['gray'])
        df_master.plot(x='t', y=[relax_filt], label=['filter'], 
            ax=ax1, logx=True, logy=False, color=['r'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax])) 
        ax1.legend()
    plt.savefig(os.path.join(f_plot, f"smooth_modulus.jpg"), dpi=300)
plot_smooth(df_master, units)

#Discretize number of Prony terms
df_dis = visco.prony.discretize(df_master,window='min')
# fig_dis = visco.prony.plot_dis(df_master, df_dis, units)
def plot_dis(df_master, df_dis, units):
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    if df_master.domain == 'freq':
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        df_master.plot(x='f', y=[stor], label=["{}'(filter)".format(modul)],
            ax=ax1, logx=True, logy=False, color=['C0'], alpha=0.5)
        df_master.plot(x='f', y=[loss], label=["{}''(filter)".format(modul)],
            ax=ax2, logx=True, logy=False, color=['C1'], alpha=0.5)
        df_dis.plot(x='f', y=[stor], label=['tau_i'], ax=ax1, 
            logx=True, logy=False, ls='', marker='o', color=['C0'])
        df_dis.plot(x='f', y=[loss], label=['tau_i'], ax=ax2, 
            logx=True, logy=False, ls='', marker='o', color=['C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus ({})'.format(units[stor]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus ({})'.format(units[stor])) 
        ax1.legend()
        ax2.legend()
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots(figsize=(4,0.75*4))
        df_master.plot(x='t', y=[relax], 
            ax=ax1, logx=True, logy=False, color=['k'])
        df_dis.plot(x='t', y=[relax], label = ['tau_i'], 
            ax=ax1, logx=True, logy=False, ls='', marker='o', color=['red'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax]))
        ax1.legend()
    plt.savefig(os.path.join(f_plot, f"discretised_modulus.jpg"), dpi=300)
plot_dis(df_master, df_dis, units)

#Fit Prony series parameter
prony, df_GMaxw = visco.prony.fit(df_dis, df_master, opt=True)
# fig_fit = visco.prony.plot_fit(df_master, df_GMaxw, units)
def plot_fit(df_master, df_GMaxw, units):
    modul = df_master.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    if df_master.domain == 'freq':
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
        df_master.plot(x='f', y=[stor], label=["{}'(filter)".format(modul)],
            ax=ax1, logx=True, logy=False, color=['C0'], 
            alpha=0.5, ls='', marker='o', markersize=3)
        df_master.plot(x='f', y=[loss], label=["{}''(filter)".format(modul)],
            ax=ax2, logx=True, logy=False, color=['C1'], 
            alpha=0.5, ls='', marker='o', markersize=3)
        df_GMaxw.plot(x='f', y=[stor], label=["Prony fit"],
            ax=ax1, logx=True, logy=False, ls='-', lw=2, color=['C0'])
        df_GMaxw.plot(x='f', y=[loss], label=["Prony fit"],
            ax=ax2, logx=True, logy=False, ls='-', lw=2, color=['C1'])
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Storage modulus ({})'.format(units[stor]))
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Loss modulus ({})'.format(units[stor])) 
        ax1.legend()
        ax2.legend()
    elif df_master.domain == 'time':
        fig, ax1 = plt.subplots(figsize=(4,0.75*4))
        df_master.plot(x='t', y=[relax], ax=ax1, logx=True, logy=False, 
            color=['gray'], ls='', marker='o', markersize=3,
            label=["{}(filter)".format(modul)])
        df_GMaxw.plot(x='t', y=[relax], ax=ax1, label=['Prony fit'],
            logx=True, logy=False, ls='-', lw=2, color=['r'])
        ax1.set_xlabel('Time ({})'.format(units['t']))
        ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax])) 
        ax1.legend()
    plt.savefig(os.path.join(f_plot, f"relaxation_modulus.jpg"), dpi=300)
plot_fit(df_master, df_GMaxw, units)

#Plot Generalized Maxwell model
# fig_GMaxw = visco.prony.plot_GMaxw(df_GMaxw, units)
def plot_GMaxw(df_GMaxw, units):
    modul = df_GMaxw.modul
    stor = '{}_stor'.format(modul)
    loss = '{}_loss'.format(modul)
    relax = '{}_relax'.format(modul)
    # if df_GMaxw.domain == 'freq':
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,0.75*4))
    df_GMaxw.plot(x='f', y=[stor], label=["{}'".format(modul)],
            ax=ax1, logx=True, logy=False, ls='-', lw=2, color=['C0'])
    df_GMaxw.plot(x='f', y=[loss], label=["{}''".format(modul)],
            ax=ax2, logx=True, logy=False, ls=':', lw=2, color=['C1'])
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Storage modulus ({})'.format(units[stor]))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Loss modulus ({})'.format(units[stor])) 
    ax1.legend()
    ax2.legend()
    plt.savefig(os.path.join(f_plot, f"GMX_fit_modulus_freq.jpg"), dpi=300)
    # elif df_GMaxw.domain == 'time':
    fig, ax1 = plt.subplots(figsize=(4,0.75*4))
    df_GMaxw.plot(x='t', y=[relax], 
            ax=ax1, logx=True, logy=False, ls='--', lw=2, color=['C2'],
            label=["{}(t)".format(modul)],)
    ax1.set_xlabel('Time ({})'.format(units['t']))
    ax1.set_ylabel('Relaxation modulus ({})'.format(units[relax])) 
    ax1.legend()
    plt.savefig(os.path.join(f_plot, f"GMX_fit_modulus_time.jpg"), dpi=300)
plot_GMaxw(df_GMaxw, units)

pronyout,N_opt,err= visco.opt.nprony(df_master, prony, opt = 1.5)
def plot_optfit(df_master, dict_prony, N, units):
    for i in range(3,N_opt+1):
        df_GMaxw = visco.prony.calc_GMaxw(**dict_prony[i])
        fig = plot_fit(df_master, df_GMaxw, units)
        plt.savefig(os.path.join(f_plot, f"GMX_optimisation_fit_"+repr(i)+".jpg"), dpi=300)
plot_optfit(df_master, pronyout,N_opt,units)


def plot_optresidual(err):
    m = err.modul
    fig, ax = plt.subplots()
    err.plot(y=['res'], ax=ax, c='k', label=['Least squares residual'], 
        marker='o', ls='--', markersize=4, lw=1)
    ax.set_xlabel('Number of Prony terms')
    ax.set_ylabel(r'$R^2 = \sum \left[{{{0}}}_{{meas}} - {{{0}}}_{{Prony}} \right]^2$'.format(m)) 
    ax.set_xlim(0,)
    ax.set_ylim(-0.01, max(2*err['res'].min(), 0.75))
    ax.legend()
    plt.savefig(os.path.join(f_plot, f"GMX_optimisation_residual.jpg"), dpi=300)
plot_optresidual(err)
    

