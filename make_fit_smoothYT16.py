import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

### 

f_data = os.path.join("..", "data")
f_plot = os.path.join("..", "plot_output")

###

def relaxation_YT16(T_h,tau_n):

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

    X_b = A_b * (tau_n ** alpha)
    X_p = A_p * np.exp(-((np.log(tau_n / tau_p))**2 / (2 * (sigma_p ** 2))))

    X = X_b + X_p
    
    return X

def calc_tansh_continuousstep(x,x0,ymin,ymax,k):

    scale = ymax-ymin
    y = scale*((1 + np.exp(-2*k*(x-x0)))**(-1)) + ymin

    return y

def relaxation_YT16_smooth(T_h,tau_n,Ap_x0,Ap_ymin,Ap_k,sigmap_x0,sigmap_k):

    A_b = 0.664
    alpha = 0.38
    tau_p = 6e-05
    beta = 0
    delta_poro = 0
    gamma = 5
    T_visc = 0.94

    A_p = calc_tansh_continuousstep(T_h,Ap_x0,Ap_ymin,0.03,Ap_k)
    sigma_p = calc_tansh_continuousstep(T_h,sigmap_x0,4.0,7.0,sigmap_k)

    X_b = A_b * (tau_n ** alpha)
    X_p = A_p * np.exp(-((np.log(tau_n / tau_p))**2 / (2 * (sigma_p ** 2))))

    X = X_b + X_p
    
    return X

def viscosity_YT16(Tc,Th,c,m):

    Tk = Tc + 273.15
    Teta = 0.94
    gamma = 5
    delphi = 0

    if Th < Teta:
        A_eta=1.
    elif Th >= Teta and Th<1.:
        A_eta=np.exp((-1.*((Th-Teta)/(Th-(Th*Teta))))*np.log(gamma))
    else:
        A_eta=(1./gamma)*np.exp(-delphi)

    log10_visc = m*(1000/Tk) + c + np.log10(A_eta)

    return log10_visc

def viscosity_YT16_smooth(Tc,Th,c,m,Aeta_x0,Aeta_k):

    Tk = Tc + 273.15

    Aeta_ymin = 1
    Aeta_ymax = 0.2

    A_eta = calc_tansh_continuousstep(Th,Aeta_x0,Aeta_ymin,Aeta_ymax,Aeta_k)

    log10_visc = m*(1000/Tk) + c + np.log10(A_eta)

    return log10_visc

### 

def import_relaxation_data():

    arr_sum = np.loadtxt(os.path.join(f_data,"#41_Xpfit.txt"),delimiter=',')
    arr_Th = arr_sum[:,2]
    arr_Tref = arr_sum[:,0]
    n_data = len(arr_Tref)
    l_data = 0

    for i in range(n_data):
        Tref = str(int(arr_Tref[i]))
        data_in = np.loadtxt(os.path.join(f_data,f"#41_T{Tref}.txt"),dtype='f',delimiter=',')
        l_data += len(data_in)

    data = np.zeros((3,l_data))
    j = 0

    for i in range(n_data):

        Tref = str(int(arr_Tref[i]))
        Th = arr_Th[i]
        data_in = np.loadtxt(os.path.join(f_data,f"#41_T{Tref}.txt"),dtype='f',delimiter=',')
        data_tau = data_in[:,3]
        data_X = data_in[:,4]
        data_Th = np.full(np.shape(data_X),Th)
        l_subset = len(data_tau)
        data[0,j:j+l_subset] = data_Th
        data[1,j:j+l_subset] = data_tau
        data[2,j:j+l_subset] = data_X
        j += l_subset

    return data

def import_viscosity_data():

    arr_sum = np.loadtxt(os.path.join(f_data,"#41_Xpfit.txt"),delimiter=',')
    idx_sort = np.argsort(arr_sum[:,2])
    arr_sum = arr_sum[idx_sort,:]
    T = arr_sum[:,1]
    Th = arr_sum[:,2]
    visc = arr_sum[:,3]
    data = arr_sum[:,[1,2,3]]
    data[:,2] = np.log10(data[:,2])

    return data.T

###

def eval_misfit(ydata,yfit):

    return np.sum((ydata - yfit)**2)

def fit_func_relaxation(var,Ap_x0,Ap_ymin,Ap_k,sigmap_x0,sigmap_k):

    Th,tau = var

    Ap_ymax = 0.03
    sigmap_ymin = 4.0
    sigmap_ymax = 7.0

    A_b = 0.664
    alpha = 0.38
    tau_p = 6e-05

    A_p = calc_tansh_continuousstep(Th,Ap_x0,Ap_ymin,Ap_ymax,Ap_k)
    sigma_p = calc_tansh_continuousstep(Th,sigmap_x0,sigmap_ymin,sigmap_ymax,sigmap_k)

    #X_b = A_b * (tau**alpha)
    #X_p = A_p * np.exp(-((np.log(tau/tau_p))**2/(2*(sigma_p**2))))

    X_b = A_b * ((10**tau)**alpha)
    X_p = A_p * np.exp(-((np.log((10**tau)/tau_p))**2/(2*(sigma_p**2))))

    X = X_b + X_p
    
    return np.log10(X)

def fit_func_viscosity(var,c,m,Aeta_x0,Aeta_k):

    Tc,Th = var
    Tk = Tc + 273.15

    Aeta_ymin = 1
    Aeta_ymax = 0.2

    A_eta = calc_tansh_continuousstep(Th,Aeta_x0,Aeta_ymin,Aeta_ymax,Aeta_k)

    log10_visc = m*(1000/Tk) + c + np.log10(A_eta)  # superposition of Arrhenius behaviour and add'l near-solidus viscosity reduction

    return log10_visc

###

def calc_relaxation_misfit(data,popt):

    arr_Th = data[0]
    unique_Th = np.unique(arr_Th)
    n_Th = len(unique_Th)

    misfit_smooth = np.zeros(n_Th)
    misfit_YT16 = np.zeros(n_Th)

    for i in range(n_Th):

        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        X = tmp_data[2]
        fit_X = relaxation_YT16_smooth(Th,tau,popt[0],popt[1],popt[2],popt[3],popt[4])
        fit_X_YT16 = relaxation_YT16(Th,tau)

        misfit_smooth[i] = eval_misfit(X,fit_X)
        misfit_YT16[i] = eval_misfit(X,fit_X_YT16)

    print(np.sum(misfit_YT16))
    print(np.sum(misfit_smooth))

def calc_viscosity_misfit(data,popt):

    arr_Th = data[1]
    arr_T = data[0]
    arr_visc = data[2]
    unique_Th = np.unique(arr_Th)
    n_Th = len(unique_Th)

    misfit_YT16 = np.zeros(n_Th)
    misfit_smooth = np.zeros(n_Th)

    for i in range(n_Th):

        T = arr_T[i]
        Th = arr_Th[i]
        visc = arr_visc[i]
        fit_visc = viscosity_YT16_smooth(T,Th,popt[0],popt[1],popt[2],popt[3])
        fit_visc_YT16 = viscosity_YT16(T,Th,popt[0],popt[1])

        misfit_smooth[i] = eval_misfit(visc,fit_visc)
        misfit_YT16[i] = eval_misfit(visc,fit_visc_YT16)

    print(np.sum(misfit_YT16))
    print(np.sum(misfit_smooth))

def plot_relaxation_fit(data,popt):

    arr_Th = data[0]
    unique_Th = np.unique(arr_Th)
    n_Th = len(unique_Th)

    fig,ax=plt.subplots(3,1,figsize=(8,10))
    colors=['black','red','green','blue','cyan','magenta','pink','orange']

    for i in range(0,n_Th):

        plot_color = colors[i]
        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        tau_theory = 10**np.linspace(-9,1,1000)
        X = tmp_data[2]
        fit_X = relaxation_YT16_smooth(Th,tau_theory,popt[0],popt[1],popt[2],popt[3],popt[4])
        fit_X_YT16 = relaxation_YT16(Th,tau_theory)
        ax[0].plot(tau_theory,fit_X,color=plot_color,linestyle='-',alpha=0.75)
        #ax[0].plot(tau_theory,fit_X_YT16,color=plot_color,linestyle='--',alpha=0.75)

    for i in range(0,n_Th):

        plot_color = colors[i]
        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        tau_theory = 10**np.linspace(-9,1,1000)
        X = tmp_data[2]
        ax[0].scatter(tau,X,marker='+',s=60,color=plot_color,label=f'Th={Th:.3f}')

    ax[0].set_xlabel(r"$\tau/tau_M$")
    ax[0].set_ylabel(r"$X$")
    ax[0].set_xscale('log', base=10)
    ax[0].set_yscale('log', base=10)
    ax[0].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
    ax[0].set_ylabel(r'$X$ (Continuous Param.)')
    ax[0].invert_xaxis()
    ax[0].legend()

    for i in range(0,n_Th):

        plot_color = colors[i]
        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        tau_theory = 10**np.linspace(-9,1,1000)
        X = tmp_data[2]
        fit_X = relaxation_YT16_smooth(Th,tau_theory,popt[0],popt[1],popt[2],popt[3],popt[4])
        fit_X_YT16 = relaxation_YT16(Th,tau_theory)
        #ax[1].plot(tau_theory,fit_X,color=plot_color,linestyle='-',alpha=0.75)
        ax[1].plot(tau_theory,fit_X_YT16,color=plot_color,linestyle='--',alpha=0.75)

    for i in range(0,n_Th):

        plot_color = colors[i]
        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        tau_theory = 10**np.linspace(-9,1,1000)
        X = tmp_data[2]
        ax[1].scatter(tau,X,marker='+',s=60,color=plot_color,label=f'Th={Th:.3f}')

    ax[1].set_xlabel(r"$\tau/tau_M$")
    ax[1].set_ylabel(r"$X$")
    ax[1].set_xscale('log', base=10)
    ax[1].set_yscale('log', base=10)
    ax[1].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
    ax[1].set_ylabel(r'$X$ (YT16)')
    ax[1].invert_xaxis()
    ax[1].legend()
    
    for i in range(0,n_Th):

        plot_color = colors[i]
        tmp_data = data[:,np.where(data[0]==unique_Th[i])[0]]
        Th = tmp_data[0,0]
        tau = tmp_data[1]
        tau_theory = 10**np.linspace(-9,1,1000)
        X = tmp_data[2]
        fit_X = relaxation_YT16_smooth(Th,tau_theory,popt[0],popt[1],popt[2],popt[3],popt[4])
        fit_X_YT16 = relaxation_YT16(Th,tau_theory)
        ax[2].plot(tau_theory,fit_X,color=plot_color,linestyle='-',alpha=0.75)
        ax[2].plot(tau_theory,fit_X_YT16,color=plot_color,linestyle='--',alpha=0.75)

    ax[2].set_xlabel(r"$\tau/tau_M$")
    ax[2].set_ylabel(r"$X$")
    ax[2].set_xscale('log', base=10)
    ax[2].set_yscale('log', base=10)
    ax[2].set_xlabel(r'Normalised timescale $\tau/\tau_M$')
    ax[2].set_ylabel(r'$X$')
    ax[2].invert_xaxis()
    ax[2].legend()

    plt.tight_layout()
    plt.savefig("../plot_output/YT16_smooth_relaxation_fit.jpg",dpi=600)
    plt.close()

def plot_viscosity_fit(data,popt):

    fig,ax=plt.subplots(1,1,figsize=(6,6))
    T = data[0]
    Tk = T + 273.15
    Th = data[1]
    visc = data[2]
    fit_invTk = np.linspace(np.min(1000/Tk),np.max(1000/Tk),1000)
    fit_Tk = 1000 / fit_invTk
    fit_Th = fit_Tk / 316.16
    fit_T = fit_Tk - 273.15
    fit_visc = viscosity_YT16_smooth(fit_T,fit_Th,popt[0],popt[1],popt[2],popt[3])
    fit_visc_YT16 = np.zeros(len(fit_visc))
    for i in range(len(fit_T)):
        fit_visc_YT16[i] = viscosity_YT16(fit_T[i],fit_Th[i],popt[0],popt[1])
    ax.scatter(1000/Tk,visc,s=50,color='k')
    ax.plot(fit_invTk,fit_visc,linestyle='-',color='k',label='cont. approx.')
    ax.plot(fit_invTk,fit_visc_YT16,linestyle='-',color='r',label='YT16')
    ax.set_xlabel(r"$1000/T$")
    ax.set_ylabel(r"log$_{10}(\eta$[Pa s]$)$")
    ax.axvline(1000/(316.16),color='k',linestyle='--',label=r'$T_S$')
    plt.legend()
    plt.tight_layout()
    plt.savefig("../plot_output/YT16_smooth_visc_fit.jpg",dpi=600)
    plt.close()

###

def fit_relaxation_params():

    data = import_relaxation_data()
    data_fit = data.copy()
    data_fit[1] = np.log10(data_fit[1])
    data_fit[2] = np.log10(data_fit[2])
    popt,pcov = sp.optimize.curve_fit(fit_func_relaxation,xdata=data_fit[[0,1],:],ydata=data_fit[2,:],p0=[0.935,0.01,100,0.96,60],bounds=([0.91,0,5,0.93,5],[1.0,0.02,200,1.0,200]))

    return data, popt, pcov

def fit_viscosity_params():

    data = import_viscosity_data()
    popt,pcov = sp.optimize.curve_fit(fit_func_viscosity,xdata=data[[0,1],:],ydata=data[2,:],p0=[-16,9,0.94,100],bounds=([-20,5,0.90,5],[-10,10,1.0,200]))

    return data, popt, pcov

data, popt, pcov = fit_relaxation_params()
calc_relaxation_misfit(data,popt)
plot_relaxation_fit(data,popt)

data, popt, pcov = fit_viscosity_params()
calc_viscosity_misfit(data,popt)
plot_viscosity_fit(data,popt)