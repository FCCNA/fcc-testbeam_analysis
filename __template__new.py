from scipy import signal
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

ampl6x6_28=3.621
ampl6x6_18=1.163

ampl3x3_28=1.236
ampl3x3_18=0.401
ampl3x3_filter=0.0425



def highpass(x, dt, fc):
    RC=1/(2*np.pi*fc)
    n = len(x)
    y = np.zeros(n)
    alpha = RC / (RC + dt)   
    y[0] = x[0]
    for i in range(1, n):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y

def db_exp_hp(x, a, tf, tr, x0, c, dt, fc):
    y = a*np.exp((-x+x0)/tf)*(1-np.exp((-x+x0)/tr))*(x>x0)+c
    y = highpass(y, dt, fc)
    return y

def template_laser_calib(x, run):
    times = np.load(f'npys/Times_{run}.npy')
    laser_wf = np.load(f'npys/Mean_Waveform_{run}.npy')
    return np.interp(x, times[:-10], laser_wf[:-10])
    
    
def spr_6x6(x, x0, ampl, dt):
    #a = 3.95
    tf = 150
    tr = 1
    c = 0
    fc = 4.3E-5
    #dt = 0.32
    return db_exp_hp(x, ampl, tf, tr, x0, c, dt, fc)

def spr_3x3(x, x0, ampl, dt):
    #a = 3.95
    tf = 48
    tr = 1
    c = 0
    fc = 8E-6
    #dt = 0.32
    return db_exp_hp(x, ampl, tf, tr, x0, c, dt, fc)


def time_dits(nevS, nevC, crystal):
    if crystal=='BGO':
        t1=50
        a1=0.044
        t2=320
        a2=0.093
        tot = a1*t1+a2*t2
        frac1 = a1*t1/tot
        frac2 = a2*t2/tot
    if crystal=='BSO':
        t1=22
        a1=0.054
        t2=98
        a2=0.087
        tot = a1*t1+a2*t2
        frac1 = a1*t1/tot
        frac2 = a2*t2/tot
    if crystal=='PWO':
        t1=7.5
        a1=1
        t2=0
        a2=0
        tot = a1*t1+a2*t2
        frac1 = a1*t1/tot
        frac2 = a2*t2/tot
    t0 = 100
    EVS = np.concatenate([np.random.exponential(t1, int(nevS*frac1)), np.random.exponential(t2, int(nevS*frac2))]) - t0
    EVC = np.random.exponential(0.1, int(nevC)) - t0
    EVT = np.concatenate([EVS, EVC])
    return EVS, EVC, EVT

def histogrammer(EVS, EVC, EVT, ranger, nbins):
    hC, bins = np.histogram(EVC, range=ranger, bins=nbins)
    hS, bins = np.histogram(EVS, range=ranger, bins=nbins)
    hT, bins = np.histogram(EVT, range=ranger, bins=nbins)
    return bins, hS, hC, hT

def get_templates(crystal, SiPM, SPR_ampl, ranger, dt, nsimS=1E7, nsimC=1E7, normalize=True, graphics=True, Laser_Calib = False, run = 477):
    
    x = np.arange(0, 10000)*dt
    
    if not Laser_Calib:
        if SiPM=='3x3':
            SPR = spr_3x3(x, .1, SPR_ampl, dt)
        if SiPM=='6x6':
            SPR = spr_6x6(x, .1, SPR_ampl, dt)
    else: 
        SPR = template_laser_calib(x, run)
        SPR = SPR_ampl/np.max(SPR) * SPR
            
            
    EVS, EVC, EVT = time_dits(nsimS, nsimC, crystal)
    bins, hS, hC, hT = histogrammer(EVS, EVC, EVT, ranger, int((ranger[1]-ranger[0])/dt))
    sigS = signal.convolve(SPR, hS)[:len(bins)]
    sigC = signal.convolve(SPR, hC)[:len(bins)]
    if normalize:
        sigS = (sigS)/nsimS
        sigC = (sigC)/nsimC
    interS = interp1d(bins, sigS, kind='linear', fill_value="extrapolate")
    interC = interp1d(bins, sigC, kind='linear', fill_value="extrapolate")
    if graphics:
        plt.plot(x, SPR)
        plt.grid()
        plt.title('Single Photon Response')
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [mV]')
        plt.show()
        plt.plot(bins, interS(bins), label='Scintillation')
        plt.plot(bins, interC(bins), label='Cherenkov')
        #plt.plot(bins, myfunc(bins, 1, 1, 100, 0.5), label='Total')
        plt.grid()
        plt.legend()
        plt.title('Normalized signal templates '+crystal)
        plt.xlabel('Time [ns]')
        plt.ylabel('Amplitude [mV]')
        plt.show()
    return interS, interC, bins

def wf_function(x, interS, interC, c, s, t0, of):
    time = x-t0
    return interC(time)*c+interS(time)*s+of
    