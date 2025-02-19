from __init__ import * 
import __template__new as mcj
import numpy as np
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
          'font.size': 20,
          'figure.figsize': (16, 9),
          'axes.grid': False,
          'grid.linestyle': '-',
          'grid.alpha': 0.2,
          'lines.markersize': 5.0,
          'xtick.minor.visible': True,
          'xtick.direction': 'in',
          'xtick.major.size': 10.0,
          'xtick.minor.size': 5.0,
          'xtick.top': True,
          'ytick.minor.visible': True,
          'ytick.direction': 'in',
          'ytick.major.size': 10.0,
          'ytick.minor.size': 5.0,
          'ytick.right': True,
          'errorbar.capsize': 0.0,
          'figure.max_open_warning': 50,
})
from scipy.optimize import curve_fit
import yaml
from iminuit import Minuit
from iminuit.cost import LeastSquares
import importlib



with open('yaml/data.yaml', 'r') as file:
    runs_info = yaml.safe_load(file)
    
make_dir('Hardware_Fit')
make_dir('Hardware_Fit/parqs')
make_dir('Hardware_Fit/figures')
make_dir('npys')
make_dir('parqs')


def argparser():
    """
    Parse options as command-line arguments.
    """

    # Re-use the base argument parser, and add options on top.

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter, add_help=True)

    
    parser.add_argument("--crystal",
                        type=str,
                        required=True,
                        help="Put the Crystal Type")

    parser.add_argument("--beam",
                        type=str,
                        required=True,
                        help="Put the Particle of the beam")

    parser.add_argument("--angle",
                        type=int,
                        required=True,
                        help="Put the angle")
    '''
    parser.add_argument("--channel",
                        type=int,
                        required=True,
                        help="Put the channel")
    '''

    parser.add_argument("--Laser",
                        action = 'store_true',
                        help="Add if you want template from laser")


    #tools.check_options_validity(parser)

    return parser

args = argparser().parse_args()
crystal = args.crystal
beam = args.beam
angle = args.angle
#ch = args.channel
laser = args.Laser

beam_text = r'$e^+$' if beam == 'e' else r'$\mu^+$'

print(f'Running script for {crystal} with {beam} at {angle} degrees')
if not laser:
    print('Template from analytic function')
else:
    print('Template from Laser Calibration Waveform')

mean =  {}
df = {}

print(f'Reading Files')
wf, df = mixing_run(runs_info[crystal][beam][angle])
times = wf[0]['times']


with open('yaml/laser_calib.yaml', 'r') as file:
    log_laser = yaml.safe_load(file)

for ch in [1,2]:
    df[f'amplitude_media_channel{ch}'] = df[f'amplitude_media_channel{ch}'].fillna(0)
    if ch == 2:
        sipm = '6x6' 
        ampl = 18
        
    if ch == 1:
        sipm = '3x3' 
        ampl = 0 if crystal == 'BGO' else 18
        
    if sipm == '3x3':
        if ampl == 0:
            SPR = mcj.ampl3x3_filter 
        elif ampl == 18:
            SPR = mcj.ampl3x3_18
        elif ampl == 28:
            SPR = mcj.ampl3x3_28
            
    if sipm == '6x6':
        if ampl == 18:
            SPR = mcj.ampl6x6_18
        if ampl == 28:
            SPR = mcj.ampl6x6_28
        
    if laser:
        path_data = '/eos/user/m/mcampajo/MAXICC_TB_analysis/data_calibrazioni/laser/'
        create_mean_wf(log_laser[sipm][ampl], path_data)

    #m = np.percentile(df[f'amplitude_media_channel{ch}'], 30)
    ev_ids = df['__event__']
    #print(len(ev_ids.tolist()))

    import warnings

    # Ignora i RuntimeWarnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def myfunc(x, c, s, t0, of):
        return mcj.wf_function(x, interS_template, interC_template, c, s, t0, of)

    dt = times[1]-times[0]
    ranger = [int(times.min()), int(times.max())]
    npoints = len(times)
    interS_template, interC_template, bins = mcj.get_templates(crystal,
                                                            sipm,
                                                            SPR,
                                                            ranger,
                                                            dt,
                                                            nsimS=1E7,
                                                            nsimC=1E7,
                                                            normalize=True,
                                                            graphics=False,
                                                            Laser_Calib = laser, 
                                                            run = log_laser[sipm][ampl])
    output = []

    print(f'Fitting Channel {ch}...')

    for ev_id in tqdm(ev_ids.to_list()):
        
        WFT = wf[ev_id][f'{ch}media']

        tmp_df = {'__event__': ev_id,
                'amplitude': np.max(WFT)}

        sigmas = np.std(WFT[:100])
        tmp_df.update(
            {
                'sigma_100' : sigmas,
                'Q_data' : np.sum(WFT)
            }
        )
        if len(times) != len(WFT):
            continue
        least_squares = LeastSquares(times, WFT, sigmas, myfunc)
        m = Minuit(least_squares, 100, 100, 0, 0)
        m.fixed['of'] = True
        m.migrad() 

        # fit_info = [f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {m.ndof:.0f} = {m.fval/m.ndof:.2f}" ]

        t0_fit = m.values['t0']

        try:
            # Create mask for times less than (t0 - 100)
            mask = times < (m.values['t0'] - 100)
            # Extract relevant data
            relevant_data = WFT[mask]
            
            # Check if there are any values selected
            if relevant_data.size > 0:
                ampl_sideband = np.max(relevant_data)
            else:
                ampl_sideband = 0
        except (IndexError, ValueError, TypeError) as e:
            # Handle specific exceptions
            ampl_sideband = 0

        tmp_df.update(
            {
                'chi2' : m.fval,
                'ndof' : m.ndof,
                'Q_fit': np.sum(myfunc(times, *m.values)),
                'ampl_sideband': ampl_sideband
            }
        )
        
        tmp_df.update({
            p: v for p, v in zip(m.parameters, m.values)
        })
        tmp_df.update({
            f'err{p}': e for p, e in zip(m.parameters, m.errors)
        })

        output.append(tmp_df)

    info = pd.DataFrame(output)

    name_piece = 'Hardware' if not laser else 'fromLaser'
    info.to_parquet(f'Hardware_Fit/parqs/FitInfo_{name_piece}_{crystal}_{beam}_{angle}_Ch{ch}.parq')