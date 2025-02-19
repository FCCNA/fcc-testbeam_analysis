# Adding Lyso to Fit DataFrame
import numpy as np
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

import argparse
from __init__ import * 

import yaml

from PIL import Image

with open('yaml/data.yaml', 'r') as file:
    runs_info = yaml.safe_load(file)
    
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

    #parser.add_argument("--beam",
    #                    type=str,
    #                    required=True,
    #                    help="Put the Particle of the beam")


    #tools.check_options_validity(parser)

    return parser



args = argparser().parse_args()
crystal = args.crystal
beam = 'e'
beam_text = r'$e^+$' if beam == 'e' else r'$\mu^+$'

angles = np.array(list(runs_info[crystal][beam].keys()))

for angle in angles:

    print(f'Running script for {crystal} with {beam} at {angle} degrees')

    mean =  {}
    df = {}

    print(f'Reading Files')
    wf, df = mixing_run(runs_info[crystal][beam][angle], addLyso=True)

    fit_df = pd.read_parquet(f'Hardware_Fit/parqs/FitInfo_fromLaser_{crystal}_{beam}_{angle}_BothChs.parq')

    output_df = pd.merge(fit_df, df[['__event__','passLyso']], on='__event__', how='left')
    
    output_df.to_parquet(f'Hardware_Fit/parqs/FitInfo_Lyso_fromLaser_{crystal}_{beam}_{angle}_BothChs.parq')
    
    



