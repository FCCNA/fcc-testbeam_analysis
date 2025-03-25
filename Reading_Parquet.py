import argparse
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
from cycler import cycler  # Permette di impostare cicli di colori

# Ottieni i colori dalla palette Pastel1
colorslist = plt.get_cmap("Set2").colors
color = {'BGO': colorslist[0], 'BSO': colorslist[1]}
conversion_factor = {'BSO': 2.08e+3, 'BGO': 9.0e+3}
crystals = ['BGO', 'BSO']

# Imposta lo stile dei colori per tutti i plot
plt.rcParams["axes.prop_cycle"] = cycler(color=colorslist)
from __init__ import *
import yaml

make_dir('GoodFigures')

with open('variables.yaml', 'r') as file:
    config = yaml.safe_load(file)

dict_plot = config['dict_plot']
variables_list = config['variables_list']

with open('yaml/data.yaml', 'r') as file:
    runs_info = yaml.safe_load(file)
beam = 'e'
beam_text = r'$e^+$' if beam == 'e' else r'$\mu^+$'
angles = np.array(list(runs_info['BGO'][beam].keys()))


def argparser():
    """
    Parse options as command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter, add_help=True)

    available_vars = ', '.join(variables_list)  
    parser.add_argument(
        "--variable",
        type=str,
        choices=variables_list,  
        nargs='+',  
        required=True, 
        help=f"Specify one or more variables from the list:\n{available_vars}"
    )
    parser.add_argument(
        "--logY",
        action='store_true',
        #choices=variables_list,  
       # nargs='+',  
        #required=True, 
        help=f"If you want a LogY scale"
    )
    parser.add_argument(
        "--logX",
        action='store_true',
        #choices=variables_list,  
       # nargs='+',  
        #required=True, 
        help=f"If you want a LogX scale"
    )
    parser.add_argument(
        "--density",
        action='store_true',
        #choices=variables_list,  
       # nargs='+',  
        #required=True, 
        help=f"If you want density = True"
    )
    parser.add_argument(
        "--errors",
        action='store_true',
        #choices=variables_list,  
       # nargs='+',  
        #required=True, 
        help=f"If you want the Error Bars in the plot"
    )

    return parser

args = argparser().parse_args()


def make_all_distr(variable, logx = False, logy = False, density = False):
    n_angles = len(angles)
    n_rows = (n_angles + 1) // 3 
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 9 * n_rows)) 
    axes = axes.flatten()
    titti = dict_plot[variable]['title']
    lalla = dict_plot[variable]['label']
    #custom_plot_layout(title=titti, xlabel=lalla, ylabel="Events",  figsize=(16, 9), angle = angle, crystal = 'BGO - BSO', beam = beam_text)
    for i, angle in enumerate(angles):
        ax = axes[i]      
        for crystal in ['BGO', 'BSO']:
            array = d_filter[crystal][angle][variable]
            if variable == 'e':
                ranges = (0,7.5) if angle in [0, 180] else (0,3)
            else:
                ranges = dict_plot[variable]['range']
            ax.hist(array, range = ranges, bins=150, histtype='step', linewidth=2, label=crystal, density = density, color = color[crystal])
            ax.axvline(np.mean(array), label=f'{crystal} Mean', linewidth=2, color = color[crystal])
            ax.axvspan(np.mean(array) - np.std(array), np.mean(array) + np.std(array), alpha=0.25, color = color[crystal])
        ax.set_title(f'Angle = {angle}')
        
        if variable == 'e' and angle not in [110, 130]:
            ax.hist(pq_sim[angle]['Ecalo']/1000, range = ranges, bins=150, histtype='step', linewidth=2, label='MC', density = density)
            ax.hist(pq_sim[angle].query('Elyso>0')['Ecalo']/1000, range = ranges, bins=150, histtype='step', linewidth=2, label='MC Lyso', density = density)
            ax.hist(pq_sim_1e3[angle]['Ecalo']/1000, range = ranges, bins=150, histtype='step', linewidth=2, label='MC 1e3', density = density)
            ax.hist(pq_sim_1e3[angle].query('Elyso>0')['Ecalo']/1000, range = ranges, bins=150, histtype='step', linewidth=2, label='MC 1e3 Lyso', density = density)
            #ax.axvline(np.mean(pq_sim[angle].query('Elyso>0')['Ecalo'])/1000, linewidth = 2)
        if logy:
            ax.set_yscale('log')
        if logx:
            ax.set_xscale('log')
        ax.set_xlabel(dict_plot[variable]['label'])
        ax.legend()
    plt.tight_layout()
    
    suffix_1 = '_logX_' if logx else ''
    suffix_2 = '_logY_' if logy else ''
    suffix_3 = '_PDF' if density else ''
    plt.savefig(f'GoodFigures/{variable}/BGO_BSO_FitDistributions_{variable}{suffix_1}{suffix_2}{suffix_3}.png', dpi = 300)
    print(f'BGO_BSO_FitDistributions_{variable}{suffix_1}{suffix_2}{suffix_3}.png Saved in GoodFigures/{variable} Directory')
    plt.close()


d = {}
for crystal in crystals:
    d[crystal] = {}
    for angle in angles:
        d[crystal][angle] = pd.read_parquet(f'Hardware_Fit/parqs/FitInfo_Lyso_fromLaser_{crystal}_{beam}_{angle}_BothChs.parq')

pq_sim = {}
for angle in angles: 
    if angle != 110 and angle != 130:
        pq_sim[angle] = pd.read_parquet(f'Hardware_Fit/parqs/OptOFF_spot1p5/e_BGO_10k_{angle}.parq')

pq_sim_1e3 = {}
for angle in angles: 
    if angle != 110 and angle != 130:
        pq_sim_1e3[angle] = pd.read_parquet(f'Hardware_Fit/parqs/OptOFF_1e3/e_BGO_10k_{angle}.parq')

d_filter = {}
for crystal in ['BSO', 'BGO']:
    d_filter[crystal] = {}
    for angle in angles:
        d_filter[crystal][angle] = d[crystal][angle]
        for sipm in ['_sipm_c', '_sipm_s']:
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f'Q_data{sipm} > 0')
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f'Q_fit{sipm} > 0')
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f'Q_data{sipm}/Q_fit{sipm} < 1.25')
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f'Q_data{sipm}/Q_fit{sipm} > 0.75')
            d_filter[crystal][angle][f'Q_data{sipm}_Q_fit{sipm}'] = d_filter[crystal][angle][f'Q_data{sipm}']/d_filter[crystal][angle][f'Q_fit{sipm}']
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f't0{sipm} < 25')
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f't0{sipm} > -50')
        d_filter[crystal][angle] = d_filter[crystal][angle].query(f'e < 10')
        d_filter[crystal][angle] = d_filter[crystal][angle].query(f'e > 0')
        d_filter[crystal][angle] = d_filter[crystal][angle].query(f'c_over_s_sipm_c < 10')
        d_filter[crystal][angle] = d_filter[crystal][angle].query(f'c_over_s_sipm_c > 0')
        d_filter[crystal][angle] = d_filter[crystal][angle].query(f'c_sipm_s < 1500')
        for sipm in ['_sipm_c', '_sipm_s']:
            mean = np.mean(d_filter[crystal][angle][f'ampl_sideband{sipm}'])
            std = np.std(d_filter[crystal][angle][f'ampl_sideband{sipm}'])
            d_filter[crystal][angle] = d_filter[crystal][angle].query(f'amplitude{sipm} > {mean + 5*std}')


for var in args.variable:
    make_dir('GoodFigures/'+var)
    make_all_distr(var, logx = args.logX, logy = args.logY, density = args.density)


for variable in args.variable:
    lists = {}
    angles_list = {}
    errors_lists = {}
    for crystal in ['BGO', 'BSO']:
        lists[crystal] = []
        errors_lists[crystal] = []
        angles_list[crystal] = []
        for angle in angles:
            if variable == 'c_over_Esim' and angle in [110, 130]:
                continue
            angles_list[crystal].append(angle)
            lists[crystal].append(np.mean(d_filter[crystal][angle][variable]))
            if args.errors:
                errors_lists[crystal].append(np.std(d_filter[crystal][angle][variable]))
            else:
                errors_lists[crystal].append(0)#np.std(d_filter[crystal][angle][variable]))


    fig, ax1 = plt.subplots(figsize=(16, 9))

    string_top_left = f"$\it FCC\,Napoli - BGO-BSO -$" + f" {beam_text}"
    string_top_right = r"$\theta_C = (180-63)$°"
        
    plt.text(0.01, 1.01, string_top_left, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='left')
    plt.text(1, 1.01, string_top_right, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='right')

    ax1.set_xlabel("Angle [°]")
    ax1.set_ylabel(dict_plot[variable]['label'], color=colorslist[0])
    ax1.errorbar(angles_list['BGO'], lists['BGO'], yerr = errors_lists['BGO'], color=color['BGO'], marker='o', label='BGO')
    ax1.tick_params(axis='y', labelcolor=colorslist[0])
    ax1.axvline(180-63, color='firebrick', linestyle='--')  # Linea verticale senza label
    ax1.set_xticks(list(ax1.get_xticks()) + [180-63])  # Aggiunge 63 ai tick esstenti
    ax1.set_xticklabels([r'$\theta_C$' if x == (180-63) else f'{x:.0f}' for x in ax1.get_xticks()])
    plt.legend(loc = 'upper left')


    ax2 = ax1.twinx()  
    ax2.set_ylabel(dict_plot[variable]['label'], color=colorslist[1])
    ax2.errorbar(angles_list['BSO'], lists['BSO'], yerr = errors_lists['BSO'], color=color['BSO'], marker='s', label='BSO')
    ax2.tick_params(axis='y', labelcolor=colorslist[1])
    plt.legend(loc = 'upper right')

    plt.title(dict_plot[variable]['title'])
    fig.tight_layout()

    plt.grid()

    suffix_4 = '_Errors' if args.errors else ''
    plt.savefig(f'GoodFigures/{variable}/Angle_scan_twoax_{variable}{suffix_4}.png', dpi = 300)
    print(f'Angle_scan_twoax_{variable}{suffix_4}.png Saved in GoodFigures/{variable} Directory')
    plt.close()

    
    fig, ax1 = plt.subplots(figsize=(16, 9))
        
    plt.text(0.01, 1.01, string_top_left, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='left')
    plt.text(1, 1.01, string_top_right, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel("Angle [°]")
    plt.ylabel(dict_plot[variable]['label'])
    plt.errorbar(angles_list['BGO'], lists['BGO'], yerr = errors_lists['BGO'], color=color['BGO'], marker='o', label='BGO')
    plt.errorbar(angles_list['BSO'], lists['BSO'], yerr = errors_lists['BSO'], color=color['BSO'], marker='s', label='BSO')    
    plt.axvline(180-63, color='firebrick', linestyle='--')  # Linea verticale senza label
    plt.xticks(list(ax1.get_xticks()) + [180-63])  # Aggiunge 63 ai tick esistenti
    ax1.set_xticklabels([r'$\theta_C$' if x == (180-63) else f'{x:.0f}' for x in ax1.get_xticks()])

    plt.legend(loc = 'best')

    plt.title(dict_plot[variable]['title'])
    fig.tight_layout()

    plt.grid()
    plt.savefig(f'GoodFigures/{variable}/Angle_scan_oneax_{variable}{suffix_4}.png', dpi = 300)
    print(f'Angle_scan_oneax_{variable}{suffix_4}.png Saved in GoodFigures/{variable} Directory')

    plt.close()

