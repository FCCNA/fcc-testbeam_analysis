import numpy as np
import datetime
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='palatino')
import sys
sys.path.append("/eos/user/g/ggaudino/tb_analysis/midas/python")
import midas.file_reader
from tqdm import tqdm 
import os
import pandas as pd
from scipy.signal import find_peaks


def filter_wf(array, index_diff):
    array = np.asarray(array)  # Assicurati che 'array' sia un array NumPy
    result = np.empty_like(array)  # Prealloca l'array di risultato
    result[:index_diff] = array[:index_diff]  # Copia i primi 4 elementi
    result[index_diff:] = array[index_diff:] - array[:-index_diff]  # Calcola i restanti elementi
    return result

def media_mobile(arr, window_size):
    if window_size < 1:
        raise ValueError("La dimensione della finestra deve essere almeno 1.")
    kernel = np.ones(window_size) / window_size
    return np.convolve(arr, kernel, mode='same')

def gaussian(x, mean, sigma, amplitude):
    return amplitude * np.exp(-((x - mean) / sigma)**2 / 2)

# Funzione per la somma di gaussiane
def sum_of_gaussians(x, *params):
    result = np.zeros_like(x)
    num_gaussians = len(params) // 3  # Ogni gaussiana ha 3 parametri: mean, sigma, amplitude
    for i in range(num_gaussians):
        mean = params[i*3]
        sigma = params[i*3 + 1]
        amplitude = params[i*3 + 2]
        result += gaussian(x, mean, sigma, amplitude)
    return result


def parse_time(time_str):
    return datetime.datetime.strptime(time_str, '%a %b %d %H:%M:%S %Y')

def sum_waveform(waveform, channel, maximum = False, ampl_max = None, ampl_min = None):
    wf = np.zeros(len(waveform[0][channel]))
    index = 0
    print(len(waveform))
    for i in range(len(waveform) - 1):
        if maximum:
            if np.max(waveform[i][channel]) < ampl_max and np.max(waveform[i][channel]) > ampl_min:
                index += 1
                wf += waveform[i][channel]
        else: 
            index += 1
            wf += waveform[i][channel]
    wf = wf/index
    return wf


def read_waveform(run_index, starting_index = 0, save_candidate = False, not_all_events = False, max_event = 1000, save_all_wf = False, file_path = '/eos/home-g/ggaudino/tb_analysis/runs/'):
    
    """
    Reads waveform data from a MIDAS file and processes it.

    Parameters:
    - run_index (int): Index of the run.
    - save_candidate (bool): If True, saves all possible peaks in the waveform.
    - not_all_events (bool, optional): If True, reads only max_event events. Default is False.
    - max_event (int, optional): Maximum number of events to read if not_all_events is True. Default is 1000.

    Returns:
    - waveform (dict): Dictionary with the processed waveforms.
    - info (pd.DataFrame): DataFrame with the waveform information.
    """
    file_name = f"run{run_index:05d}.mid.gz"

    mfile = midas.file_reader.MidasFile(file_path + file_name, use_numpy=True)
    mfile.jump_to_start()
    end_odb = mfile.get_eor_odb_dump().data
    Channel_Enabled = end_odb['Equipment']['Trigger']['Settings']['Channel Enable']
    scale = end_odb['Equipment']['Trigger']['Settings']['Channel Scale']
    position = end_odb['Equipment']['Trigger']['Settings']['Channel Position']
    HPOS = end_odb['Equipment']['Trigger']['Settings']['Horizontal Position']
    SARA = end_odb['Equipment']['Trigger']['Settings']['Sample Rate']

    bit = 2**16
    waveform = {}
    mfile.jump_to_start()

    if not not_all_events:
        max_event = 0
        for event in mfile:
            if event.header.event_id != 1:
                continue
            max_event += 1
        mfile.jump_to_start()

    progress_bar = tqdm(total=max_event, colour='green') if max_event > 0 else None
    new_data_list = []
    
    for event in mfile:
        if event.header.serial_number > max_event and not_all_events:
            break
        if event.header.event_id != 1:
            continue
        new_data = {'__event__': event.header.serial_number + starting_index, 
                    '__run__': run_index}
        for bank_name, bank in event.banks.items():
            if len(bank.data):
                channel = np.where(Channel_Enabled)[0][int(bank_name[-1])]
                ch_osci = channel + 1
                if ch_osci not in [1,2]:
                    continue
                wf = 1000 * (((bank.data - (bit / 2)) * scale[channel] * 10 / bit) - (position[channel] * scale[channel]))

                piedistallo = np.mean(wf[:15])
                wf = wf - piedistallo

                #filtered10_wf = filter_wf(wf, 10)
                media = media_mobile(wf, 20)

                times = np.arange(0, 1 / SARA * len(wf), 1 / SARA)
                times -= (1 / SARA * len(wf)) * HPOS / 100
                times *= 1.e9

                ampl = np.max(np.abs(wf))
                #ampl_10 = np.argmax(filtered10_wf)
                ampl_media = np.max(media)

                new_data.update({
                    f'amplitude_channel{ch_osci}': ampl,
                    f'amplitude_media_channel{ch_osci}': ampl_media
                })

                if save_candidate:
                    peaks, _ = find_peaks(np.abs(media), height=1, width=100)
                    sorted_indices = np.argsort(media[peaks])[::-1]
                    new_data['amplitude_sum'] = np.sum(media[peaks[sorted_indices]])
                    for i, j in enumerate(sorted_indices):
                        candidate = new_data.copy()
                        candidate.update({
                            f'__ncandidates__channel{ch_osci}': len(peaks),
                            f'__candidate__channel{ch_osci}': i,
                            f'amplitude_media_cand_channel{ch_osci}': media[peaks[j]]
                        })
                        new_data_list.append(candidate)
                        '''
                else:
                    new_data.update({
                        f'amplitude_10_channel{ch_osci}': filtered10_wf[ampl_10],
                    })
                        '''
                    

                if event.header.serial_number + starting_index not in waveform:
                    waveform[event.header.serial_number + starting_index] = {}
                if save_all_wf:
                    waveform[event.header.serial_number + starting_index][f'{ch_osci}'] = wf
                waveform[event.header.serial_number + starting_index][f'{ch_osci}media'] = media
        if progress_bar:
            progress_bar.update(1)
        new_data_list.append(new_data)
        if starting_index == 0:
            waveform[0]['times'] = times
    if progress_bar:
        progress_bar.close()
    info = pd.DataFrame(new_data_list)
    return waveform, info



def mixing_run(run_index, save_candidate = False, not_all_events = False, max_event = 1000, save_all_wf = False):
    list_wf = []
    list_df = []
    index = 0
    for run_number in run_index:
        print(index)
        tmp_1, tmp_2 = read_waveform(run_number, index, save_candidate, not_all_events, max_event, save_all_wf)
        print(len(tmp_1))
        index = len(tmp_1)
        list_wf.append(tmp_1)
        list_df.append(tmp_2)

    result = {}
    for d in list_wf:
        result.update(d)

    df_final = pd.concat(list_df, ignore_index=True)

    return result, df_final


def custom_plot_layout(title="", xlabel="", ylabel="", figsize=(16, 9), isWaveform = False, crystal = None, beam = None, angle = None, channels = None):
    
    string_top_left = f"$\it FCC\,Napoli - {crystal} -$" + f" {beam}"
    string_top_right = f"{angle}°"
    
    if not isWaveform:
        plt.figure(figsize=figsize)
        plt.text(0.01, 1.01, string_top_left, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='left')
        if angle != None:
            plt.text(1, 1.01, string_top_right, transform=plt.gca().transAxes, fontsize=25,  fontstyle = 'italic', fontfamily = 'serif', verticalalignment='bottom', horizontalalignment='right')
        plt.title(title)
    else:
        fig, ax = plt.subplots(len(channels), 1, sharex = True, figsize = (21, 9*len(channels)))
        for i, channel in enumerate(channels):
            ax[i].set_ylabel(ylabel, fontsize = 20)
            ax[i].grid(True)
            ax[i].text(0.01, 1.01, string_top_left,  transform=ax[i].transAxes, fontsize=25,  fontstyle = 'italic', verticalalignment='bottom', horizontalalignment='left')
            if angle != None:
                ax[i].text(1, 1.01, string_top_right + f' - Channel {channel}',  transform=ax[i].transAxes, fontsize=25,  fontstyle = 'italic', verticalalignment='bottom', horizontalalignment='right')
        plt.subplots_adjust(hspace=0.05)
        plt.title(title)    
    plt.xlabel(xlabel, fontsize = 25)
    plt.ylabel(ylabel, fontsize = 25)


def create_mean_wf(run, path_data, check = True):
    
    times_file = f'npys/Times_{run}.npy'
    mean_waveform_file = f'npys/Mean_Waveform_{run}.npy'

    if check:
        if os.path.exists(times_file) and os.path.exists(mean_waveform_file):
            print(f"I file {times_file} e {mean_waveform_file} esistono già")
            return

    df, wf = read_waveform(run, save_all_wf=True, file_path=path_data)
    lens = 0
    sum_ = np.zeros(len(df[0]['times']))
    for i in df.keys():
        if len(df[i]['1media']) == 12500:
           sum_ += df[i]['1media']
           lens+=1
    sum_ = sum_ / lens
    times = df[0]['times'] 

    np.save(times_file, times)
    np.save(mean_waveform_file, sum_)
    
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return f"Cartella '{path}' creata con successo."
    else:
        return f"La cartella '{path}' esiste già."
