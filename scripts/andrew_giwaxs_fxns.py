import numpy as np
import matplotlib.pyplot as plt

# Functions / misc:
def plot_reduced_image(sample='S1', log=True, save=False):
    """
    For quickly plotting reshaped GIWAXS from GIXSGUI-exported tif files
    """
    data = plt.imread(list(reducedPath.glob(f'*{sample}*'))[0])  # importantly, this uses global variable reducedPath
    data = data[:,:,0].copy()
    data.shape

    with np.errstate(divide='ignore'):
        data_log = np.log(data)

    # replace -inf's with zeros
    data_log[(data_log == -np.inf)] = 0

    if log:
        plt.imshow(data_log, origin='lower', extent=[-1.5,0.5,0,2])
    else:
        plt.imshow(data, origin='lower', extent=[-1.5,0.5,0,2])
    
    plt.title(f'{sample}_{sample_dict[sample]}')
    plt.xlabel('Qxy [1/A]')
    plt.ylabel('Qz [1/A]')
    plt.colorbar()
    
    if save==True:
        plt.savefig(savePath.joinpath(f'{sample_dict[sample]}_reduced.svg'))
    
    plt.show()


def linecut_plotter(dict1, dict2, sample, sample_dict, save=False, bot=1, cmap='cool'):
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set(size_inches=(15,5))
    fig.suptitle(f'{sample}: {sample_dict[sample]} linecuts', size=16)
    
    # Choose specified color map, uses exec() to easily input the cmap chosen
    c = {}
    exec(f'c1 = plt.cm.{cmap}(np.linspace(0,1,len(IP_OOP_dict)))', None, c)
    exec(f'c2 = plt.cm.{cmap}(np.linspace(0,1,len(data_dict)))', None, c)
    c1, c2 = c['c1'], c['c2']

    for i, key in enumerate(dict1):
        q = dict1[key][:,0]
        I = dict1[key][:,1]
        ax1.plot(q, I, label=key, c=c1[i])

        # print(data_dict[key])

    for i, key in enumerate(dict2):
        q = dict2[key][:,0]
        I = dict2[key][:,1]
        ax2.plot(q, I, label=key, c=c2[i])

    for ax in (ax1,ax2):
        ax.set(yscale='log', ylim=(3))
        ax.legend()

    ax1.set_ylim(bottom=bot)
    ax2.set_ylim(bottom=bot)

    ax1.set(xlabel='Q [1/A]', ylabel='Intensity')
    ax2.set(xlabel='Q [1/A]', ylabel='Intensity')
    
    if save==True:
        plt.savefig(savePath.joinpath(f'{sample_dict[sample]}.svg'))
    plt.show()


def intensity_at(q_norm):
    print(f'Intensities at q={round(q_norm,5)}')
    for key in data_dict:
        q = data_dict[key][:,0]
        I = data_dict[key][:,1]
        try:
            norm_ind = np.where(np.abs(q-q_norm)<1e-2)[0][1]
            print(f'{key}: {I[norm_ind]}')
        except IndexError:
            norm_ind = 'out of bound'
            print(f'{key}: {norm_ind}')
    print()
    return None

def avg_intensity(q_norm):
    intensities = []
    for key in data_dict:
        q = data_dict[key][:,0]
        I = data_dict[key][:,1]
        try:
            norm_ind = np.where(np.abs(q-q_norm)<1e-2)[0][1]
            # print(f'{key}: {I[norm_ind]}')
            intensities.append(I[norm_ind])
        except IndexError:
            norm_ind = 'out of bound'
            # print(f'{key}: {norm_ind}')
    return sum(intensities)/len(intensities)

def linecut_data(cut='IP', q_max_setpoint=None):
    """ 
    Input: cut = 'IP' (default) or 'OOP'
    Outputs: (q, I) for chosen linecut
    """
    if cut=='IP':
        chi_slice = '-90to-75'
    elif cut=='OOP':
        chi_slice = '-15to0'
    data = normed_IP_OOP_dict[f'{sample}_masked_chi_{chi_slice}']
    q = data[:,0]
    I = data[:,1]
    
    if q_max_setpoint:
        q_max_index = np.where(np.abs(q-q_max_setpoint)<1e-2)[0][0]
        q = q[:q_max_index]
        I = I[:q_max_index]
    
    return q, I

def fit_plotter(q, I, out, top=5, save_plot=False, save_data=False, save_peaks=False):
    """
    Plots a figure with 2 axes, the full fit on a log scale on the left and the components in normal scale on the right
    """
    fig, (ax1, ax2) = plt.subplots(1,2)
    fig.set(size_inches=(12,4))
    fig.suptitle(f'{sample}_{sample_dict[sample]}_{sector}_curve_fitting')

    ax1.plot(q, I, label='data')
    ax1.plot(q, out.best_fit, label='full_fit')
    ax1.set(yscale='log', xlabel='Q [1/A]', ylabel='Intensity')
    ax1.legend()
    
    ax2.plot(q, I, label='data')
    ax2.plot(q, out.best_fit, label='full_fit')
    for key in out.eval_components():
        ax2.plot(q, out.eval_components()[key], label=f'{key}')
    ax2.set(ylim=(-0.1, top), xlabel='Q [1/A]', ylabel='Intensity')
    ax2.legend()
    
    if save_plot:
        plt.savefig(savePath.joinpath(f'{sample}_{sample_dict[sample]}_{sector}_fitted_plot.svg'))
        
    if save_data:
        columns = ['q', 'I', 'full_fit']
        for key in out.eval_components():
            columns.append(key)
        
        output_data = {}
        
        for column in columns:
            if column == 'q':
                output_data[column] = q
            elif column == 'I':
                output_data[column] = I
            elif column == 'full_fit':
                output_data[column] = out.best_fit
            else:
                output_data[column] = out.eval_components()[column]
        
        df = pd.DataFrame(output_data)
        df.to_csv(savePath.joinpath(f'{sample}_{sample_dict[sample]}_{sector}_fitted_plot_data.csv'), index=False)
        
    if save_peaks:
        with savePath.joinpath(f'{sample}_{sample_dict[sample]}_{sector}_fitted_plot_peak_values.json').open('w') as outfile:
            json.dump(out.best_values, outfile)
        
    plt.show()