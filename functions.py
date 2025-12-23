import ROOT as R
import array

def get_shortest_interval(distribution, contained_fraction):
    # Function provided by Ernest Olivart (eolivart@icc.ub.edu)
    """This function gives the limits of the contained_fraction (min=1, max=1)
    of the given distribution."""
    import numpy as np
    distribution = distribution[np.isfinite(distribution)] #deletes NaN values
    sorted_data = np.sort(distribution)
    size = len(distribution)
    # print('size =',size)
    n_outside = int((1-contained_fraction)*size) #the number of events you let out
    # print('n_outside =', n_outside)
    intervals = [(x,y) for x, y in zip(sorted_data[:n_outside+1], sorted_data[-n_outside-1:])]
    # print('intervals :', intervals)
    interval_lengths = [y-x for x,y in intervals]
    # print('interval_lengths :', interval_lengths)
    # print(intervals[np.argmin(interval_lengths)])
    return intervals[np.argmin(interval_lengths)]

# -----------------------------------------------------

def plot_pandas_histogram(df_dat, df_sim, col, bin_num, savefig, save_directory, suffix, diff_folder):
    """This function plots a histogram of the colum col in the df_dat and df_sim dataframes.
    Both dataframes need to be pandas dataframes.
    bin_num is the number of bins to be plotted (integer).
    savefig is a boolean that toggles if the figure is saved or not. Figures are saved in
    a folder called histograms in the given save_directory, and in a separate folder for each
    particle."""

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print('********************************')

    print(f'PLOTTING {col}...')

    particle = col.split('_')[0]

    dat_lims = get_shortest_interval(df_dat[col], 0.99)
    sim_lims = get_shortest_interval(df_sim[col], 0.99)
    left_lim = min(dat_lims[0], sim_lims[0])
    right_lim =  max(dat_lims[1], sim_lims[1])

    print('Left limit=', left_lim)
    print('Right limit=', right_lim)

    plt.figure() #start a new figure
    plt.hist(df_dat[col], bins=np.linspace(left_lim, right_lim, bin_num), density=True, 
            alpha=0.7, label=f'Data ({str(len(df_dat[col]))} events)') #plot data
    plt.hist(df_sim[col], bins=np.linspace(left_lim, right_lim, bin_num), density=True, 
            alpha=0.7, label=f'Simulation ({str(len(df_sim[col]))} events)') #plot sim
    plt.legend() #add legend prop={'size': 8}
    #labels
    plt.xlabel(col+suffix)
    # plt.xlabel(r'$m(\pi^+\mu^+\mu^-)$')
    plt.ylabel('# events (density)')
    plt.xlim((left_lim, right_lim))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
    if savefig and diff_folder:
        plt.savefig(f'{save_directory}/{particle}/hist_{col}{suffix}.pdf', format='pdf', dpi=900)
    elif savefig and not diff_folder:
        plt.savefig(f'{save_directory}/hist_{col}{suffix}.pdf', format='pdf', dpi=900)
    plt.close()

    print(f'{col} PLOTTED')

# -----------------------------------------------------

def plot_one_pandas_histogram(df, col, bin_num, savefig, save_directory, suffix):
    """This function plots a histogram of the colum col in the df_dat and df_sim dataframes.
    Both dataframes need to be pandas dataframes.
    bin_num is the number of bins to be plotted (integer).
    savefig is a boolean that toggles if the figure is saved or not. Figures are saved in
    a folder called histograms in the given save_directory, and in a separate folder for each
    particle."""

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    print('********************************')

    print(f'PLOTTING {col}...')

    particle = col.split('_')[0]

    lims = get_shortest_interval(df[col], 0.99)
    left_lim = lims[0]
    right_lim = lims[1]

    print('Left limit=', left_lim)
    print('Right limit=', right_lim)

    plt.figure() #start a new figure
    plt.hist(df[col], bins=np.linspace(left_lim, right_lim, bin_num), density=False, 
            alpha=1, label=f'Data ({str(len(df[col]))} events)') #plot
    # plt.axvline(x=1864.84, linestyle='--', color='tab:orange', label=r'$D_0$ mass')
    plt.legend() #add legend prop={'size': 8}
    #labels
    plt.xlabel(col+suffix)
    # plt.xlabel(r'$m(\pi^+\mu^+\mu^-)$')
    plt.ylabel('# events')
    plt.xlim((left_lim, right_lim))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
    if savefig:
        plt.savefig(f'{save_directory}/hist_{col}{suffix}.pdf', format='pdf', dpi=900)
    plt.close()

    print(f'{col} PLOTTED')



def save_to_root(pandas_df, save_directory, outputfile, suffix):
	print('********************************')
	print('SAVING DF TO ROOT FILE...')
	selected_dat_ROOT = R.RDF.FromPandas(pandas_df) #Get PD df to ROOT df
	root_output_directory = f'{save_directory}/tuples'
	outputfile_name = f'{root_output_directory}/{outputfile}{suffix}.root'

	# Create root file in which to save. If file already exists, it gets rewritten ('RECREATE')
	rootoutputfile = R.TFile.Open(outputfile_name, 'RECREATE')

	# Store df in a tree. 3 args: ROOTDF.Snapshot(tree name, output file name, branches ('' if all))
	selected_dat_ROOT.Snapshot('DecayTree', outputfile_name, '')

	print('DF SAVED TO ROOT FILE')



def save_branch_to_root(pandas_df, branch, save_directory, outputfile, suffix):
    print('********************************')
    print('SAVING DF TO ROOT FILE...')

    # Create tree
    tree = R.TTree('DecayTree', 'DecayTree')

    branch_buffer = array.array('f', [0.0])

    # Create branch
    tree.Branch(branch, branch_buffer, f'{branch}/F')

    # Fill the tree
    for val in pandas_df[branch]:
        branch_buffer[0] = val
        tree.Fill()

    # Create an RDataFrame from this TTree
    selected_dat_ROOT = R.RDataFrame(tree)

    root_output_directory = f'{save_directory}/tuples'
    outputfile_name = f'{root_output_directory}/{outputfile}{suffix}.root'

    # Create root file in which to save. If file already exists, it gets rewritten ('RECREATE')
    rootoutputfile = R.TFile.Open(outputfile_name, 'RECREATE')

    # Store df in a tree. 3 args: ROOTDF.Snapshot(tree name, output file name, branches ('' if all))
    selected_dat_ROOT.Snapshot('DecayTree', outputfile_name, '')

    print('DF SAVED TO ROOT FILE')