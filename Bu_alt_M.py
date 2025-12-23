import numpy as np
import pandas as pd
import ROOT as R


# -----------------------------------------------------
# -------------------- LHCB STYLE ---------------------
# -----------------------------------------------------

def set_mpl_LHCb_style():
    # Function provided by Ernest Olivart (eolivart@icc.ub.edu)
    import mplhep
    import matplotlib as mpl
    mplhep.style.use("LHCb2")

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1f77b4', '#ff7f0e', "#2ca02c", '#d62728', 
														'#9467bd', '#8c564b', '#e377c2', '#7f7f7f', 
														'#bcbd22', '#17becf'])

    mpl.rcParams["figure.autolayout"] = False

set_mpl_LHCb_style()




def veto_alt_masses(pandas_df_dat, select):

    # -----------------------------------------------------
    # ---------------------- MASSES -----------------------
    # -----------------------------------------------------

    K_M = 493.677
    Pi_M = 139.57039
    Mu_M = 105.6583755
    D0_M = 1864.84

    # -----------------------------------------------------
    # --------------------- MuMu->KPi ---------------------
    # -----------------------------------------------------

    pandas_df_dat['Pip_Energy'] = np.sqrt(Pi_M**2+pandas_df_dat['Mup_P']**2)
    pandas_df_dat['K_Energy'] = np.sqrt(K_M**2+pandas_df_dat['Mum_P']**2)


    pandas_df_dat['MuMuToKPi_MASS'] = np.sqrt((pandas_df_dat['K_Energy']+pandas_df_dat['Pip_Energy'])**2-
                                            ((pandas_df_dat['Mup_PX']+pandas_df_dat['Mum_PX'])**2+
                                            (pandas_df_dat['Mup_PY']+pandas_df_dat['Mum_PY'])**2+
                                            (pandas_df_dat['Mup_PZ']+pandas_df_dat['Mum_PZ'])**2))

    # -----------------------------------------------------
    # --------------------- MuMu->KK ----------------------
    # -----------------------------------------------------

    pandas_df_dat['K_Mup_Energy'] = np.sqrt(K_M**2+pandas_df_dat['Mup_P']**2)
    pandas_df_dat['K_Mum_Energy'] = np.sqrt(K_M**2+pandas_df_dat['Mum_P']**2)

    pandas_df_dat['MuMuToKK_MASS'] = np.sqrt((pandas_df_dat['K_Mup_Energy']+pandas_df_dat['K_Mum_Energy'])**2-
                                            ((pandas_df_dat['Mup_PX']+pandas_df_dat['Mum_PX'])**2+
                                            (pandas_df_dat['Mup_PY']+pandas_df_dat['Mum_PY'])**2+
                                            (pandas_df_dat['Mup_PZ']+pandas_df_dat['Mum_PZ'])**2))

    # -----------------------------------------------------
    # -------------------- MuMu->PiPi ---------------------
    # -----------------------------------------------------

    pandas_df_dat['Pip_Mup_Energy'] = np.sqrt(Pi_M**2+pandas_df_dat['Mup_P']**2)
    pandas_df_dat['Pip_Mum_Energy'] = np.sqrt(Pi_M**2+pandas_df_dat['Mum_P']**2)

    pandas_df_dat['MuMuToPiPi_MASS'] = np.sqrt((pandas_df_dat['Pip_Mup_Energy']+pandas_df_dat['Pip_Mum_Energy'])**2-
                                            ((pandas_df_dat['Mup_PX']+pandas_df_dat['Mum_PX'])**2+
                                            (pandas_df_dat['Mup_PY']+pandas_df_dat['Mum_PY'])**2+
                                            (pandas_df_dat['Mup_PZ']+pandas_df_dat['Mum_PZ'])**2))

    # Apply vetoes

    if select:
        pandas_df_dat = pandas_df_dat[(abs(pandas_df_dat['MuMuToPiPi_MASS']-D0_M)>=12)]
        pandas_df_dat = pandas_df_dat[(abs(pandas_df_dat['MuMuToKPi_MASS']-D0_M)>=12)]
        pandas_df_dat = pandas_df_dat[(abs(pandas_df_dat['MuMuToKK_MASS']-D0_M)>=12)]

    return(pandas_df_dat)

# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------

def jpsi_constr(pandas_df):
    Pi_M = 139.57039
    Mu_M = 105.6583755
    Jpsi_M = 3096.900

    pandas_df['Pip_Energy'] = np.sqrt(Pi_M**2+pandas_df['Pip_P']**2)
    pandas_df['Mup_Energy'] = np.sqrt(Mu_M**2+pandas_df['Mup_P']**2)
    pandas_df['Mum_Energy'] = np.sqrt(Mu_M**2+pandas_df['Mum_P']**2)

    pandas_df['MuMuPi_MASS'] = np.sqrt((pandas_df['Pip_Energy']+pandas_df['Mup_Energy']+pandas_df['Mum_Energy'])**2-
                                            ((pandas_df['Pip_PX']+pandas_df['Mup_PX']+pandas_df['Mum_PX'])**2+
                                            (pandas_df['Pip_PY']+pandas_df['Mup_PY']+pandas_df['Mum_PY'])**2+
                                            (pandas_df['Pip_PZ']+pandas_df['Mup_PZ']+pandas_df['Mum_PZ'])**2))
    
    pandas_df['MuMu_MASS'] = np.sqrt((pandas_df['Mup_Energy']+pandas_df['Mum_Energy'])**2-
                                            ((pandas_df['Mup_PX']+pandas_df['Mum_PX'])**2+
                                            (pandas_df['Mup_PY']+pandas_df['Mum_PY'])**2+
                                            (pandas_df['Mup_PZ']+pandas_df['Mum_PZ'])**2))
    
    pandas_df['JpsiConstr_MASS'] = pandas_df['MuMuPi_MASS']-pandas_df['MuMu_MASS']+Jpsi_M

    return(pandas_df)
    
    


# -----------------------------------------------------
# -----------------------------------------------------
# -----------------------------------------------------

if __name__=='__main__':

    print('IMPORTING LIBRARIES...')
    import ROOT as R
    print('a')
    import numpy as np
    print('b')
    import pandas as pd
    print('c')
    import matplotlib.pyplot as plt
    print('d')
    import time
    print('e')
    from functions import get_shortest_interval, plot_one_pandas_histogram
    print('LIBRARIES IMPORTED')

    print()
    print('********************************')
    print()

    start_time=time.time()

    # -----------------------------------------------------
    # ---------------- TOGGLES + SETTINGS -----------------
    # -----------------------------------------------------

    select = False
    save_plots = True
    save_directory = '../results/alt_mass'

    # -----------------------------------------------------
    # --------------------- BRANCHES ----------------------
    # -----------------------------------------------------

    Bu_branches = ['Bu_M', 'Bu_DTFPV_PiToK_MASS']

    Jpsi_branches = ['Jpsi_M']

    Pip_branches = ['Pip_M', 'Pip_P', 'Pip_PX', 'Pip_PY', 'Pip_PZ']

    Mup_branches = ['Mup_M', 'Mup_P', 'Mup_PX', 'Mup_PY', 'Mup_PZ']

    Mum_branches = ['Mum_M', 'Mum_P', 'Mum_PX', 'Mum_PY', 'Mum_PZ']

    branches = Bu_branches + Jpsi_branches + Pip_branches + Mup_branches + Mum_branches

    # -----------------------------------------------------
    # -------------------- DF SETTING ---------------------
    # -----------------------------------------------------

    tree = 'BuToPipMuMu/DecayTree'
    datapath = '/home/msalvador/tuplesTFG/b2pimumu_coll24_magupspr24c2.root'
    simpath = '/home/msalvador/tuplesTFG/b2pimumu_mc24_magupw3134.root'

    print('SETTING DATAFRAMES...')

    root_df_dat = R.RDataFrame(tree, datapath, branches)
    root_df_sim = R.RDataFrame(tree, simpath, branches+['Bu_BKGCAT'])

    root_df_sim = root_df_sim.Filter('Bu_BKGCAT==10')

    # We work in rare mode
    root_df_dat = root_df_dat.Filter('Jpsi_M*Jpsi_M<8000000')
    root_df_sim = root_df_sim.Filter('Jpsi_M*Jpsi_M<8000000')


    print('ROOT DFs SET')

    numpy_df_dat = root_df_dat.AsNumpy(branches)
    numpy_df_sim = root_df_sim.AsNumpy(branches+['Bu_BKGCAT']) #add Bu_BKGCAT to trim data
    print('NUMPY DFs SET')

    pandas_df_dat = pd.DataFrame(numpy_df_dat)
    pandas_df_sim = pd.DataFrame(numpy_df_sim)
    pandas_df_dat = pandas_df_dat.dropna() #drop NaNs from data
    pandas_df_sim = pandas_df_sim.dropna() #drop NaNs from data


    print('PANDAS DFs SET')

    print('ALL DATAFRAMES SET')

    pandas_df_dat = veto_alt_masses(pandas_df_dat, select)

    if select:
        suffix = '_selected'
    else:
        suffix = ''

    print('PLOTTING BRANCHES...')
    branch_directory = save_directory
    bin_num = 50
    # plot_one_pandas_histogram(pandas_df_dat, 'Bu_PiToK_MASS', bin_num, save_plots, branch_directory, suffix)
    plot_one_pandas_histogram(pandas_df_dat, 'MuMuToKPi_MASS', bin_num, save_plots, '../results/presentation', suffix)
    plot_one_pandas_histogram(pandas_df_dat, 'MuMuToKK_MASS', bin_num, save_plots, '../results/presentation', suffix)
    plot_one_pandas_histogram(pandas_df_dat, 'MuMuToPiPi_MASS', bin_num, save_plots, '../results/presentation', suffix)
    # plot_one_pandas_histogram(pandas_df_sim, 'Bu_DTFPV_PiToK_MASS', bin_num, save_plots, branch_directory, suffix)