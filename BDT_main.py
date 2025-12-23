print('IMPORTING LIBRARIES...')
import ROOT as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from functions import get_shortest_interval, plot_pandas_histogram, plot_one_pandas_histogram, save_to_root, save_branch_to_root
from Bu_alt_M import veto_alt_masses, jpsi_constr
from fits import exponentialFit
print('LIBRARIES IMPORTED')

print()
print('********************************')
print()

start_time=time.time()

# -----------------------------------------------------
# ---------------- TOGGLES + SETTINGS -----------------
# -----------------------------------------------------

# Model toggles
preselection = False
postselection = False
training = False
FoM = False

# Plot toggles
plot_branches = True
save_plots = True
bdt_plots = False
Bu_plot = False

# Save file toggles
save_root = False
save_fit = False

# Directories and files
save_directory = '../results'
preselection_file = './preselection.txt'

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


# -----------------------------------------------------
# --------------------- BRANCHES ----------------------
# -----------------------------------------------------

Bu_branches = ['Bu_M', 'Bu_DTF_JpsiConstr_MASS', 'Bu_PT', 'Bu_BPVIP', 'Bu_BPVIPCHI2', 'Bu_MAXDOCA', 'Bu_CHI2', 'Bu_ETA']

Jpsi_branches = ['Jpsi_M', 'Jpsi_PT', 'Jpsi_MAXDOCA', 'Jpsi_ETA']

Pip_branches = ['Pip_ETA', 'Pip_PT', 'Pip_CHI2DOF'] 

Mup_branches = ['Mup_BPVIP', 'Mup_ETA', 'Mup_PT', 'Mup_CHI2DOF']

Mum_branches = ['Mum_BPVIP', 'Mum_ETA', 'Mum_PT', 'Mum_CHI2DOF']

preselection_branches = ['Pip_PROBNN_GHOST', 'Mup_PROBNN_E', 'Mup_PROBNN_GHOST', 'Mup_PROBNN_K',
						'Mup_PROBNN_MU', 'Mup_PROBNN_PI', 'Mum_PROBNN_GHOST', 'Mum_PROBNN_K',
						'Mum_PROBNN_MU', 'Mum_PROBNN_PI', 'Bu_Hlt1TwoTrackMVADecisionDecision_TOS',
						'Bu_Hlt1TrackMuonMVADecision_TOS', 'Bu_Hlt1TrackMVADecisionDecision_TOS']

alt_mass_branches = 	['Pip_P', 'Pip_PX', 'Pip_PY', 'Pip_PZ',
						'Mup_P', 'Mup_PX', 'Mup_PY', 'Mup_PZ', 'Mum_P', 'Mum_PX', 'Mum_PY', 'Mum_PZ']

post_BDT_selection_branches = ['Pip_PROBNN_PI', 'Pip_PROBNN_K'] #two PROBNN branches added to be able to cut after BDT

# Masses not used for training
bumass = 'Bu_M'
alt_bumass = 'Bu_DTF_JpsiConstr_MASS' 
jpsimass = 'Jpsi_M'

branches = Bu_branches + Jpsi_branches + Pip_branches + Mup_branches + Mum_branches + post_BDT_selection_branches
allbranches = branches + preselection_branches + alt_mass_branches

# Branches without mass
branchesnomass = branches.copy()
branchesnomass.remove(bumass)
branchesnomass.remove(alt_bumass)	
branchesnomass.remove(jpsimass)
branchesnomass = [branch for branch in branchesnomass if branch not in post_BDT_selection_branches] #post_BDT_sel not used in BDT


# -----------------------------------------------------
# -------------------- DF SETTING ---------------------
# -----------------------------------------------------

tree = 'BuToPipMuMu/DecayTree'
datapath = '/home/msalvador/tuplesTFG/b2pimumu_coll24_magupspr24c2.root'
simpath = '/home/msalvador/tuplesTFG/b2pimumu_mc24_magupw3134.root'

print('SETTING DATAFRAMES...')

root_df_dat = R.RDataFrame(tree, datapath, allbranches)
root_df_sim = R.RDataFrame(tree, simpath, allbranches+['Bu_BKGCAT'])
print('ROOT DFs SET')
print('Number of data proxy entries before everything:', root_df_dat.Count().GetValue())


# -----------------------------------------------------
# -------------------- PRESELECTION -------------------
# -----------------------------------------------------

if preselection:
	print('********************************')
	print('PRESELECTION')
	# Get preselection cuts from file
	with open(preselection_file, 'r') as presel_file:
		cuts = presel_file.read().replace('\n', ' && ')
	print(cuts)
	# Apply preselection
	root_df_dat = root_df_dat.Filter(cuts)
	root_df_sim = root_df_sim.Filter(cuts)
	print('PRESELECTION DONE')
	print('Number of data proxy entries after cuts:', root_df_dat.Count().GetValue())
	print('********************************')

root_df_sim = root_df_sim.Filter('Bu_BKGCAT==10') #select only MC events with 'Bu_BKGCAT'==10

# Separate rare and Jpsi regions (we will work with rare region)
jpsi_df_dat = root_df_dat.Filter('(Jpsi_M*Jpsi_M>8000000) && (Jpsi_M*Jpsi_M<11000000)')
jpsi_df_sim = root_df_sim.Filter('(Jpsi_M*Jpsi_M>8000000) && (Jpsi_M*Jpsi_M<11000000)')

rare_df_dat = root_df_dat.Filter('Jpsi_M*Jpsi_M<8000000')
rare_df_sim = root_df_sim.Filter('Jpsi_M*Jpsi_M<8000000')

print('Number of rare region data proxy after everything:', rare_df_dat.Count().GetValue())

numpy_df_dat = rare_df_dat.AsNumpy(branches+alt_mass_branches+preselection_branches)
numpy_df_sim = rare_df_sim.AsNumpy(branches+alt_mass_branches+preselection_branches)

jpsi_np_dat = jpsi_df_dat.AsNumpy(branches+alt_mass_branches)
jpsi_np_sim = jpsi_df_sim.AsNumpy(branches)
print('NUMPY DFs SET')

pandas_df_dat = pd.DataFrame(numpy_df_dat)
pandas_df_sim = pd.DataFrame(numpy_df_sim)
pandas_df_dat = pandas_df_dat.dropna() #drop NaNs from data
pandas_df_sim = pandas_df_sim.dropna() #drop NaNs from data

if preselection:
	pandas_df_dat = veto_alt_masses(pandas_df_dat, True)

pandas_df_dat = pandas_df_dat.sample(n=len(pandas_df_sim), random_state=42)

jpsi_pd_dat = pd.DataFrame(jpsi_np_dat)
jpsi_pd_sim = pd.DataFrame(jpsi_np_sim)
print('PANDAS DFs SET')

print('ALL DATAFRAMES SET')

print('BLINDING SIGNAL REGION...')
blinded_region_lims = get_shortest_interval(pandas_df_sim['Bu_M'], 0.995) #gets limits of blinded region (99.5% of sig events)
print('99.5% simulation mass limits', blinded_region_lims[0], blinded_region_lims[1])

blinded_pd_df_dat = pandas_df_dat[(pandas_df_dat['Bu_M']<=blinded_region_lims[0]) |
                              (pandas_df_dat['Bu_M']>=blinded_region_lims[1])] #blind the signal region 
                                                                               #(keep only values outside of it)
print('SIGNAL REGION BLINDED')


# -----------------------------------------------------
# ------------------- BRANCH PLOTS --------------------
# -----------------------------------------------------

if plot_branches:
	print('PLOTTING BRANCHES...')
	branch_directory = save_directory+'/histograms/rawdata_final'
	bin_num = 50
	diff_folder = True
	suffix=''

	for branch in branches+preselection_branches:
		plot_pandas_histogram(blinded_pd_df_dat, pandas_df_sim, branch, bin_num, save_plots, branch_directory, suffix, diff_folder)


# -----------------------------------------------------
# ---------------------- TRAINING ---------------------
# -----------------------------------------------------

# Add new column to pd dataframe (0 if it comes from data-->BKG; 1 if it comes from simulation-->SIG) 
blinded_pd_df_dat['category'] = 0 
pandas_df_sim['category'] = 1
BDT_data = pd.concat([blinded_pd_df_dat, pandas_df_sim], copy=True, ignore_index=True) #Join both DF
training_data_wm, validation_data_wm = train_test_split(BDT_data, test_size=0.2, random_state=22) #wm=with mass
# If I train, I need to drop the mass from the list of features and recover it later
training_data = training_data_wm.drop([bumass,jpsimass], axis = 1)
validation_data = validation_data_wm.drop([bumass,jpsimass], axis = 1)

if preselection:
		suffix='_preselected'
else:
	suffix='_not_preselected'

if training:
	print('********************************')
	print('TRAINING...')
	bdt = xgb.XGBClassifier()
	bdt.fit(training_data[branchesnomass], training_data['category'])
	print(bdt)

	print('********************************')
	print('SAVING MODEL...')
	
	with open(f'{save_directory}/models/bdt_model{suffix}.pkl', 'wb') as f:
		pickle.dump(bdt, f)

else:
	print('********************************')
	print('IMPORTING MODEL...')
	print('********************************')
	with open(f'{save_directory}/models/bdt_model{suffix}.pkl', 'rb') as f:
		bdt = pickle.load(f)


# I need to recover the mass in my pandas dataframe
validation_data[bumass] = validation_data_wm[bumass]
training_data[bumass] = training_data_wm[bumass]

# -----------------------------------------------------
# ---------------------- OUTPUTS ----------------------
# -----------------------------------------------------

# I apply the generated model to my data
training_data['BDT'] = bdt.predict_proba(training_data[branchesnomass])[:,1]
validation_data['BDT'] = bdt.predict_proba(validation_data[branchesnomass])[:,1]


# -----------------------------------------------------
# --------------------- BDT PLOTS ---------------------
# -----------------------------------------------------

if bdt_plots:
	if preselection:
		suffix = '_preselected_final'
	else:
		suffix = '_not_preselected'


	# -------------------- PROBABILITY --------------------

	print('********************************')
	print('PLOTTING PROBABILITIES...')

	train_bkg = training_data.query('category==0')
	train_sig = training_data.query('category==1')
	test_bkg = validation_data.query('category==0')
	test_sig = validation_data.query('category==1')

	# Probability distribution of BDT
	plt.figure()
	plt.hist(train_bkg['BDT'], bins=50, range=(0,1), density=True, alpha=0.5, label='Train BKG') #plot BKG from training
	plt.hist(train_sig['BDT'], bins=50, range=(0,1), density=True, alpha=0.5, label='Train SIG') #plot SIG from training
	plt.hist(test_bkg['BDT'], bins=50, range=(0,1), density=True, alpha=1, label='Test BKG', 
			edgecolor='green', histtype='step') #plot BKG from test
	plt.hist(test_sig['BDT'], bins=50, range=(0,1), density=True, alpha=1, label='Test SIG', 
			edgecolor='red', histtype='step') #plot SIG from test
	plt.xlim(0,1)
	plt.xlabel(f'Probability')
	plt.ylabel('Arbitrary units')
	plt.legend()
	plt.savefig(f'{save_directory}/bdtoutputs/outcome_probs{suffix}.pdf', format='pdf', dpi=900)
	plt.close()

	# Probability distribution of BDT in log scale
	plt.figure()
	plt.hist(train_bkg['BDT'], bins=50, range=(0,1), density=True, alpha=0.5, label='Train BKG', 
			log=True) #plot BKG from training
	plt.hist(train_sig['BDT'], bins=50, range=(0,1), density=True, alpha=0.5, label='Train SIG', 
			log=True) #plot SIG from training
	plt.hist(test_bkg['BDT'], bins=50, range=(0,1), density=True, alpha=1, label='Test BKG', 
			edgecolor='green', histtype='step', log=True) #plot BKG from test
	plt.hist(test_sig['BDT'], bins=50, range=(0,1), density=True, alpha=1, label='Test SIG', 
			edgecolor='red', histtype='step', log=True) #plot SIG from test
	plt.xlim(0,1)
	plt.xlabel(f'Probability')
	plt.ylabel('Arbitrary units')
	plt.legend()
	if save_plots:
			plt.savefig(f'{save_directory}/bdtoutputs/outcome_probs_log{suffix}.pdf', format='pdf', dpi=900)
	plt.close()

	print('PROBABILITIES PLOTTED')


	# ----------------- FEATURE IMPORTANCE ----------------

	print('********************************')
	print('PLOTTING FEATURE IMPORTANCES')

	bdt.get_booster().feature_names = branchesnomass #define features
	importance = bdt.get_booster().get_score(importance_type='gain') #get importances
	for key in importance.keys():
			importance[key] = round(importance[key],1) #round all importances to 1 decimal position

	# Plot
	plt.figure()
	xgb.plot_importance(importance)
	ax = xgb.plot_importance(importance)
	for text in ax.texts:
		text.set_fontsize(24)
	plt.yticks(fontsize=24)
	plt.title(None)
	plt.ylabel(None)
	plt.xlabel(f'F score')
	plt.grid(False)
	plt.tight_layout()
	if save_plots:
			plt.savefig(f'{save_directory}/bdtoutputs/feature_importances{suffix}.pdf', format='pdf')
	plt.close()

	print('FEATURE IMPORTANCES PLOTTED')


	# --------------- TRUE POS VS FALSE POS ---------------

	print('********************************')
	print('PLOTTING TRUE POS VS FALSE POS...')

	def bdt_performance(bdt, data, features):
			from sklearn.metrics import roc_curve, auc
			y_score = bdt.predict_proba(data[features])[:,1]
			fpr, tpr, thresholds = roc_curve(data['category'], y_score)
			auc = auc(fpr, tpr)
			return (auc, fpr, tpr)

	train_auc, train_fpr, train_tpr = bdt_performance(bdt, training_data, branchesnomass)
	test_auc, test_fpr, test_tpr = bdt_performance(bdt, validation_data, branchesnomass)

	plt.figure()
	plt.plot(train_fpr, train_tpr, label=f'AUC: Train = {str(round(train_auc,4))}', zorder=1)
	plt.plot(test_fpr, test_tpr, label=f'AUC: Test = {str(round(test_auc,4))}', color='tab:orange', linestyle=(0, (5, 7)), zorder=1)
	# (0, (5, 10))
	plt.hlines(y=1, xmin=-0.1, xmax=1.1, color='grey', linestyle='--', zorder=0)
	plt.vlines(x=0, ymin=-0.1, ymax=1.1, color='grey', linestyle='--', zorder=0)
	plt.xlim(-0.1, 1.1)
	plt.ylim(-0.1, 1.1)
	plt.xlabel(f'False Positive Rate')
	plt.ylabel(f'True Positive Rate')
	plt.legend()
	if save_plots:
			plt.savefig(f'{save_directory}/bdtoutputs/tpr_fpr{suffix}.pdf', format='pdf')
	plt.close()


# -----------------------------------------------------
# --------------------- Bu_M PLOT ---------------------
# -----------------------------------------------------

if preselection:
	suffix='_preselected'
else:
	suffix='_not_preselected'


# -----------------------------------------------------
# --------------------- APPLY BDT ---------------------
# -----------------------------------------------------

# Apply trained BDT to raw data and to simulation (to get FoM)
jpsi_pd_dat = jpsi_constr(jpsi_pd_dat) #add Jpsi constrain column to dataframe

jpsi_pd_dat['BDT'] = bdt.predict_proba(jpsi_pd_dat[branchesnomass])[:,1]
jpsi_pd_sim['BDT'] = bdt.predict_proba(jpsi_pd_sim[branchesnomass])[:,1]
pandas_df_dat['BDT'] = bdt.predict_proba(pandas_df_dat[branchesnomass])[:,1]

# Cut data at prob>=cut
cut = 0.8774
# cut = 0.97 if cuts before BDT
selected_jpsi_dat = jpsi_pd_dat[(jpsi_pd_dat['BDT']>=cut)]
selected_rare_dat = pandas_df_dat[(pandas_df_dat['BDT']>=cut)]

# Apply cuts at PROBNN to reduce K contribution
if postselection:
	selected_jpsi_dat = selected_jpsi_dat[(selected_jpsi_dat['Pip_PROBNN_PI']>0.5)]
	selected_jpsi_dat = selected_jpsi_dat[(selected_jpsi_dat['Pip_PROBNN_K']<0.2)]
	selected_rare_dat = selected_rare_dat[(selected_rare_dat['Pip_PROBNN_PI']>0.5)]
	selected_rare_dat = selected_rare_dat[(selected_rare_dat['Pip_PROBNN_K']<0.2)]

# Save to ROOT file
if save_root:
	save_branch_to_root(selected_jpsi_dat, 'Bu_M', save_directory, 'b2pimumu_BDT_jpsi', '_allcuts') #save to file
	save_branch_to_root(selected_jpsi_dat, 'Bu_DTF_JpsiConstr_MASS', save_directory, 'b2pimumu_BDT_jpsi_constr', '_allcuts') #save to file
	save_branch_to_root(selected_jpsi_dat, 'JpsiConstr_MASS', save_directory, 'b2pimumu_BDT_jpsi_manual_constr', '_allcuts') #save to file

if save_root:
	save_branch_to_root(selected_rare_dat, 'Bu_M', save_directory, 'b2pimumu_BDT_rare', '_allcuts') #save to file

# Plot both rare and jpsi regions
if Bu_plot:
	print('********************************')
	print('PLOTTING NEW Bu_M...')
	branch_directory = f'{save_directory}/bdtoutputs/final'
	diff_folder = False

	suffix_jpsi = suffix+'_jpsi'
	suffix_rare = suffix+'_rare'
	
	bin_num = 50
	plot_one_pandas_histogram(selected_jpsi_dat, bumass, bin_num, save_plots, branch_directory, suffix_jpsi)
	plot_one_pandas_histogram(selected_rare_dat, bumass, bin_num, save_plots, branch_directory, suffix_rare)


# -----------------------------------------------------
# -------------------- ROOT SAVING --------------------
# -----------------------------------------------------

# if save_root:
# 	save_Bu_M_to_root(selected_jpsi_dat, save_directory, 'b2pimumu_BDT', suffix)
# 	save_Bu_M_to_root(jpsi_pd_dat, save_directory, 'b2pimumu_noBDT', suffix)

# -----------------------------------------------------
# ------------------ FIGURE OF MERIT ------------------
# -----------------------------------------------------

# FoM optimized in jpsi mode

if FoM:

	def exponential(c, x):
		import math
		return math.e**(c*x)

	def bkg_yield_in_signal_window(sideband_yield, c, cerr, signal_window, sideband_window): #sideband_yield: initial number of events that enter
		integral_1 = (exponential(c, signal_window[1])-exponential(c, signal_window[0]))
		integral_2 = (exponential(c, sideband_window[1])-exponential(c, sideband_window[0]))
		di1 = abs(exponential(c, signal_window[1])*signal_window[1]-exponential(c, signal_window[0])*signal_window[0])*cerr
		di2 = abs(exponential(c, sideband_window[1])*sideband_window[1]-exponential(c, sideband_window[0])*sideband_window[0])*cerr
		error = sideband_yield*((abs(di1/integral_2))**2+(abs(integral_1*di2/(integral_2)**2))**2)
		return integral_1*sideband_yield/integral_2, error
	
	selected_sim = jpsi_pd_sim[(jpsi_pd_sim['BDT']>=0.2)] #MC data for prob>0.2 used to find efficiencies
	total_sig = len(selected_sim['Bu_M']) #Taken from selected_sim (cut at prob=0.2)
	sig_yield = 17331 #Taken from selected_dat fit (cut at prob=0.2)
	# 3830 
	sig_yield_err = 164
	# 70

	ncuts = 100
	prob_min = 0
	prob_max = 1
	prob = prob_min #Start value for prob cut
	S_values = []
	B_values = []
	FoM_values = []
	S_err_values = []
	B_err_values = []
	FoM_err_values = []
	prob_values = []

	# Loop that gets FoM values for each probability cut
	for i in range(0, ncuts):
		# print(prob)
		new_selected_dat = jpsi_pd_dat[(jpsi_pd_dat['BDT']>=prob)]
		new_selected_sim = jpsi_pd_sim[(jpsi_pd_sim['BDT']>=prob)]

		sig_BDT = len(new_selected_sim['Bu_M'])
		efficiency = sig_BDT/total_sig
		S = sig_yield*efficiency
		S_err = sig_yield_err*efficiency

		# print(efficiency)

		# Save df to file for fitting
		save_branch_to_root(new_selected_dat, 'Bu_M', save_directory, 'FoM', '_final')

		# Fit BKG data to get exponential coefficients
		file     = 'FoM_final.root'
		cuts     = ''
		folderin   = '/home/msalvador/results/tuples'
		folderout   = '/home/msalvador/results/fits'
		name     = 'Bu_M'
		tree     = 'DecayTree'
		xmin     = 5500
		xmax     = 6000
		out      = f'FoM_{prob}_{xmin}_{xmax}'
		
		tau, tau_err, sideband_yield = exponentialFit(file, cuts, folderin, folderout, name, tree, xmin, xmax, out, save_fit)

		# Get value for B by extrapolating combinatorial BKG from fit
		signal_window = [5100, 5400]
		sideband_window = [5500, 6000]

		B, B_err = bkg_yield_in_signal_window(sideband_yield, tau, tau_err, signal_window, sideband_window)

		# Store values to lists
		S_values.append(S)
		B_values.append(B)
		FoM_values.append(S/np.sqrt(S+B))
		S_err_values.append(S_err)
		B_err_values.append(B_err)
		FoM_err = np.sqrt(((S+2*B)/(2*(S+B)**(3/2))*S_err)**2+(S/(2*(S+B)**3/2)*B_err)**2)
		FoM_err_values.append(FoM_err)
		prob_values.append(prob)

		# Next prob value
		prob += (prob_max-prob_min)/ncuts
	
	# Plot FoM S(prob) and B(prob)
	print('PLOTTING FoM FIGURES...')
	plt.figure()
	plt.errorbar(prob_values, FoM_values, yerr=FoM_err_values, elinewidth=0.7)
	plt.axvline(x = 0.8774, color = 'tab:orange', label = 'Optimal cut = 0.8774', linewidth=0.7)
	plt.xlabel('BDT cut')
	plt.ylabel('FoM')
	plt.xlim((prob_min-0.05, prob_max+0.05))
	plt.legend()
	plt.savefig(f'{save_directory}/FoM/FoM_final.pdf', format='pdf', dpi=900)
	plt.close()

	plt.figure()
	plt.errorbar(prob_values, S_values, yerr=S_err_values, elinewidth=0.7)
	plt.xlabel('BDT cut')
	plt.ylabel('S')
	plt.xlim((prob_min-0.05, prob_max+0.05))
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
	plt.savefig(f'{save_directory}/FoM/S_final.pdf', format='pdf', dpi=900)
	plt.close()

	plt.figure()
	plt.errorbar(prob_values, B_values, yerr=B_err_values, elinewidth=0.7)
	plt.xlabel('BDT cut')
	plt.ylabel('B')
	plt.xlim((prob_min-0.05, prob_max+0.05))
	plt.ticklabel_format(axis='y', style='sci', scilimits=(-4,-3))
	plt.savefig(f'{save_directory}/FoM/B_final.pdf', format='pdf', dpi=900)
	plt.close()

	print('FoM FIGURES PLOTTED')

	max_FoM = max(FoM_values)
	max_FoM_index = FoM_values.index(max_FoM)
	max_FoM_cut = prob_values[max_FoM_index]

	print(f'Max FoM: {max_FoM} located at --> {max_FoM_cut}')

print()
print('********************************')
print('END OF PROGRAM')
print()
print('Elapsed time (s):', time.time()-start_time)