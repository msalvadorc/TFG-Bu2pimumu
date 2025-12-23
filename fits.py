# Adapted from original code by Ernest Olivart (eolivart@icc.ub.edu)

import os
import ROOT as R

RooFit         = R.RooFit
RooRealVar     = R.RooRealVar
RooArgList     = R.RooArgList
RooArgSet      = R.RooArgSet
RooDataSet     = R.RooDataSet
RooGaussian    = R.RooGaussian
RooExponential = R.RooExponential
RooAddPdf      = R.RooAddPdf
RooCrystalBall = R.RooCrystalBall
RooAbsReal     = R.RooAbsReal
RooFormulaVar     = R.RooFormulaVar

# LHCB STYLE
R.gROOT.ProcessLine(".L /home/pvidrier/lhcbStyle.C")
R.lhcbStyle()

# -----------------------------------------------------
# -------------------- EXPONENTIAL --------------------
# -----------------------------------------------------

def exponentialFit(file, cuts, folderin, folderout, name, tree, xmin, xmax, out, save_fit):
    # file= name of the .root file with the bdt applied
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, '' if none
    # xmin= minimum of the range
    # xmax= maximum of the range
    # name= name for the fit plot

    # define variables and pdfs
    Bu_Jpsimode_MASS = RooRealVar(name, name, xmin, xmax)

    # *************** COMBINATORIAL ***************
    tau = RooRealVar('tau', 'tau', -0.0016602798005369492, -1, 0.)
    exp = RooExponential('exp', 'exp', Bu_Jpsimode_MASS, tau)

    # define coefficiencts
    ncom = RooRealVar('ncom', 'ncom', 21082.709422450684, 0, 500000)

    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(exp)
    
    coeff.add(ncom)
    
    model = R.RooAddPdf('model', 'model', suma, coeff)

    file = f'{folderin}/{file}'

    # define data 
    entries = 0
    filtered = R.RDataFrame(tree, file).Filter(f'({name}>{xmin}) && ({name}<{xmax})') #Take only data between xmin and xmax
    for var in list(filtered.GetColumnNames()):
        filtered = filtered.Filter(f'({var}>0) || ({var}<=0)') #Smart way to remove NaNs
    treee = filtered.Snapshot('DecayTree',f'{folderout}/FilteredTree_data_{name}.root',f'{name}') # 
    filee = R.TFile(f'{folderout}/FilteredTree_data_{name}.root') #Create new file to store filtered tree
    treeecut = filee.Get('DecayTree')
    entries += treeecut.GetEntries()
    ds = RooDataSet('data', 'dataset with x', treeecut, RooArgSet(Bu_Jpsimode_MASS))

    #create and open the canvas
    can = R.TCanvas('hist','hist', 200,10, 1000, 800)
    pad1 = R.TPad('pad1', 'Histogram', 0., 0.20, 1.0, 0.975, 0)  # Shifted down slightly
    pad2 = R.TPad('pad2', 'Residual plot', 0., 0.10, 1.0, 0.20, 0)  # Centered gap
    can.cd()

    pad1.Draw()
    pad2.Draw()
    ################

    # plot dataset and fit
    massFrame = Bu_Jpsimode_MASS.frame()
    
    ds.plotOn(massFrame)
 
    fitResults = model.fitTo(ds, RooFit.Save())
    model.plotOn(massFrame,
                 RooFit.Name('curve_model'))


    #Construct the histogram for the residual plot and plot it on pad2
    hresid = massFrame.pullHist()
    chi2 = massFrame.chiSquare()
    hresid.GetYaxis().SetLabelSize(0.15)
    hresid.GetXaxis().SetLabelSize(0.15)

    pad2.cd()
    hresid.SetTitle('')
    hresid.Draw()
    
    model.plotOn(massFrame, RooFit.Components('exp'), RooFit.LineColor(2),
                RooFit.Name('exp'))
    ds.plotOn(massFrame)

    #Draw the fitted histogram into pad1
    pad1.cd()
    massFrame.SetTitle('')
    massFrame.GetXaxis().SetTitle('Bu_M')

    # print results
    print('{} has been fit to {} with a chi2 = {}'.format(model.GetName(), file, chi2))
 
    print('Total number of entries is: {}'.format(ds.numEntries()))

    massFrame.Draw()

    # create legend
    legend = R.TLegend(0.65, 0.65, 0.9, 0.9)
    legend.SetTextSize(0.03)
    legend.AddEntry('exp', 'Combinatorial', 'l')
    legend.Draw()

    # Create a TPaveText for the additional text box
    extra_info = R.TPaveText(0.3, 0.2, 0.5, 0.35, 'NDC')  # Coordinates in normalized device coordinates (NDC)
    extra_info.SetFillColor(0)        # Transparent background
    extra_info.SetFillStyle(0)        # No fill style
    extra_info.SetLineColor(0)        # No border
    extra_info.SetBorderSize(0)       # Remove the shadow/border effect
    extra_info.SetTextAlign(12)       # Align left, vertically centered
    extra_info.SetTextSize(0.03)      # Text size

    # Add entries to the text box
    extra_info.AddText(f'#chi^{{2}} / NDF = {chi2:.2f}')
    extra_info.AddText(f'Total entries: {ds.numEntries():.0f}')

    # Draw the TPaveText on the same canvas
    pad1.cd()  # Switch to the pad where the fit is drawn
    extra_info.Draw()
    
    #Save the result
    if save_fit:
        can.SaveAs(f'{folderout}/Fit_fixed_{out}.pdf')

    #####################
        with open(f'{folderout}/Fit_fixed_{out}.txt', 'w') as f:
                f.write('chi2:       '+str(chi2)+'\n')
                f.write('tau:            '+str(tau.getValV())+'\n')
                f.write('tau err:        '+str(tau.getError())+'\n')
                f.write('Com yield:      '+str(ncom.getValV())+'\n')
                f.write('Com yield err:  '+str(ncom.getError())+'\n')
        f.close

        print(f'Fit_fixed_{out}.pdf done')

    return tau.getValV(), tau.getError(), ds.sumEntries()


# -----------------------------------------------------
# -------------------- GAUSS + EXP --------------------
# -----------------------------------------------------

def gauss_exponentialFit(file, cuts, folderin, folderout, name, tree, mean_val, xmin, xmax, out):
    # file= name of the .root file with the bdt applied
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, '' if none
    # mean_val= initial value of the mean
    # xmin= minimum of the range
    # xmax= maximum of the range
    # name= name for the fit plot

    # define variables and pdfs
    Bu_Jpsimode_MASS = RooRealVar(name, name, xmin, xmax)

    # *************** SIGNAL ***************
    mean_gauss  = RooRealVar('Mean', 'mean_gauss', mean_val, mean_val-100, mean_val+100)    
    sigma_gauss = RooRealVar('Sigma', 'sigma_gauss', 36.95859787742036, 10, 35)
    gauss_sig  = RooGaussian('gauss', 'gauss', Bu_Jpsimode_MASS, mean_gauss, sigma_gauss)

    # *************** COMBINATORIAL ***************
    tau = RooRealVar('tau', 'tau', -0.0018498385144551026, -1, 0)
    exp = RooExponential('exp', 'exp', Bu_Jpsimode_MASS, tau)
    tau.setConstant()

    # Tau value for final allcuts jpsi: -0.0018498385144551026
    # -0.0018498385144551026

    # define coefficiencts
    nsig = RooRealVar('nsig', 'nsig', 231474.29084763874, 0, 500000) 
    ncom = RooRealVar('ncom', 'ncom', 21082.709422450684, 0, 500000)

    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(gauss_sig)
    suma.add(exp)
    
    coeff.add(nsig)
    coeff.add(ncom)
    
    model = R.RooAddPdf('model', 'model', suma, coeff)

    file = f'{folderin}/{file}'

    # define data 
    entries = 0
    filtered = R.RDataFrame(tree, file).Filter(f'({name}>{xmin}) && ({name}<{xmax})') #Take only data between xmin and xmax
    for var in list(filtered.GetColumnNames()):
        filtered = filtered.Filter(f'({var}>0) || ({var}<=0)') #Smart way to remove NaNs
    treee = filtered.Snapshot('DecayTree',f'{folderout}/FilteredTree_data_{name}.root',f'{name}') # 
    filee = R.TFile(f'{folderout}/FilteredTree_data_{name}.root') #Create new file to store filtered tree
    treeecut = filee.Get('DecayTree')
    entries += treeecut.GetEntries()
    ds = RooDataSet('data', 'dataset with x', treeecut, RooArgSet(Bu_Jpsimode_MASS))

    #create and open the canvas
    can = R.TCanvas('hist','hist', 200, 10, 1000, 800)
    pad1 = R.TPad('pad1', 'Histogram', 0., 0.20, 1.0, 0.975, 0)  # Shifted down slightly
    pad2 = R.TPad('pad2', 'Residual plot', 0., 0.10, 1.0, 0.20, 0)  # Centered gap
    can.cd()

    pad1.Draw()
    pad2.Draw()
    ################

    # plot dataset and fit
    massFrame = Bu_Jpsimode_MASS.frame(RooFit.Bins(30)) #in parentheses RooFit.Bins(n)->n bins. Leave blank for auto number of bins
    
    ds.plotOn(massFrame)
 
    fitResults = model.fitTo(ds, RooFit.Save())
    model.plotOn(massFrame,
                 RooFit.Name('curve_model'))


    #Construct the histogram for the residual plot and plot it on pad2
    hresid = massFrame.pullHist()
    chi2 = massFrame.chiSquare()
    hresid.GetYaxis().SetLabelSize(0.25)
    hresid.GetXaxis().SetLabelSize(0)
    hresid.GetXaxis().SetTitle("")
    hresid.GetYaxis().SetTitle("")
    hresid.GetYaxis().SetNdivisions(505)


    pad2.cd()
    hresid.SetTitle('')
    hresid.Draw()
    
    model.plotOn(massFrame, RooFit.Components('gauss'), RooFit.LineColor(3),
                RooFit.Name('gauss'))
    model.plotOn(massFrame, RooFit.Components('exp'), RooFit.LineColor(2),
                RooFit.Name('exp'))
    ds.plotOn(massFrame)

    #Draw the fitted histogram into pad1
    pad1.cd()
    massFrame.SetTitle('')
    massFrame.GetXaxis().SetTitle("m(\\pi^{+}#mu^{+}#mu^{-}) [MeV/c^{2}]")

    # print results
    print('{} has been fit to {} with a chi2 = {}'.format(model.GetName(), file, chi2))
 
    print('Total number of entries is: {}'.format(ds.numEntries()))
    print('Number of sig entries is: {} +- {}'.format(nsig.getValV(),nsig.getError()))
    print(f'Signal entries/total entries is: {nsig.getValV()/ds.numEntries()}')

    massFrame.Draw()

    # create legend
    legend = R.TLegend(0.675, 0.675, 0.9, 0.9)
    legend.SetTextSize(0.045)
    legend.AddEntry('curve_model', 'Total Fit', 'l')
    legend.AddEntry('gauss', 'Signal', 'l')
    legend.AddEntry('exp', 'Combinatorial', 'l')
    legend.Draw()
    legend.Draw()

    # Create a TPaveText for the additional text box
    # extra_info = R.TPaveText(0.45, 0.45, 0.65, 0.60, 'NDC')  # Coordinates in normalized device coordinates (NDC)
    # extra_info.SetFillColor(0)        # Transparent background
    # extra_info.SetFillStyle(0)        # No fill style
    # extra_info.SetLineColor(0)        # No border
    # extra_info.SetBorderSize(0)       # Remove the shadow/border effect
    # extra_info.SetTextAlign(12)       # Align left, vertically centered
    # extra_info.SetTextSize(0.03)      # Text size

    # # Add entries to the text box
    # extra_info.AddText(f'#chi^{{2}} / NDF = {chi2:.2f}')
    # extra_info.AddText(f'Signal yield: {nsig.getValV():.0f} #pm {nsig.getError():.0f}')
    # extra_info.AddText(f'Total entries: {ds.numEntries():.0f}')

    # Draw the TPaveText on the same canvas
    # pad1.cd()  # Switch to the pad where the fit is drawn
    # extra_info.Draw()
    
    #Save the result
    can.SaveAs(f'{folderout}/Fit_fixed_{out}.pdf')

    #####################
    with open(f'{folderout}/Fit_fixed_{out}.txt', 'w') as f:
            f.write('mean_gauss:     '+str(mean_gauss.getValV())+'\n')
            f.write('sigma_gauss:    '+str(sigma_gauss.getValV())+'\n')
            f.write('sigma_gauss err:'+str(sigma_gauss.getError())+'\n')
            f.write('chi2:           '+str(chi2)+'\n')
            f.write('Sig yield:      '+str(nsig.getValV())+'\n')
            f.write('Sig yield err:  '+str(nsig.getError())+'\n')
            f.write('tau:            '+str(tau.getValV())+'\n')
            f.write('tau err:        '+str(tau.getError())+'\n')
            f.write('Com yield:      '+str(ncom.getValV())+'\n')
            f.write('Com yield err:  '+str(ncom.getError())+'\n')
    f.close

    return print(f'Fit_fixed_{out}.pdf done')


# -----------------------------------------------------
# -------------------- GAUSS + BKG --------------------
# -----------------------------------------------------

def megaballFit(file, cuts, folderin, folderout, name, tree, mean_val, xmin, xmax, out):
    # file= name of the .root file with the bdt applied
    # tree= name of the tree inside the file
    # cuts= cuts applied to the tree, "" if none
    # mean_val= initial value of the mean
    # xmin= minimum of the range
    # xmax= maximum of the range
    # name= name for the fit plot

    R.gROOT.SetBatch(True)

    # define variables and pdfs
    Bu_Jpsimode_MASS = RooRealVar(name, name, xmin, xmax) #RooRealVar("Bu_Jpsimode_MASS","Bu_Jpsimode_MASS", xmin, xmax)
    
    # *************** SIGNAL ***************
    mean_gauss  = RooRealVar('Mean', 'mean_gauss', mean_val, mean_val-100, mean_val+100)    
    sigma_gauss = RooRealVar('Sigma', 'sigma_gauss', 36.95859787742036, 10, 50)
    gauss_sig  = RooGaussian('gauss', 'gauss', Bu_Jpsimode_MASS, mean_gauss, sigma_gauss)

    # # *************** SIGNAL WITH SHIFT ***************
    # Bu_M = RooRealVar('Bu_M', 'Bu_M', 5279.41, 5279.41-100, 5279.41+100) #signal centered at Bu_M
    # Bu_M.setConstant()
    # shift = RooRealVar('shift', 'shift', 0, -80, 80)
    # Bu_M_list = RooArgList()
    # Bu_M_list.add(Bu_M)
    # Bu_M_list.add(shift)    
    # mean_gauss  = RooFormulaVar('Mean', 'mean_gauss', 'Bu_M-shift', Bu_M_list)    
    # sigma_gauss = RooRealVar('Sigma', 'sigma_gauss', 36.95859787742036, 10, 50)
    # gauss_sig  = RooGaussian('gauss', 'gauss', Bu_Jpsimode_MASS, mean_gauss, sigma_gauss)

    # *************** COMBINATORIAL ***************
    tau = RooRealVar("tau", "tau", -0.0018410733057335486, -1, 0.)
    exp = RooExponential("exp", "exp", Bu_Jpsimode_MASS, tau)

    tau.setConstant()

    # *************** MISS-ID ***************
    # ******* B+ -> K+ J/PSI (-> e+ e-) ******* ADD
    meanballm  = RooRealVar("Meanm","meanballm",5238.3, 0.01, 100000)
    sigmaballm = RooRealVar("Sigmam", "sigmaballm", 17.5, 0.01, 100000)
    alphaLm    = RooRealVar("AlphaLm", "alphaLm", 0.331, 0.01, 100000)
    nLm        = RooRealVar("nLm", "nLm", 1.74, 0.01, 100000)
    alphaRm    = RooRealVar("AlphaRm", "alphaRm", 1.44, 0.01, 100000)
    nRm        = RooRealVar("nRm", "nRm", 1.77, 0.01, 100000)
    ball_mID  = RooCrystalBall("ballm","ballm",Bu_Jpsimode_MASS,meanballm,sigmaballm,alphaLm,nLm,alphaRm,nRm)

    meanballm.setConstant()
    sigmaballm.setConstant()
    alphaLm.setConstant()
    nLm.setConstant()
    alphaRm.setConstant()
    nRm.setConstant()

    # *************** MISS-ID WITH SIFT ***************
    # ******* B+ -> K+ J/PSI (-> e+ e-) ******* ADD
    # fixmeanm = RooRealVar("fixmeanm","fixmeanm",5238.3, 5238.3-10, 5238.3+10) #original value of mean
    # fixmeanm.setConstant()
    # fixmeanm_list = RooArgList()
    # fixmeanm_list.add(fixmeanm)
    # fixmeanm_list.add(shift)
    # meanballm = RooFormulaVar('MeanM', 'meanballm', 'fixmeanm-shift', fixmeanm_list)
    # sigmaballm = RooRealVar("Sigmam", "sigmaballm", 17.5, 0.01, 100000)
    # alphaLm    = RooRealVar("AlphaLm", "alphaLm", 0.331, 0.01, 100000)
    # nLm        = RooRealVar("nLm", "nLm", 1.74, 0.01, 100000)
    # alphaRm    = RooRealVar("AlphaRm", "alphaRm", 1.44, 0.01, 100000)
    # nRm        = RooRealVar("nRm", "nRm", 1.77, 0.01, 100000)
    # ball_mID  = RooCrystalBall("ballm","ballm",Bu_Jpsimode_MASS,meanballm,sigmaballm,alphaLm,nLm,alphaRm,nRm)

    # meanballm.setConstant()
    # sigmaballm.setConstant()
    # alphaLm.setConstant()
    # nLm.setConstant()
    # alphaRm.setConstant()
    # nRm.setConstant()

    # *************** PART-RECOS ***************
    # ******* B0 -> f0 (-> pi+ pi-) J/PSI (-> e+ e-) ******* 
    meanball1  = RooRealVar("Mean1","meanball1", 5146.81, 0.01, 100000)
    sigmaball1 = RooRealVar("Sigma1", "sigmaball1", 45.89, 0.01, 100000)
    alphaL1    = RooRealVar("AlphaL1", "alphaL1", 0.16199403, 0.01, 100000)
    nL1        = RooRealVar("nL1", "nL1", 108.4, 0.01, 100000)
    alphaR1    = RooRealVar("AlphaR1", "alphaR1", 1.533, 0.01, 100000)
    nR1        = RooRealVar("nR1", "nR1", 2.109, 0.01, 100000)
    ball_pr1  = RooCrystalBall("ball1","ball1",Bu_Jpsimode_MASS,meanball1,sigmaball1,alphaL1,nL1,alphaR1,nR1)

    meanball1.setConstant()
    sigmaball1.setConstant()
    alphaL1.setConstant()
    nL1.setConstant()
    alphaR1.setConstant()
    nR1.setConstant()

    # ******* B0 -> rho0 (-> pi+ pi-) J/PSI (-> e+ e-) *******
    meanball2  = RooRealVar("Mean2","meanball2",5077.4, 0.01, 100000)
    sigmaball2 = RooRealVar("Sigma2", "sigmaball2", 47.46, 0.01, 100000)
    alphaL2    = RooRealVar("AlphaL2", "alphaL2", 0.1901, 0.01, 100000)
    nL2        = RooRealVar("nL2", "nL2", 100.0, 0.01, 100000)
    alphaR2    = RooRealVar("AlphaR2", "alphaR2", 1.532, 0.01, 100000)
    nR2        = RooRealVar("nR2", "nR2", 1.914, 0.01, 100000)
    ball_pr2  = RooCrystalBall("ball2","ball2",Bu_Jpsimode_MASS,meanball2,sigmaball2,alphaL2,nL2,alphaR2,nR2)

    meanball2.setConstant()
    sigmaball2.setConstant()
    alphaL2.setConstant()
    nL2.setConstant()
    alphaR2.setConstant()
    nR2.setConstant()

    # ******* B0 -> K*0 (-> K+ pi+) J/PSI (-> e+ e-) ******* ADD
    meanball3  = RooRealVar("Mean3","meanball3",4561.8, 0.01, 100000)
    sigmaball3 = RooRealVar("Sigma3", "sigmaball3", 96.4, 0.01, 100000)
    alphaL3    = RooRealVar("AlphaL3", "alphaL3", 1.292, 0.01, 100000)
    nL3        = RooRealVar("nL3", "nL3", 21, 0.01, 100000)
    alphaR3    = RooRealVar("AlphaR3", "alphaR3", 2.45, 0.01, 100000)
    nR3        = RooRealVar("nR3", "nR3", 4.8, 0.01, 100000)
    ball_pr3  = RooCrystalBall("ball3","ball3",Bu_Jpsimode_MASS,meanball3,sigmaball3,alphaL3,nL3,alphaR3,nR3)

    meanball3.setConstant()
    sigmaball3.setConstant()
    alphaL3.setConstant()
    nL3.setConstant()
    alphaR3.setConstant()
    nR3.setConstant()
    
    # define coefficients
    nsig = RooRealVar("nsig", "nsig", 1000, 0, 100000)
    ncom = RooRealVar("ncom", "ncom", 1000, 0, 10000)
    nmID = RooRealVar("nmID", "nmID", 1000, 0, 10000)
    # npr1 = RooRealVar("npr1", "npr1", 1000, 0, 10000)
    npr2 = RooRealVar("npr2", "npr2", 1000, 0, 10000)
    npr3 = RooRealVar("npr3", "npr3", 1000, 0, 10000)
    
    # build model
    suma = RooArgList()
    coeff = RooArgList()
    
    suma.add(gauss_sig)
    suma.add(ball_mID) #ADD
    # suma.add(ball_pr1)
    suma.add(ball_pr2) #ADD
    suma.add(ball_pr3) #ADD
    suma.add(exp)
    
    coeff.add(nsig)
    coeff.add(nmID) #ADD
    # coeff.add(npr1)
    coeff.add(npr2) #ADD
    coeff.add(npr3) #ADD
    coeff.add(ncom)
    
    model = R.RooAddPdf("model", "model", suma, coeff)

    file = f"{folderin}/{file}"
    
    # define dataset
    if (cuts==""): 
        file = R.TFile(file)
        treee = file.Get(tree)
    else:
        R.EnableImplicitMT() # Going parallel
        filtered = R.RDataFrame(tree, file).Filter(cuts)
        treee = filtered.Snapshot(tree,f"{folderout}/FilteredTree_{out}.root",name) #"Bu_Jpsimode_MASS"
        file = R.TFile(f"{folderout}/FilteredTree_{out}.root")
        treee = file.Get(tree)
    ds = RooDataSet("data", "dataset with x", treee, RooArgSet(Bu_Jpsimode_MASS))
    

    #create and open the canvas
    can = R.TCanvas("hist","hist", 200,10, 1000, 800)
    pad1 = R.TPad( "pad1", "Histogram",0.,0.15,1.0,1.0,0)
    pad2 = R.TPad( "pad2", "Residual plot",0.,0.05,1.0,0.15,0)
    can.cd()

    pad1.Draw()
    pad2.Draw()
    ################

    # plot dataset and fit
    massFrame = Bu_Jpsimode_MASS.frame()
    
    ds.plotOn(massFrame)
 
    fitResults = model.fitTo(ds)
    model.plotOn(massFrame, RooFit.VisualizeError(fitResults, 1),
                 RooFit.Name("curve_model"))


    #Construct the histogram for the residual plot and plot it on pad2
    hresid = massFrame.pullHist()
    chi2 = massFrame.chiSquare()
    hresid.GetYaxis().SetLabelSize(0.225)
    hresid.GetXaxis().SetLabelSize(0)
    hresid.GetXaxis().SetTitle("")
    hresid.GetYaxis().SetTitle("")
    hresid.GetYaxis().SetNdivisions(506)

    pad2.cd()
    hresid.SetTitle("")
    hresid.Draw()

    ###################
    
    model.plotOn(massFrame, RooFit.Components("gauss"), RooFit.LineColor(3),
                RooFit.VisualizeError(fitResults, 1), RooFit.Name("gauss"))
    model.plotOn(massFrame, RooFit.Components("exp"), RooFit.LineColor(2),
                RooFit.VisualizeError(fitResults, 1), RooFit.Name("exp"))
    model.plotOn(massFrame, RooFit.Components("ballm"), RooFit.LineColor(6),
                RooFit.VisualizeError(fitResults, 1), RooFit.Name("ballm"))
    model.plotOn(massFrame, RooFit.Components("ball2"), RooFit.LineColor(5),
                RooFit.VisualizeError(fitResults, 1), RooFit.Name("ball2"))
    model.plotOn(massFrame, RooFit.Components("ball3"), RooFit.LineColor(7),
                RooFit.VisualizeError(fitResults, 1), RooFit.Name("ball3"))
    model.plotOn(massFrame, RooFit.VisualizeError(fitResults, 1),
               RooFit.Name("curve_model"))
    ds.plotOn(massFrame)

    #Draw the fitted histogram into pad1
    pad1.cd()
    massFrame.SetTitle("")
    # massFrame.GetXaxis().SetTitle("m(\\pi^{+}J/\\psi(\\rightarrow#mu^{+}#mu^{-})) [MeV/c^{2}]")
    massFrame.GetXaxis().SetTitle("m(\\pi^{+}J/#psi) [MeV/c^{2}]")

    # print results
    print("{} has been fit to {} with a chi2 = {}".format(model.GetName(), file, chi2))
 
    print("Total number of entries is: {}".format(ds.numEntries()))
    print("Number of sig entries is: {} +- {}".format(nsig.getValV(),nsig.getError()))

    #sigVal = ufloat(nsig.getValV(), nsig.getError())
    
    #args=RooArgSet(nsig,meanball,sigmaball,alphaL,alphaR,nL,nR)  #all others don't work
    #model.paramOn(massFrame, RooFit.Layout(0.6,.85,.85),RooFit.Parameters(args))
    massFrame.Draw()

    # create legend
    legend = R.TLegend(0.55, 0.55, 0.9, 0.9)
    legend.SetTextSize(0.045)
    legend.AddEntry("curve_model", "Total Fit", "l") 
    legend.AddEntry('gauss', 'B^{+} \\rightarrow \\pi^{+} J/\psi', 'l')
    legend.AddEntry("exp", "Combinatorial", "l")
    legend.AddEntry("ballm", "B^{+} \\rightarrow K^{+} J/\psi", "l")
    # legend.AddEntry("ball1", "B^{0} \\rightarrow f^{0} J/\psi", "l")
    legend.AddEntry("ball2", "B^{0} \\rightarrow \\rho^{0} J/\psi", "l")#ADD
    legend.AddEntry("ball3", "B^{0} \\rightarrow K^{*0} J/\psi", "l")#ADD
    legend.Draw()
    
    #Save the result
    can.SaveAs(f"{folderout}/Fit_fixed_{out}.pdf")
    #####################
    with open(f"{folderout}/Fit_fixed_{out}.txt", 'w') as f:
            f.write('mean_gauss:   '+str(mean_gauss.getValV())+'\n')
            f.write('sigma_gauss:  '+str(sigma_gauss.getValV())+'\n')
            f.write("chi2:       "+str(chi2)+"\n")
            f.write("Sig yield:  "+str(nsig.getValV())+"\n")
            f.write("Sig yield:  "+str(nsig.getError())+"\n")
            f.write("Com yield:  "+str(ncom.getValV())+"\n")
            f.write("Com yield:  "+str(ncom.getError())+"\n")
            f.write("mID yield:  "+str(nmID.getValV())+"\n")
            f.write("mID yield:  "+str(nmID.getError())+"\n")
            # f.write("f0 yield:  "+str(npr1.getValV())+"\n")
            # f.write("f0 yield:  "+str(npr1.getError())+"\n")
            f.write("rho0 yield:  "+str(npr2.getValV())+"\n")#ADD
            f.write("rho0 yield:  "+str(npr2.getError())+"\n")#ADD
            f.write("K*0 yield:  "+str(npr3.getValV())+"\n")#ADD
            f.write("K*0 yield:  "+str(npr3.getError())+"\n")#ADD
            # f.write("shift:  "+str(shift.getError())+"\n")
    f.close

    return print("Fit_%s done" % out)



if __name__=='__main__':
    import os
    import sys
    sys.path.append(os.path.abspath('/home/msalvador/results'))    
    def opendir(folder):

        import os

        if not os.path.exists(os.path.join(os.getcwd(),folder)):
                os.makedirs(os.path.join(os.getcwd(),folder))

        out = print('Out files in folder: ', os.getcwd()+'/'+folder)

        return out
    file     = 'b2pimumu_BDT_jpsi_constr_allcuts.root'
    cuts     = ''
    folderin   = '/home/msalvador/results/tuples'
    opendir(folderin)
    folderout   = '/home/msalvador/results/fits/final/jpsi'
    opendir(folderout)
    name     = 'Bu_DTF_JpsiConstr_MASS' #Bu_DTF_JpsiConstr_MASS
    tree     = 'DecayTree'
    mean_val = 5279.41
    xmin     = 5000
    xmax     = 6000
    out      = f'LHCBstyle_Signal_rare_BDT_allBKG_{xmin}_{xmax}'
    save_fit = True
    # exponentialFit(file, cuts, folderin, folderout, name, tree, xmin, xmax, out, True)
    # gauss_exponentialFit(file, cuts, folderin, folderout, name, tree, mean_val, xmin, xmax, out)
    megaballFit(file, cuts, folderin, folderout, name, tree, mean_val, xmin, xmax, out)