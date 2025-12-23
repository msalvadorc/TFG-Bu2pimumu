# Study of the $B^+\to\pi^+\mu^+\mu^-$ decay with the LHCb experiment

This is the code for my final degree project titled Study of the $B^+\to\pi^+\mu^+\mu^-$ decay with the LHCb experiment, which was presented in June 2025.

The project consisted on applying Machine Learning (ML) techniques to select events corresponding to the $B^+\to\pi^+\mu^+\mu^-$ decay from a fraction of the data collected at the LHCb experiment during 2024. The full report for this project is available in `TFG-Salvador-Carreté-Marc.pdf`.

## The code
The code works using 4 python files. The main file `BDT_main.py` first applies a preselection to the data and then applies a Boosted Decision Tree (BDT) to select the desired events (see the project report `TFG-Salvador-Carreté-Marc.pdf` for details). To do this, it uses functions in the `functions.py`, `Bu_alt_M.py` and `fits.py` files. The first of these files contains general useful functions (mainly plotting functions and saving data to a `.ROOT` file). The file `Bu_alt_M.py` applies a special relativity calculation to calculate the invariant mass of the muon pair changing the mass hypothesis (for example, considering a pair of pions insetad of a pair of muons) and applies some vetoes to these masses. Finally, the `fits.py` file -which is an adaptation of code originally made by Ernest Olivart (eolivart@icc.ub.edu)- performs fits to the different contributions to the data by using different curve shapes for each distribution (again, see the project report `TFG-Salvador-Carreté-Marc.pdf` for details).

## Running the code
The code uses the following set of libraries:
- `ROOT`
- `numpy`
- `pandas`
- `matplotlib.pyplot`
- `time` (this one is only used to track runtime, so it may be skipped)
- `xgboost`
- `pickle`
- `sklearn.model_selection`
- `os`

Unfortunately, this code requires access to the LHCb data, which is not public, so it cannot be directly run as is.
