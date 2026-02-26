#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from scipy.io import savemat, loadmat
from einops import rearrange
from netCDF4 import Dataset


import torch.nn  as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from netCDF4 import Dataset
from torch.utils import data
import pandas as pd
from tqdm import tqdm
from NNarch import ResMLP
import copy

from mpl_toolkits.axes_grid1 import make_axes_locatable          
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt                                  
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
    #model.to(device)

"""
What this colorbar_tight utility does
Attaches a colorbar next to an axis without resizing your figure.
Keeps your plot layout tight and stable.
Allows configurable:
text size
number of ticks
scientific vs normal notation
colorbar position (right/left/top/bottom)
bar size
This is especially useful when generating many figure panels or automated plots in training/evaluation scripts.
If you want, I can also add:
versions for horizontal colorbars,
matching style presets for your bathymetry plots (e.g., vmin/vmax, ocean colormap),
or a helper that attaches multiple colorbars to subplots automatically.
"""

def colorbar_tight(ax,im,fontsize = 20,numTicks = 5,notation=0, pad = 0.05, sizeb = 2, pos = "right"):                                      
    divider = make_axes_locatable(ax)                            
    cax = divider.append_axes(pos, size=str(sizeb)+"%", pad=pad)
    orient = "vertical" if pos == "right" else "horizontal"      
    #cbar=plt.colorbar(im, cax=cax, shrink=0.6, orientation = orient) 
    cbar=plt.colorbar(im, cax = cax, shrink=0.6, extend = 'both' )    
    cbar.ax.tick_params(labelsize=fontsize) 
    tick_locator = ticker.MaxNLocator(nbins=numTicks)
    cbar.locator = tick_locator                                  
    cbar.update_ticks()
    if notation==1:                                              
       cbar.formatter.set_powerlimits((0, 0))                    
       cbar.update_ticks()

"""
Flatten a 2D grid of matplotlib Axes into a simple Python list.

    Parameters
    ----------
    axs : np.ndarray or list of lists
        2D array-like structure of Axes objects, typically returned from:
            fig, axs = plt.subplots(m, n)
    m : int
        Number of rows.
    n : int
        Number of columns.

    Returns
    -------
    list
        A flat list of Axes, ordered row-by-row.
"""

def axesList(axs, m, n):
    axl = []
    for i in range(m):
        for j in range(n):
           axl.append(axs[i, j])     
    return axl
#############################################
def unnormDf(df, mean, std):
    """
    Undo normalization on a DataFrame:
        x = x_norm * std + mean
    """
    return df * std + mean


def normDf(df, mean, std):
    """
    Normalize a DataFrame:
        x_norm = (x - mean) / std
    """
    return (df - mean) / std

#############################################
def normDict(v, m, s, var):
    """
    Normalize a scalar or array 'v' that belongs to variable 'var',
    using mean and std dictionaries m and s.

        v_norm = (v - m[var]) / s[var]

    Parameters
    ----------
    v : numeric / np.ndarray / torch.Tensor
        Value to normalize.
    m : dict
        Dictionary of means keyed by variable name.
    s : dict
        Dictionary of std deviations keyed by variable name.
    var : str
        Key for selecting correct mean/std from dicts.
    """
    return (v - m[var]) / s[var]


def unnormDict(v, m, s, var):
    """
    Undo normalization applied via normDict:

        v_unnorm = v * s[var] + m[var]
    """
    return v * s[var] + m[var]

def ftype(fname):
    """
    Return the file extension of a filename.

    Parameters
    ----------
    fname : str
        Filename including extension, e.g. 'image.png'

    Returns
    -------
    str
        The substring after the final '.', e.g. 'png'.
    """
    ext = fname.split('.')[-1]
    return ext
#############################################
"""
Optional improvements (if you want them)
I can provide versions that:
infer dtype automatically (float16/32/64, int types, bool)
compute exact memory in bytes/KB/MB/GB
handle PyTorch tensors on CUDA
return per-element size information
include batch dimensions & overhead for models
"""

def sizeAr(data):
    """
    Estimate the memory footprint of a numeric array in megabytes (MB),
    assuming 32-bit (4-byte) floating-point values.

    Parameters
    ----------
    data : array-like (NumPy array, torch.Tensor, etc.)
        Any object that supports `.flatten()` and `len()`.

    Returns
    -------
    float
        Approximate size in megabytes (MB), computed as:
            N_elements * 32 bits / 1024^2
        where 32 bits = 4 bytes per value.
    """
    # Total number of elements in the array
    N = len(data.flatten())

    # Convert: (N elements * 32 bits) → megabytes
    return (N * 32) / 1024**2

#############################################
"""
What this function accomplishes
It produces a clean, standardized feature matrix, ideal for training MLPs or U-Nets:
Log-transformed buoyancy frequency (LN2)
Normalized physical gradients (dSdz, dTdz)
Properly scaled latitudes and depths
Ordered feature vector of fixed size
Efficient float32 format
Automatic file type detection (CSV / MAT)
Reliable normalization pipeline
Perfect for large atmospheric/oceanographic ML datasets.

Optional upgrades (if you want them)
I can extend this function with:
automatic dtype and memory reporting
nan-handling and outlier clipping
column existence checking
batching for extremely large CSV/MAT files
returning both normalized and unnormalized frames
allowing arbitrary variable ordering
"""

def genMatCsv(fname, mean, std, Zmin):
    """
    Load a dataset from CSV or MAT, normalize selected variables,
    apply simple scaling to latitude and depth, and return the result
    as a float32 NumPy array suitable for ML models.

    Processing Steps
    ----------------
    1. Load file:
        - If extension is '.csv' → load with pandas.
        - If extension is '.mat' → convert using matToDf() (user-defined).

    2. Compute LN2 = log10(N2)

    3. Normalize the variables:
        ['hab', 'S', 'T', 'dSdz', 'dTdz', 'LN2', 'N2']
        using:
            x_norm = (x - mean[var]) / std[var]

    4. Scale geographic/depth variables:
        lat  → lat / 90       (map latitude to [-1, 1] or [0,1] depending on input)
        Z    → Z / Zmin       (depth normalization)

    5. Reorder columns into a fixed expected order

    6. Convert to float32 NumPy array and report memory footprint.

    Parameters
    ----------
    fname : str
        Path to a CSV or MAT file.
    mean : dict
        Dictionary of means per variable (same keys as normalized vars).
    std : dict
        Dictionary of standard deviations per variable.
    Zmin : float
        Normalizing constant for Z.

    Returns
    -------
    np.ndarray (float32)
        Normalized data matrix ready for ML models.
    """

    # ------------------------------------
    # 1. Determine file type and load data
    # ------------------------------------
    ext = fname.split('.')[-1]
    print(ext)

    if ext.lower() == 'csv':
        df0 = pd.read_csv(fname)

    elif ext.lower() == 'mat':
        # BUG FIX: previously matfile was undefined.
        # Now fname is passed correctly.
        df0 = matToDf(fname)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Make a working copy
    df = df0.copy()

    # ------------------------------------
    # 2. Compute LN2 = log10(N2)
    # ------------------------------------
    df['LN2'] = np.log10(df0['N2'])

    # ------------------------------------
    # 3. Normalize selected physical variables
    # ------------------------------------
    vars_to_norm = ['hab', 'S', 'T', 'dSdz', 'dTdz', 'LN2', 'N2']

    for var in vars_to_norm:
        df[var] = normDf(df[var], mean[var], std[var])

    # ------------------------------------
    # 4. Normalize latitude & depth separately
    # ------------------------------------
    df['lat'] = df0['lat'] / 90.0        # scale lat into ~[-1,1] or [0,1]
    df['Z']   = df0['Z']   / Zmin        # depth scaling

    # ------------------------------------
    # 5. Reorder columns into a fixed ML-ready order
    # ------------------------------------
    df = df[['hab', 'S', 'T', 'dSdz', 'dTdz', 'LN2', 'N2', 'lat', 'Z']]

    # ------------------------------------
    # 6. Convert to float32 matrix
    # ------------------------------------
    indata = df.values.astype(np.float32)

    # ------------------------------------
    # 7. Report size
    # ------------------------------------
    print(f"Input data shape: {indata.shape}, size {np.floor(sizeAr(indata))} MB")

    return indata

"""
What this function does well
Converts MATLAB data structures into a clean pandas DataFrame
Standardizes column names by removing suffixes
Removes all rows containing NaN values
Provides a simple, robust pathway for .mat → CSV → ML workflows
✔️ Optional Enhancements (if you want them)
I can extend this function to:
automatically detect which .mat keys are numeric arrays
handle nested MATLAB structs
support .mat files saved with -v7.3 (HDF5)
produce a log of dropped rows
keep both the “raw” and cleaned DataFrame
"""

def matToDf(matfile):
    """
    Load a MATLAB .mat file and convert selected variables into a pandas DataFrame.

    Assumes the .mat file contains variables with names following a pattern such as:
        ['hab_1', 'S_1', 'T_1', ...]
    and extracts the variable *base names* by taking the substring before the first '_'.

    Example:
        MAT variable names:     ['hab_1', 'S_1', 'T_1']
        Extracted DataFrame columns: ['hab', 'S', 'T']

    Steps
    -----
    1. Load the .mat file via scipy.io.loadmat()
    2. Ignore the first three keys (MATLAB metadata entries: __header__, __version__, __globals__)
    3. Extract base variable names (before the underscore) as DataFrame column names
    4. Copy each variable array into the DataFrame, squeezing to 1D
    5. Count and drop rows containing NaNs
    6. Return a clean DataFrame

    Parameters
    ----------
    matfile : str
        Path to a .mat file.

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame with one column per extracted variable.
    """

    # ----------------------------------------
    # 1. Load .mat file (as a dictionary)
    # ----------------------------------------
    aa = io.loadmat(matfile)

    # ----------------------------------------
    # 2. Extract variable names
    #    MATLAB stores metadata in the first 3 keys
    # ----------------------------------------
    varlisti = list(aa.keys())[3:]  # skip __header__, __version__, __globals__

    # ----------------------------------------
    # 3. Convert variable names like 'T_1' → 'T'
    # ----------------------------------------
    varlisto = [var.split('_')[0] for var in varlisti]

    # ----------------------------------------
    # 4. Create DataFrame with these column names
    # ----------------------------------------
    df = pd.DataFrame(columns=varlisto)

    # ----------------------------------------
    # 5. Populate DataFrame with the MAT arrays
    # ----------------------------------------
    for vari, varo in zip(varlisti, varlisto):
        # aa[vari] is typically a MATLAB Nx1 or 1xN array → squeeze into 1D
        df[varo] = aa[vari].squeeze()

    # ----------------------------------------
    # 6. Report and remove NaNs
    # ----------------------------------------
    print('Num NaNs:', np.sum(df.isna()))

    # Drop rows where any NaN appears
    df.dropna(inplace=True)

    return df

#############################################
def getNcData(
    fname,
    mean,
    std,
    zmin=-5515,
    varlist=['hab', 'S', 'T', 'dSdZ', 'dTdZ', 'LN2', 'N2', 'lat', 'Z']
):
    """
    Load variables from a NetCDF file, normalize them, flatten to 2D,
    and return a feature matrix suitable for ML models.

    Workflow
    --------
    1. Open NetCDF file.
    2. Read variables in `varlist` into a dictionary (valdict).
    3. Fix N2:
         - Values <= 0 are set to NaN.
         - Compute LN2 = log10(N2).
    4. Ensure all variables have the same spatial shape as N2:
         - Broadcast lower-dimensional variables where needed.
    5. Normalize variables:
         - Z   → Z / zmin
         - lat → lat / 90
         - all others via normDict(varval, mean, std, var)
           with a small fallback hack if variable names differ slightly.
    6. Flatten each (H, W) variable into (H*W,) and stack into a 2D array (N, D):
         - N = H * W
         - D = len(varlist)
    7. Print final shape & approximate memory size.

    Parameters
    ----------
    fname : str
        Path to the NetCDF file.
    mean : dict
        Dictionary mapping variable names → mean values (for normalization).
    std : dict
        Dictionary mapping variable names → std values (for normalization).
    zmin : float, optional (default = -5515)
        Normalization constant for depth Z (Z / zmin).
    varlist : list of str, optional
        List of variables to read and process, in output feature order.

    Returns
    -------
    indata : np.ndarray, shape (H*W, len(varlist)), dtype float32
        Flattened and normalized feature matrix.
    H : int
        Number of grid points in first spatial dimension.
    W : int
        Number of grid points in second spatial dimension.
    """
    # Open the NetCDF file in read mode
    nci = Dataset(fname, 'r')

    inlist = []   # list of 1D flattened feature columns
    valdict = {}  # raw (or partly processed) variable arrays

    print(varlist)

    # -----------------------------
    # 1. Read variables into valdict
    # -----------------------------
    for ivar, var in enumerate(varlist):
        try:
            # Read full variable, squeeze to remove length-1 dimensions,
            # and cast to float32
            valdict[var] = nci.variables[var][...].squeeze().data.astype(np.float32)
        except Exception:
            print(f'Skipping reading {var} from netcdf file')

    # -----------------------------
    # 2. Post-process N2 and LN2
    # -----------------------------
    # Set non-positive N2 values to NaN (log is undefined for <= 0)
    valdict['N2'][valdict['N2'] <= 0] = np.nan

    # Compute LN2 = log10(N2)
    valdict['LN2'] = np.log10(valdict['N2'])

    # Assume N2 defines the reference spatial shape (H, W)
    shape = valdict['N2'].shape   # e.g., (H, W)

    # -----------------------------
    # 3. Broadcast and normalize each variable
    # -----------------------------
    for ivar, var in enumerate(varlist):
        varval = valdict[var]

        # Broadcast lower-dimensional variables up to full spatial shape
        # so that everything matches the shape of N2.
        #
        # Example:
        #   - If varval has shape (H,) and shape is (H, W),
        #     broadcast to (H, 1) → (H, W)
        #   - If varval has shape () or (1,), broadcast to (H, W)
        if len(varval.shape) == len(shape) - 1:
            # e.g. varval: (H,) or (W,) vs shape: (H, W)
            varval = np.broadcast_to(varval[..., None], shape)
        elif len(varval.shape) == len(shape) - 2:
            # e.g. varval: scalar () vs shape: (H, W)
            varval = np.broadcast_to(varval[..., None, None], shape)

        # Sanity check: now we expect at least 2D
        assert (len(varval.shape) > 1)

        # -----------------------------
        # 4. Variable-specific normalization
        # -----------------------------
        if var == 'Z':
            # Depth normalization
            varval = varval / zmin
        elif var == 'lat':
            # Latitude normalization (assuming degrees → scale by 90)
            varval = varval / 90.0
        else:
            # Physical variables: use dict-based normalization
            try:
                varval = normDict(varval, mean, std, var)
            except Exception:
                # Fallback: try a slightly modified name if, for example,
                # the dictionary uses 'dSdZ' vs 'dSdz', etc.
                print(f"Can't seem to normalize {var}. Trying some hackery")
                var_alt = var[:-1] + 'z'
                varval = normDict(varval, mean, std, var_alt)
                print('... which worked')

        # -----------------------------
        # 5. Flatten variable to 1D and collect
        # -----------------------------
        # Rearrange from 2D (H, W) → 1D (H*W)
        varval1d = rearrange(varval, 'h w -> (h w)')

        # Add as a column vector (N, 1)
        inlist.append(varval1d[:, None])

    # -----------------------------
    # 6. Stack feature columns into final matrix
    # -----------------------------
    indata = np.concatenate(inlist, axis=1)   # shape: (H*W, len(varlist))

    # -----------------------------
    # 7. Report size
    # -----------------------------
    print(f'Input data shape: {indata.shape}, size {np.floor(sizeAr(indata))} MB')

    # Return data matrix and original spatial dimensions
    return indata, shape[0], shape[1]

"""
If you want, I can also write:
a version that validates that both CSVs contain matching variables,
a version that loads from a single CSV (mean & std columns),
or a pretty-print summary of available stats.
"""

def getStats(fmean='mean.csv', fstd='std.csv'):
    """
    Load mean and standard deviation statistics from two CSV files
    and return them as Python dictionaries.

    The CSV files are expected to have the structure:
        variable, value

    Example mean.csv:
        S, 34.7
        T, 12.9
        N2, 0.0013
        ...

    Parameters
    ----------
    fmean : str, default 'mean.csv'
        Path to CSV file containing mean values.
    fstd  : str, default 'std.csv'
        Path to CSV file containing std values.

    Returns
    -------
    meandict : dict
        Mapping from variable name → mean
    stddict : dict
        Mapping from variable name → std deviation
    """

    # Read CSVs
    mean_df = pd.read_csv(fmean)
    std_df  = pd.read_csv(fstd)

    # mean_df.T.values:
    #    row 0 → variable names
    #    row 1 → mean values
    meandict = dict(zip(mean_df.T.values[0], mean_df.T.values[1]))

    # same logic for std
    stddict = dict(zip(std_df.T.values[0], std_df.T.values[1]))

    return meandict, stddict

#############################################
"""
What this function does
Constructs 10 ResMLP models, each with the same architecture.
Loads pretrained weights from filenames that encode:
number of layers
hidden size
learning rate
batch size
fold index
Moves each model to the correct device (cpu or GPU).
Puts them into inference mode (model.eval()).
Returns a Python list shaped like:
[
  model_fold_0,
  model_fold_1,
  ...,
  model_fold_9
]
This is your ensemble predictor.
✔️ Optional improvements (if you want them)
I can easily add:
✓ Automatic detection of available models
(no need to hard-code 10 folds)
✓ Error handling if a fold file is missing
✓ A wrapper to compute predictions using the ensemble
(e.g., mean, median, spread per prediction)
✓ Support for multiple architectures (e.g. passing model class as argument)
✓ A version returning Δfields reshaped to spatial grids (H,W)
"""

def loadModelEns(dr='./10fold', lr0=0.0035, bsize=1000,
                 numLayers=3, nhidden=120):
    """
    Load a 10-model ensemble of ResMLP networks, each corresponding
    to one fold in a 10-fold cross-validation setup.

    Model filenames are expected to follow the pattern:
        {dr}/ResMLP_{numLayers}_{nhidden}_{lr0}_{bsize}_{i}.pth
    where i = 0, 1, ..., 9

    Parameters
    ----------
    dr : str, default './10fold'
        Directory containing the saved model .pth files.
    lr0 : float, default 0.0035
        Learning rate used during training (encoded in filename).
    bsize : int, default 1000
        Batch size used during training (encoded in filename).
    numLayers : int, default 3
        Number of residual FcLayers used inside ResMLP.
    nhidden : int, default 120
        Hidden dimension inside ResMLP.

    Returns
    -------
    list of ResMLP
        A list containing 10 ResMLP models, each loaded with
        pretrained weights and set to evaluation mode.
    """

    model_fold = []

    # Your ML setup uses 9 predictors → ninp = 9
    # and 2 target variables → nout = 2
    numPredictors, numTargets = 9, 2

    # Load 10 models corresponding to a 10-fold CV ensemble
    for imod in tqdm(range(10)):
        PATH = dr + f'/ResMLP_{numLayers}_{nhidden}_{lr0}_{bsize}_{imod}.pth'

        # Initialize a fresh model with the correct architecture
        model = ResMLP(
            numLayers=numLayers,
            nhidden=nhidden,
            ninp=numPredictors,
            nout=numTargets
        ).to(device)

        # Load pretrained parameters
        model.load_state_dict(torch.load(PATH, map_location=device))

        # Set in evaluation mode (no dropout, no BN updates)
        model.eval()

        model_fold.append(model)

    return model_fold


def predModelNC(indata, mean, std, model_fold, h, w):
    """
    Run an ensemble of models on flattened NetCDF input data and
    return un-normalized 2D fields for LK and Leps.

    Workflow
    --------
    1. Loop over each model in the ensemble (model_fold).
    2. Convert `indata` (N x D) to a torch.Tensor on `device`.
    3. Run a forward pass to get outputs:
           out.shape = (N, 2)
           out[:, 0] = LK (normalized)
           out[:, 1] = Leps (normalized)
    4. Reshape each output vector from (H*W,) → (H, W) using einops.rearrange.
    5. Un-normalize LK and Leps using `unnormDict` with keys 'LK' and 'Leps'.
    6. Collect ensemble members in lists, then stack to get arrays of shape:
           LK.shape   = (n_models, H, W)
           Leps.shape = (n_models, H, W)

    Parameters
    ----------
    indata : np.ndarray, shape (H*W, D)
        Flattened, normalized input feature matrix (float32).
        Typically the output of `getNcData`.
    mean : dict
        Dictionary of means per variable, used here for 'LK' and 'Leps'.
    std : dict
        Dictionary of standard deviations per variable, used here for 'LK' and 'Leps'.
    model_fold : list of torch.nn.Module
        List of ensemble models (e.g. from loadModelEns).
    h : int
        Height of the original spatial grid (first dimension).
    w : int
        Width of the original spatial grid (second dimension).

    Returns
    -------
    LK : np.ndarray, shape (n_models, H, W)
        Un-normalized LK field for each ensemble member.
    Leps : np.ndarray, shape (n_models, H, W)
        Un-normalized Leps field for each ensemble member.
    """
    LKl = []    # list to store LK maps from each ensemble model
    Lepsl = []  # list to store Leps maps from each ensemble model

    # No gradients needed during inference
    with torch.no_grad():
        for model in tqdm(model_fold):
            # Ensure input is a float32 tensor on correct device
            x = torch.tensor(indata.astype(np.float32)).to(device)

            # Forward pass through the current model
            temp = model(x)

            # Move output back to CPU and convert to NumPy
            out = temp.detach().cpu().numpy()  # shape: (H*W, 2)

            # Split output into LK and Leps (still flattened)
            # out[:, 0] → LK, out[:, 1] → Leps
            LK_flat   = out[:, 0]
            Leps_flat = out[:, 1]

            # Reshape from (H*W,) → (H, W)
            LK   = rearrange(LK_flat,   '(h w) -> h w', h=h, w=w)
            Leps = rearrange(Leps_flat, '(h w) -> h w', h=h, w=w)

            # Un-normalize LK and Leps using stats dictionaries
            LK_unnorm   = unnormDict(LK,   mean, std, 'LK')
            Leps_unnorm = unnormDict(Leps, mean, std, 'Leps')

            # Append to ensemble lists
            LKl.append(LK_unnorm)
            Lepsl.append(Leps_unnorm)

    # Stack into arrays:
    #   LK.shape   = (n_models, H, W)
    #   Leps.shape = (n_models, H, W)
    LK   = np.array(LKl)
    Leps = np.array(Lepsl)

    return LK, Leps

"""
If you want, I can also extend this function to:
reshape results back into (H, W) grids and save NetCDF,
write separate CSVs for mean and error,
compute ensemble percentiles (5th, 95th),
or directly generate plots of ensemble spread.
"""

def writeDf(model_fold, indata, mean, std, fname):
    """
    Run an ensemble of models on flattened input data, unnormalize the
    outputs (LK and Leps), compute ensemble mean and ensemble std, and
    write them to a CSV file.

    Ensemble outputs:
        LK      : array (n_models, N)
        Leps    : array (n_models, N)
        LKm     : ensemble mean    of LK
        LKerr   : ensemble std dev of LK
        Lepsm   : ensemble mean    of Leps
        Lepserr : ensemble std dev of Leps

    Parameters
    ----------
    model_fold : list of torch.nn.Module
        Ensemble of trained models (e.g. from loadModelEns()).
    indata : np.ndarray, shape (N, D)
        Input features (already normalized and flattened).
    mean : dict
        Dictionary of means for denormalizing predictions.
    std : dict
        Dictionary of std devs for denormalizing predictions.
    fname : str
        Filename for output CSV.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the ensemble mean and error estimates.
    """

    LKl = []      # list of LK outputs per model
    Lepsl = []    # list of Leps outputs per model

    with torch.no_grad():
        for model in tqdm(model_fold):
            # Convert input to tensor on the correct device
            X = torch.tensor(indata.astype(np.float32)).to(device)

            # Forward pass → shape (N, 2)
            Y = model(X).detach().cpu().numpy()

            # Unnormalize outputs
            LK   = unnormDict(Y[:, 0], mean, std, 'LK')
            Leps = unnormDict(Y[:, 1], mean, std, 'Leps')

            LKl.append(LK)
            Lepsl.append(Leps)

    # Stack across ensemble dimension
    LK   = np.array(LKl)     # shape (n_models, N)
    Leps = np.array(Lepsl)   # shape (n_models, N)

    # Create DataFrame — fix the column names here:
    df = pd.DataFrame()

    # Ensemble means
    df['LKm']   = np.mean(LK, axis=0)
    df['Lepsm'] = np.mean(Leps, axis=0)

    # Ensemble uncertainties (std deviation)
    df['LKerr']   = np.std(LK, axis=0)
    df['Lepserr'] = np.std(Leps, axis=0)

    # Save
    df.to_csv(fname, index=False)

    return df


# In[ ]:





# In[ ]:




