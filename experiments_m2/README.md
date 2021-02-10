This directory contains the scripts to reproduce the results on M2. 
Results are loaded by jupyter notebooks in the `./jupyter/` directory. 

The key results on M2 in our main text can be found in `./jupyter/m2_results.ipynb`. 

# Download the data  

The data is loaded into the `../sdss_stage_dir/` folder. To download the data, type 

```
# change directory to the sdss folder
cd ../sdss_stage_dir 

# download data
./get_sdss_data.sh
```

The Hubble catalog is contained in the `./hubble_data/` folder in this directory. To download,

```
# change directory to the Hubble folder
./hubble_data/

# download Hubble data
wget https://archive.stsci.edu/pub/hlsp/acsggct/ngc7089/hlsp_acsggct_hst_acs-wfc_ngc7089_r.rdviq.cal.adj.zpt

```

# Fit StarNet (optional)

StarNet can be fit by running 

```
./train_wake_sleep.sh
```

which runs 200 epochs of an initial sleep phase on SDSS data and two subsequent wake-sleep cycles. 
On a single GPU, this takes 10-15 minutes. 
The StarNet fits will be saved into the `./fits/` folder. 

The StarNet catalog and comparisons with PCAT, DAOPHOT, and the Hubble catalogs are printed in the jupyter notebook `./jupyter/m2_results.ipynb`. 

(For convenience, pre-fitted networks are stored in the `fits` folder. The notebook will run even without running `./train_wake_sleep.sh`). 


# The DAOPHOT and PCAT catalogs

In the `./jupyter/m2_results.ipynb`, comparisons are made with the PCAT and DAOPHOT catalogs. 
The DAOPHOT catalog is contained in the `./daophot_data/` catalog. 
To download,

```
# change directory to the DAOPHOT folder
cd ./doaphot_data/

# download daophot DAOPHOT
wget http://das.sdss.org/va/osuPhot/v1_0/m2_2583.phot

```

Our run of PCAT came from [this fork](https://github.com/Runjing-Liu120/multiband_pcat/tree/bryan-run-on-m2) of the 
[multiband_pcat](https://github.com/RichardFeder/multiband_pcat) repository. 
After installation of PCAT at that fork, run 

```
./run_pcat.sh
``` 

to obtain results on M2. 

A pre-run MCMC chain saved at this Google Drive [link](https://drive.google.com/file/d/1y6QxxiG6akgPDHVwoGpbf4TYIdMZgyBX/view?usp=sharing). 
A `wget` script is available in the `./fits/` folder to download this chain: 
```
# chage directory to the fits folder, 
cd ./fits/

# download pcat chain 
./get_pcat_chain.sh
```
