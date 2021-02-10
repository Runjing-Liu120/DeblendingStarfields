This folder contains the script to run StarNet on SDSS run 94 camcol 1 field 12. 

The SDSS data is located in the `../sdss_stage_dir/` folder. To download the data, type 

```
# change directory to the sdss folder
cd ../sdss_stage_dir 

# download data
./get_sdss_data.sh
```

The `coadd_field_catalog_runjing_liu.fit` contains the catalog constructed from a co-added version of this SDSS image.
We use the catalog contained in this fit file as a ground truth. 


To train the encoder, run 
```
python train_sleep-sparse_field.py 
```

The notebook `./jupyter/sparse_field_results.ipynb` loads the fitted neural network and produces the figures in Appendix B. 