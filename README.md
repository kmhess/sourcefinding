# sourcefinding
Various programs/tools for source finding and cleaning of Apertif HI cubes


### Step 1: sourcefinding.py
Does a single source finding run with a 4 sigma smooth+clip cutoff and produces inspection plots of the candidate detections.  (Sidelobes of super bright sources can be identified and ignored in the next step.)

```
python sourcefinding.py -t <taskid> -b <beams> -c <cubes> -o <optionally overwrites continuum filtered & spline fitted files> -j <number of parallel jobs> -n <no spline fittting>
```
Default for beams is 0-39 (all).  
Default for cubes is 1,2,3  (these are the 3 nearest cubes in which we expect direct detections--in 150 MHz processing nomenclature).  

First, does continuum filtering using `template_filtering.par`.  This is separate so it only has to be done once, but source finding can be repeated without recreating the same continuum filtered file over and over again. Second, by default does a spline fitting and subtraction of the filtered cube before source finding. This is parallelized and so done separately so SF can be repeated without recreating it.  Third, does smooth+clip source finding using a default parameter file called `parameter_template_4sig.par`.

After source finding, calls `src/checkmasks.py` to plot the mask from continuum filtering (in grey) and the 2D masks of the source finding (in color). Results for the 3 nearest cubes are plotted together.  Individual cubes are ignored if they aren't present.  Plots the integrated spectrum for all sources.

Output:  
`HI_image_cube*_filtered.fits` - used for plotting & in `finalsources.py`  
`HI_image_cube*_filtered_spline.fits` - used in `clean.py`  
`HI_image_cube*_4sig_cat.txt` - used in `clean.py`  
`HI_image_cube*_4sig_chan.fits`  
`HI_image_cube*_4sig_mask-2d.fits` - used for plotting  
`HI_image_cube*_4sig_mask.fits` - used in `clean.py`  
`HI_image_cube*_4sig_rel.eps `

`HI_imaging_4sig_summary_filtspline.png` - inspection plot of continuum & 2D masks
`HI_imaging_4sig_summary_spec_filtspline.png` - inspection plot of spectra  


\**PLOTS SHOULD BE VISUALLY INSPECTED TO DETERMINE THE RIGHT WAY TO RUN `clean.py`.\**


### Step 2: clean2.py
Repairs spline fitted cube if necessary.  Cleans the HI emission per beam/cube based on input mask files and user source selection.  
```
python clean.py -t <taskid> -b <beams> -c <cubes> -s <sources> -o <optionally overwrites prev clean/model/residual FITS, repaired spline fit files.> -j <number of parallel jobs> -n <no spline fittting> -a <make model and residual cubes as well>
```
Default for beams is 0-39 (all).  
Default for cubes is 1,2,3 (these are the 3 nearest cubes in which we expect direct detections--in 150 MHz processing nomenclature).  
Ignores beam/cube combinations which don't have a SoFiA mask generated by `sourcefinding.py`.

For cubes that contain both real sources and artefacts among the candidates, these cubes should be run individually and source selection should be done from command line with `-s` followed by their source number(s).  Program takes the output mask from SoFiA over the whole cube and cleans within that mask, or user specified regions within that mask (with `-s`) down to 0.5 sigma.

Generates a catalog of what SoFiA sources have been cleaned, for each \*beam*.  (e.g. Results for all cubes are summarized in one file per beam in beam directory.)  If run multiple times, one needs to sort and clean the catalog file by hand.

Output:  
`HI_image_cube*_all_spline.fits` - (default) repaired filtered_spline cube which is then cleaned.  
`HI_image_cube*{_rep}_clean.fits` - cleaned cube, used in `finalsources.py`.  
`HI_image_cube*{_rep}_clean_mask.fits` - mask of only the cleaned sources.  
`HI_image_cube*{_rep}_model.fits` - model cube (optional).    
`HI_image_cube*{_rep}_residual.fits` - residual cube (optional).  

`{rep}_clean_cat.txt` - modified version of SoFiA output catalog, appended with taskid, beam, cube numbers.  Used in `finalsources.py`.

\**OUTPUT NAME VARIES DEPENDING ON SPLINE FITTING INPUT OR NOT.\**

### Step 3: finalsources.py
```
python finalsources.py -t <taskid> -b <beams> -c <cubes> -p <optionally graps panstarrs images (4') instead of DSS2 (6' & adjustable)>
```
Same defaults as above.  
Uses SoFiA-1 `writecubelets` module to create FITS subcubes/maps based on the existing SoFiA masks and cleaned cubes for every CLEANED source (if it meets selection criteria and lives in `clean_cat.txt`).  The output includes all moment maps, a position velocity slice along the SoFiA determined kinematic major axis, and an ascii text file of the integrated spectrum in the 3D mask.

Calculates physical properties of detected sources and stores them in `final_cat.txt` which is stored at the taskid level! (Combines results for all beams/cubes!)  If run multiple times on same beam/cube combos, one needs to sort and clean the catalog file by hand.  SoFiA source name is replaced with the position derived source name.

Creates png images including an HI contour overlay on a DSS2 Blue image, on a PanSTARRS false color image, and on an HI gray scale image; intensity weighted velocity map; postion-velocity map through SoFiA kinematic major axis; and an ascii text file of integrated HI spectrum over 2D mask (includes noise).  All pngs, fits, txt files are saved in beam level directory. BB = beam number; C = cube number; S = SoFiA original source number (combo is unique & can reference back to `clean_cat.txt` and `HI_image_cubeC_4sig_cat.txt`)

PNG maps have flags: Red `!` means map impacted by continuum filtering. Orange `!` means spectrum impacted by fully flagged channels.

Output:  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S.fits` - subcube
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom0.fits` 
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom1.fits` 
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom2.fits` 
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_pv.fits` 
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_specfull.txt` - Integrated spectrum over 2D mask  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_spec.txt` - SoFiA generated spectrum 

`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom0.png` - HI contours on DSS2 Blue image  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom0hi.png`  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom0color.png` - PanSTARRS false color image  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_signif.png` - Pixel-by-pixel SNR    
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_mom1.png`  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_pv.png`  - With 2nd velocity axis.  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_specfull.png` - Integrated spectrum over 2D mask  
`AHC_Jhhmmss.s+ddmmss_TASKID_BB_C_S_spec.png` - SoFiA generated spectrum

`final_cat.txt` - Source catalog with physically derived source properties

\*\*ALL OUTPUT HAS SAME NAME REGARDLESS OF INPUT. UP TO USER TO KEEP TRACK OF WHETHER SPLINE FITTING HAS BEEN DONE OR NOT.\*\*