# Workflow for generating ELM dataset

OMFIT scripts discussed below are available under `elm-tools` in this OMFIT project:
`/fusion/projects/diagnostics/bes/smithdr/omfit/edgeml-dataset-tools.zip`

Workflow steps:

1) Use OMFIT to perform D3DRDB query to get ELM-relevant shots.
Under `elm_tools` in OMFIT project, run `step_1_make_shotlist.py`.
Returns ~ 1500 shots.
Copy output file to `data/step_1_shotlist.csv`.

2) Run batch job and python script to get metadata for ~ 1000 shots with BES data.
Submit batch job `step_2_package_metadata.sbatch.sh` which runs `bes_data_tools.package_h5.package_bes()`.
Runtime ~ 1.5 hours.
When happy with results, copy to `data/step_2_metadata.hdf5`.
 
3) Run batch job and python script to package metadata and BES signals 
for shots with valid 8x8 BES configuration.
Submit batch job `step_3_package_signals_8x8_only.sbatch.sh` which runs `bes_data_tools.package_h5.package_signals_8x8()`.
Returns ~ 350 shots with specified 8x8 BES data.
Runtime ~ 11 hours.
When happy with results, move directory to `data/step_3_signals_8x8_only/`.

4) Use OMFIT to generate approximate timestamps for ELM events.
Link metadata file `data/step_3_signals_8x8_only/step_3_metadata.hdf5`
in OMFIT tree as `step_3_8x8_metadata`.
Run OMFIT script `step_4_make_elm_list.py`.
Runtime ~ 1 hour.
Copy output file to  `data/step_4_elm_list.csv`.

5) Run batch job and python script to create unlabeled dataset with unique ELM events.
Submit batch job with `step_5_package-unlabeled-elm-events.sbatch.sh` which runs 
`package-unlabeled-elm-events()` in `elms.py`.
Runtime ~ 1 hour.
When happy with results, rename directory to `data/step_5_unlabeled_elm_events/`.

6) Run the ELM labeler tool in `step_6_labeling_tool/elm-labeler.py`.
The user selects timeslices for 'no elm', 'start elm', 'stop elm', and 'end'.


Shot count and down selection

1512 ELMy shots returned by D3DRDB query (see `elm-shotslist.csv`)
976 shots have valid BES data
437 shots have 8x8 BES data
348 shots have 8x8 BES data within an r/z window

# `bes_data_tools` package

`bes_data_tools` is a package for fetching BES metadata and signals,
storing the data in HDF5 files, and visualizing results.

`bes_data.py` defines a class that fetches and stores BES metadata and signals.

`package_h5.py` defines functions that packages BES metadata and signals
in HDF5 files.  Also, it can analyze metadata to determine BES data
with an 8x8 spatial configuration.

`plots.py` contains utility plotting functions.

By default, data is read from and written to `data/`, and figures are saved
to `figures/`.  These directories are created locally wherever the package or 
modules are imported.  Internally, data files like `shotlist.csv` are interpreted as
`data/shotlist.csv`, and figures like `plot.pdf` are interpreted as
`figures/plots.pdf`.