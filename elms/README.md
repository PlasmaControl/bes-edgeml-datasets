# Workflow for generating ELM dataset

OMFIT scripts discussed below are available under `elm-dataset-tools`
in this OMFIT project:
`/fusion/projects/diagnostics/bes/smithdr/omfit/edgeml-dataset-tools.zip`


1) Run OMFIT script `make_shotlist.py` to perform D3DRDB query to get ELM
-relevant shots.  Returns ~ 1500 shots.  Copy output file to `data/shot-list.csv`.

2) Package BES metadata for ELM-relevant shotlist using `package_metadata()` 
in `elms.py`. Submit a batch job using `package-metadata.slurm.sh`.  
Gets metadata for ~ 1000 shots with BES data. Runtime ~ 1.5 hours.  when happy 
with results, directory to `data/metadata/`.
 
3) Package BES metadata and signals for subset of shots with BES in 8x8
 configuration.  Use `package_signals_8x8_only()` in `elms.py`. Submit a batch job
 using `package-signals-8x8-only.slurm.sh`.  Returns ~ 350 shots with specified 8x8 BES data.
 Runtime ~ 11 hours.  When happy with results, directory to 
 `data/signals-8x8-only/`.

4) Run OMFIT script `make-elm-list.py` to generate approximate timestamps for
 ELM events. Link metadata file `data/signals-8x8-only/bes_metadata.hdf5` in
 OMFIT tree as `8x8-metadata`. Runtime ~ 1 hour. Copy output file to 
 `data/elm-list.csv`.

5) Using BES signals from 3) and approximate ELM timestamps from 4), create a
dataset with only ELM-relevent time windows by running 
`package-unlabeled-elm-events()` in `elms.py`.  
Submit batch job with `package-unlabeled-elm-events.sbatch.sh`.
Runtime ~ 1 hour.  When happy with results, rename directory to 
`data/unlabeled-elm-events/`.

6) Run the ELM labeler tool in `../elm-labeling-tool/elm-labeler.py`.
The user selects timeslices for 'no elm', 'start elm', 'stop elm', and 'end'.


Shot count and down selection

1512 ELMy shots returned by D3DRDB query (see `elm-shotslist.csv`)
976 shots have valid BES data
437 shots have 8x8 BES data
348 shots have 8x8 BES data within an r/z window
