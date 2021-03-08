# Workflow for generating ELM dataset

1) Run OMFIT script `make_shotlist.py` to perform D3DRDB query to get ELM
-relevant shots.  Returns ~ 1500 shots.  Copy output file to `shotlist.csv` in this directory.

2) Package BES metadata for ELM-relevant shotlist using
 `package_shotlist_metadata()` in `elms.py`. Submit a batch job using
 `package_elms.slurm.sh`.  Gets metadata for ~ 1000 shots with BES data.
 Runtime ~ 1.5 hours.  When job completes, rename job 
 directory to `data/metadata_full_shotlist/`.
 
3) Package BES metadata and signals for subset of shots with BES in 8x8
 configuration.  Use `package_8x8_sublist()` in `elms.py`. Submit a batch job
 using `package_elms.slurm.sh`.  Returns ~ 350 shots with specified 8x8 BES data.
 Runtime ~ 11 hours.  When job completes, rename 
 job directory to `data/signals_8x8_only/`.

4) Run OMFIT script `make-elm-list.py` to generate approximate timestamps for
 ELM events.  Copy output file to `elm-list.csv` in this directory.

5) Using BES signals from 3) and approximate ELM timestamps from 4), create a
dataset with only ELM-relevent times by running `package_elm_events()` in 
`elms.py`.  Submit batch job with `package_elm_events.sbatch.sh`.
Runtime ~ 3.5 hours.  When job completes, rename job directory to 
`data/elm_events/`.

6) Run the ELM labeler in `elm-labeler.py`.  The user selects timeslices for 
'no elm', 'start elm', 'stop elm', and 'end'.  The data and labels are saved.


OMFIT scripts are available in OMFIT project:
`/fusion/projects/diagnostics/bes/smithdr/omfit/edgeml-dataset-tools.zip`


Shot count and down selection

1512 ELMy shots returned by D3DRDB query (see `elm-shotslist.csv`)
976 shots have valid BES data
437 shots have 8x8 BES data
348 shots have 8x8 BES data within an r/z window
