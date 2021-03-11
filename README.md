# BES edge-ml

Code base for BES edge-ML at DIII-D

- `edgeml/` - Module for packaging BES data and metadata in HDF5.
Do `from edgeml import bes2hdf5`.  Consider adding `<repo-dir>/edgeml/' to
`PYTHONPATH`.
- `elm-dataset-workflow/` - Python and shell scripts to generate ELM-relevant
shotlists and elm candidates
- `elm-labeling-tool/` - GUI for interactively labeling ELM events
- `elm-training/` - NN models for ELM classification
- `qhmode/` - work area for QH-mode classification
- `ae/` - work area for AE classification
