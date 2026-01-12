#!/usr/bin/env bash
#SBATCH --account=m3586
#SBATCH --constraint=cpu
#SBATCH --mail-user=kevin.gill@wisc.edu
#SBATCH --mail-type=ALL

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128

#SBATCH --nodes=1
#SBATCH --time=06:30:00
#SBATCH --qos=regular
###SBATCH --array=0

echo Python executable: $(which python)
echo
echo Job name: $SLURM_JOB_NAME
echo QOS: $SLURM_JOB_QOS
echo Account: $SLURM_JOB_ACCOUNT
echo Submit dir: $SLURM_SUBMIT_DIR
echo
echo Job array ID: $SLURM_ARRAY_JOB_ID
echo Job ID: $SLURM_JOBID
echo Job array task: $SLURM_ARRAY_TASK_ID
echo Job array task count: $SLURM_ARRAY_TASK_COUNT
echo
echo Nodes: $SLURM_NNODES
echo Head node: $SLURMD_NODENAME
echo hostname $(hostname)
echo Nodelist: $SLURM_NODELIST
echo Tasks per node: $SLURM_NTASKS_PER_NODE
echo CPUs per node: $SLURM_CPUS_PER_NODE

if [[ -n $SLURM_ARRAY_JOB_ID ]]; then
    export UNIQUE_IDENTIFIER=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
else
    export UNIQUE_IDENTIFIER=$SLURM_JOBID
fi
echo UNIQUE_IDENTIFIER: $UNIQUE_IDENTIFIER

JOB_DIR=/pscratch/sd/k/kevinsg/CBC_jobs/
mkdir --parents $JOB_DIR || exit
cd $JOB_DIR || exit
echo Job directory: $PWD

PYTHON_SCRIPT=$(cat << END

import h5py
import numpy as np

# File paths for existing BES data and previously saved separatrix data
bes_file_path = '/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20240429_final.hdf5'
separatrix_file_path = '/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20240429_separatrix_data.hdf5'  
new_output_file_path = '/pscratch/sd/k/kevinsg/bes_ml_jobs/confinement_data/20241014.hdf5'
# Specify the shot to start processing from
start_shot = '149993'  # Change this to the shot number where you want to resume

# Open the existing BES data file and the separatrix data file
with h5py.File(bes_file_path, 'r') as bes_file, \
     h5py.File(separatrix_file_path, 'r') as separatrix_file, \
     h5py.File(new_output_file_path, 'w') as new_file:

    # Iterate over all shots in the BES data file
    start_processing = False  # Flag to indicate whether we've reached the start point

    for shot in bes_file.keys():
        # Skip shots until reaching the specified starting point
        if shot == start_shot:
            start_processing = True

        if not start_processing:
            continue

        try:
            # Create a corresponding group in the new file for this shot
            bes_shot_grp = bes_file[shot]
            new_shot_grp = new_file.create_group(shot)

            # Copy the attributes from the BES shot group to the new file
            for attr_name, attr_value in bes_shot_grp.attrs.items():
                new_shot_grp.attrs[attr_name] = attr_value

            # Load the separatrix data for this shot from the separatrix file
            if shot in separatrix_file:
                sep_grp = separatrix_file[shot]
                sep_times = sep_grp['times'][:]  # (N_sep_times,)
                filtered_separatrix = sep_grp['filtered_separatrix'][:]  # (N_sep_times, 6, 2)
            else:
                print(f"Separatrix data for shot {shot} not found, skipping shot.")
                continue

            # Iterate over each event within the shot (e.g., 1300, 1397, etc.)
            for event in bes_shot_grp.keys():
                try:
                    event_grp = bes_shot_grp[event]
                    new_event_grp = new_shot_grp.create_group(event)

                    # Copy the existing datasets ('signals', 'time', 'labels') to the new file
                    for dataset_name in event_grp.keys():
                        event_grp.copy(dataset_name, new_event_grp)

                    # Extract the BES time array for this event
                    if 'time' not in event_grp:
                        print(f"Time dataset for event {event} in shot {shot} not found, skipping event.")
                        continue

                    bes_times = event_grp['time'][:]  # (len_bes_times,)

                    # Initialize the new separatrix dataset with shape (len_bes_times, 6, 2)
                    separatrix_data = np.zeros((len(bes_times), 6, 2))

                    # Map the BES time points to the nearest separatrix time points
                    for i, bes_time in enumerate(bes_times):
                        # Find the index of the closest separatrix time point
                        closest_sep_idx = (np.abs(sep_times - bes_time)).argmin()
                        # Assign the corresponding filtered separatrix data to the new dataset
                        separatrix_data[i, :, :] = filtered_separatrix[closest_sep_idx]

                    # Create the 'separatrix' dataset for this event in the new file
                    new_event_grp.create_dataset('separatrix', data=separatrix_data)

                except KeyError as e:
                    print(f"KeyError for event {event} in shot {shot}: {e}. Skipping this event.")
                except Exception as e:
                    print(f"Unexpected error for event {event} in shot {shot}: {e}. Skipping this event.")

        except KeyError as e:
            print(f"KeyError for shot {shot}: {e}. Skipping this shot.")
        except Exception as e:
            print(f"Unexpected error for shot {shot}: {e}. Skipping this shot.")

END
)

echo Script:
echo "${PYTHON_SCRIPT}"


START_TIME=$(date +%s)
srun python -c "${PYTHON_SCRIPT}"
EXIT_CODE=$?
END_TIME=$(date +%s)
echo Slurm elapsed time $(( (END_TIME - START_TIME)/60 )) min $(( (END_TIME - START_TIME)%60 )) s

exit $EXIT_CODE