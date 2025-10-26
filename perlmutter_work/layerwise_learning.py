import os
import pickle
from pathlib import Path
from model_trainer.main_multitask_v3 import main

if __name__=='__main__':

    seed = int(os.getenv("RAND_SEED"))
    print(f'Linux RAND_SEED: {seed}')

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    batch_size_values = [256, 512, 1024]
    batch_size = batch_size_values[task_id % len(batch_size_values)]

    bad_elm_files = Path('/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/')
    bad_elm_indices = []
    for bad_elm_file in bad_elm_files.glob('count_of_bad_elms_multi_256*.pkl'):
        with open(bad_elm_file, 'rb') as f:
            bad_elm_data = pickle.load(f)
            bad_elm_indices.extend(bad_elm_data['bad_elms'])

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v29',
        trial_name_prefix=f'L3_',
        # data
        seed=seed,
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=240,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.4,
        fraction_validation=0.1,
        fraction_test=0.1,
        num_workers=4,
        bad_elm_indices=bad_elm_indices,
        # model
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='sum_score/train',
        fir_bp=(None, 100),
        # training
        max_epochs=251,
        lr=1e-2,
        lr_warmup_epochs=10,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=1e-5,
        batch_size=batch_size,
        use_wandb=True,
        early_stopping_patience=125,
    )
