import os
import pickle
from pathlib import Path
from model_trainer.main_multitask_v3 import main

if __name__=='__main__':

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    backbone_model_path = f'multi_256_v30/L3_r46911715_{task_id}'

    bad_elm_files = Path('/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/')
    bad_elm_indices = []
    for bad_elm_file in bad_elm_files.glob('count_of_bad_elms_multi_256*.pkl'):
        with open(bad_elm_file, 'rb') as f:
            bad_elm_data = pickle.load(f)
            bad_elm_indices.extend(bad_elm_data['bad_elms'])

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v30',
        trial_name_prefix='L4',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=480,
        max_confinement_event_length=int(40e3),
        confinement_dataset_factor=0.5,
        fraction_validation=0.1,
        fraction_test=0.1,
        num_workers=4,
        bad_elm_indices=bad_elm_indices,
        # model
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            # {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='sum_score/train',
        fir_bp=(None, 100),
        # training
        max_epochs=300,
        lr=1e-2,
        lr_warmup_epochs=10,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=1e-5,
        batch_size=256,
        use_wandb=True,
        early_stopping_patience=125,
        backbone_model_path=backbone_model_path,
        backbone_unfreeze_at_epoch=10,
        backbone_first_n_layers=3,
        backbone_initial_lr=1e-3,
        backbone_warmup_rate=2,
    )
