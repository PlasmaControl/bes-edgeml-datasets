import os
from model_trainer.main_multitask_v3 import main

if __name__=='__main__':

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    fir_choices = (
        (8, None),
        (12, None),
        (None, 100),
        (None, 200),
    )
    fir_bp = fir_choices[task_id%len(fir_choices)]
    weight_decay_choices = (1e-4, 1e-5)
    weight_decay = weight_decay_choices[(task_id//len(fir_choices))%len(weight_decay_choices)]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v26',
        trial_name_prefix=f'L3_BS512_T{task_id:02d}',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=240,
        max_confinement_event_length=int(40e3),
        confinement_dataset_factor=0.5,
        fraction_validation=0.125,
        fraction_test=0.125,
        num_workers=4,
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
        fir_bp=fir_bp,
        # training
        max_epochs=200,
        lr=1e-2,
        lr_warmup_epochs=10,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=512,
        use_wandb=True,
        early_stopping_patience=125,
    )
