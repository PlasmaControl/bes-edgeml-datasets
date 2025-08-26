import os
from numpy import random
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    jobs = (
        ('r42002857_14_2025_08_25_11_30_57', (None, 200)),
        ('r42002857_45_2025_08_25_16_44_36', (None, 200)),
        ('r42010042_43_2025_08_25_23_36_33', (None, 200)),
        ('r42010042_13_2025_08_25_19_52_47', (8, None)),
        ('r42002857_11_2025_08_25_10_52_35', (8, None)),
        ('r42002857_28_2025_08_25_13_44_37', (8, None)),
        ('r42010042_7_2025_08_25_19_09_05', (8, None)),
        ('r42010042_22_2025_08_25_20_56_00', (8, 200)),
        ('r42010042_11_2025_08_25_19_23_42', (8, 200)),
        ('r42010042_41_2025_08_25_23_22_31', (8, 200)),
    )
    job_id = jobs[i_array][0]
    fir_bp = jobs[i_array][1]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v11',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=40,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.3,
        fraction_validation=0.15,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None,16,1],
            'conf_onehot': [None,16,4],
        },
        monitor_metric='sum_loss/train',
        fir_bp=fir_bp,
        # training
        max_epochs=500,
        log_freq=100,
        lr=1e-2,
        lr_warmup_epochs=20,
        deepest_layer_lr_factor=0.1,
        lr_scheduler_patience=90,
        weight_decay=1e-4,
        batch_size={0:128, 20:256, 70:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        backbone_model_path=f'multi_256_v10/{job_id}',
        backbone_first_n_layers = 3,
        backbone_initial_ratio_lr = 0.01,
        backbone_unfreeze_at_epoch = 50,
        backbone_warmup_rate = 2,
    )
