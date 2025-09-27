import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    jobs = (
        ('r43153008_0_2025_09_26_13_03_13', (None, 100), 200, 1e-4),
        ('r43153008_1_2025_09_26_13_03_11', (None, 100), 200, 1e-4),
        ('r43153008_2_2025_09_26_13_12_43', (12, None), 200, 1e-5),
        ('r43153008_3_2025_09_26_13_12_45', (None, 100), 200, 1e-4),
        ('r43153008_4_2025_09_26_13_46_20', (12, None), 200, 1e-5),
        ('r43153008_5_2025_09_26_13_48_20', (None, 100), 200, 1e-4),
        ('r43153008_6_2025_09_26_13_25_15', (None, 100), 200, 1e-5),
        ('r43153008_7_2025_09_26_13_25_20', (None, 200), 270, 1e-5),
        ('r43153008_8_2025_09_26_13_25_11', (None, 200), 270, 1e-5),
        ('r43153008_9_2025_09_26_13_25_21', (None, 200), 270, 1e-5),
        ('r43153008_10_2025_09_26_13_55_26', (None, 200), 270, 1e-5),
        ('r43153008_11_2025_09_26_13_55_34', (None, 200), 270, 1e-5),
        ('r43153008_12_2025_09_26_14_18_07', (None, 200), 270, 1e-5),
    )
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    i_job = i_array % len(jobs)
    job_id = jobs[i_job][0]
    fir_bp = jobs[i_job][1]
    max_elms = jobs[i_job][2]
    weight_decay = jobs[i_job][3]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v23',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=max_elms,
        max_confinement_event_length=int(50e3),
        confinement_dataset_factor=0.6,
        fraction_validation=0.15,
        num_workers=4,
        # model
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='sum_score/train',
        fir_bp=fir_bp,
        # training
        max_epochs=600,
        lr=1e-3,
        lr_warmup_epochs=20,
        lr_scheduler_patience=75,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=512,
        use_wandb=True,
        early_stopping_patience=200,
        backbone_model_path=f'multi_256_v22/{job_id}',
        backbone_first_n_layers=100,
        backbone_unfreeze_at_epoch=0,
        backbone_initial_lr=3e-5,
        backbone_warmup_rate=2,
    )
