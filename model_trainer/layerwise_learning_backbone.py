import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    jobs = (
        ('r43166502_0_2025_09_26_20_00_19', (None, 100), 300, 1e-4),
        ('r43166502_1_2025_09_26_20_00_50', (None, 100), 300, 1e-4),
        ('r43166502_2_2025_09_26_20_12_44', (12, None), 300, 1e-5),
        ('r43166502_3_2025_09_26_20_12_42', (None, 100), 300, 1e-4),
        ('r43166502_4_2025_09_26_20_33_30', (12, None), 300, 1e-5),
        ('r43166502_5_2025_09_26_20_33_35', (None, 100), 300, 1e-4),
        ('r43166502_6_2025_09_26_20_47_37', (None, 100), 300, 1e-5),
        ('r43166502_7_2025_09_26_20_47_33', (None, 200), 400, 1e-5),
        ('r43166502_8_2025_09_26_21_30_57`', (None, 200), 400, 1e-5),
        ('r43166502_9_2025_09_26_21_32_15', (None, 200), 400, 1e-5),
        ('r43166502_10_2025_09_26_22_50_13', (None, 200), 400, 1e-5),
        ('r43166502_11_2025_09_26_22_54_55', (None, 200), 400, 1e-5),
        ('r43166502_12_2025_09_26_23_09_22', (None, 200), 400, 1e-5),
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
        experiment_name='multi_256_v24',
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
        max_epochs=800,
        lr=3e-3,
        lr_warmup_epochs=20,
        lr_scheduler_patience=100,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=512,
        use_wandb=True,
        early_stopping_patience=250,
        backbone_model_path=f'multi_256_v23/{job_id}',
        backbone_first_n_layers=100,
        backbone_unfreeze_at_epoch=0,
        backbone_initial_lr=3e-5,
        backbone_warmup_rate=2,
    )
