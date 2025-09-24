import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    jobs = (
        ('r42970939_3_2025_09_22_03_26_40', (None, 100), 60, 1e-4),
        ('r42970967_3_2025_09_22_04_52_26', (None, 100), 60, 1e-4),
        ('r42970939_0_2025_09_22_03_26_48', (12, None), 60, 1e-5),
        ('r42970915_1_2025_09_21_19_50_54', (None, 100), 60, 1e-5),
        ('r42970920_6_2025_09_21_22_46_24', (None, 200), 60, 1e-4),
        ('r42970967_1_2025_09_22_04_52_37', (None, 100), 60, 1e-5),
        ('r42970920_3_2025_09_21_22_33_44', (None, 100), 60, 1e-4),
        ('r42970920_0_2025_09_21_22_19_18', (12, None), 60, 1e-5),
        ('r42970915_3_2025_09_21_20_12_40', (None, 100), 60, 1e-4),
        ('r42970920_1_2025_09_21_22_19_20', (None, 100), 60, 1e-5),
        ('r42970915_12_2025_09_21_21_18_03', (None, 200), 80, 1e-5),
        ('r42970967_8_2025_09_22_05_19_30', (None, 200), 80, 1e-5),
        ('r42970934_12_2025_09_22_02_21_46', (None, 200), 80, 1e-5),
        ('r42970939_12_2025_09_22_04_03_19', (None, 200), 80, 1e-5),
        ('r42970939_15_2025_09_22_04_09_45', (12, None), 80, 1e-5),
        ('r42970967_12_2025_09_22_05_31_32', (None, 200), 80, 1e-5),
        ('r42970920_12_2025_09_21_23_10_35', (None, 200), 80, 1e-5),
        ('r42970934_11_2025_09_22_02_17_51', (None, 100), 80, 1e-3),
        ('r42970915_10_2025_09_21_21_05_15', (12, None), 80, 1e-4),
        ('r42970928_12_2025_09_22_00_52_59', (None, 200), 80, 1e-5),
    )
    job_id = jobs[i_array][0]
    fir_bp = jobs[i_array][1]
    max_elms = jobs[i_array][2]
    weight_decay = jobs[i_array][3]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v20',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=max_elms,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.3,
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
        max_epochs=500,
        lr=3e-3,
        lr_warmup_epochs=10,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=256,
        use_wandb=True,
        early_stopping_patience=150,
        backbone_model_path=f'multi_256_v19/{job_id}',
        backbone_first_n_layers=4,
        backbone_unfreeze_at_epoch=20,
        backbone_initial_lr=1e-4,
        backbone_warmup_rate=2,
    )
