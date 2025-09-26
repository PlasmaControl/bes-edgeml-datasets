import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    jobs = (
        #
        ('r43041913_0_2025_09_25_10_03_58', (None, 100), 90, 1e-4),
        ('r43041898_1_2025_09_24_00_47_31', (None, 100), 90, 1e-4),
        ('r43041901_2_2025_09_24_15_35_08', (12, None), 90, 1e-5),
        # ('r42970915_1_2025_09_21_19_50_54', (None, 100), 60, 1e-5),
        # ('r42970920_6_2025_09_21_22_46_24', (None, 200), 60, 1e-4),
        # ('r42970967_1_2025_09_22_04_52_37', (None, 100), 60, 1e-5),
        ('r43041912_6_2025_09_25_05_32_06', (None, 100), 90, 1e-4),
        ('r43041913_7_2025_09_25_11_10_26', (12, None), 90, 1e-5),
        ('r43041913_8_2025_09_25_12_26_10', (None, 100), 90, 1e-4),
        ('r43041912_9_2025_09_25_06_30_23', (None, 100), 90, 1e-5),
        #
        ('r43041912_10_2025_09_25_07_08_03', (None, 200), 120, 1e-5),
        # ('r42970967_8_2025_09_22_05_19_30', (None, 200), 80, 1e-5),
        ('r43041912_12_2025_09_25_07_41_36', (None, 200), 120, 1e-5),
        ('r43041898_13_2025_09_24_12_48_10', (None, 200), 120, 1e-5),
        # ('r42970939_15_2025_09_22_04_09_45', (12, None), 80, 1e-5),
        ('r43041898_15_2025_09_24_13_11_37', (None, 200), 120, 1e-5),
        ('r43041912_16_2025_09_25_08_28_04', (None, 200), 120, 1e-5),
        # ('r42970934_11_2025_09_22_02_17_51', (None, 100), 80, 1e-3),
        # ('r42970915_10_2025_09_21_21_05_15', (12, None), 80, 1e-4),
        ('r43041912_19_2025_09_25_08_54_03', (None, 200), 120, 1e-5),
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
        experiment_name='multi_256_v21',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=max_elms,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.4,
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
        lr=1e-3,
        lr_warmup_epochs=20,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=512,
        use_wandb=True,
        early_stopping_patience=150,
        backbone_model_path=f'multi_256_v20/{job_id}',
        backbone_first_n_layers=100,
        backbone_unfreeze_at_epoch=0,
        backbone_initial_lr=3e-5,
        backbone_warmup_rate=2,
    )
