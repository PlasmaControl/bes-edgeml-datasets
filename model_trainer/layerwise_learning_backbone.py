import os
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    i_array = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    jobs = (
        ('r42242222_3_2025_08_31_21_01_38', (8, None)),
        ('r42242222_7_2025_08_31_21_27_39', (8, None)),
        ('r42242222_67_2025_09_01_05_56_11', (8, None)),
        ('r42242222_19_2025_08_31_23_02_20', (8, None)),
        ('r42242222_63_2025_09_01_05_24_00', (8, None)),
        ('r42242222_57_2025_09_01_04_58_50', (8, None)),
        ('r42242222_78_2025_09_01_06_53_26', (8, None)),
        ('r42242222_8_2025_08_31_21_27_42', (8, None)),
        ('r42242222_28_2025_08_31_23_56_50', (8, None)),
        ('r42242222_16_2025_08_31_22_31_11', (8, None)),
        ('r42242222_38_2025_09_01_01_21_08', (8, None)),
        ('r42242222_75_2025_09_01_06_43_31', (8, None)),
        ('r42242222_20_2025_08_31_23_02_31', (8, None)),
        ('r42242222_80_2025_09_01_07_02_10', (8, None)),
        ('r42242222_46_2025_09_01_02_07_03', (8, None)),
        ('r42242222_44_2025_09_01_02_04_24', (8, None)),
        ('r42242222_62_2025_09_01_05_24_04', (8, None)),
        ('r42242222_34_2025_09_01_01_08_12', (8, None)),
        ('r42242222_21_2025_08_31_23_13_02', (None, 200)),
        ('r42242222_79_2025_09_01_06_56_24', (None, 200)),
        ('r42242222_6_2025_08_31_21_09_45', (None, 200)),
    )
    job_id = jobs[i_array][0]
    fir_bp = jobs[i_array][1]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v16',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        # max_elms=rng.choice([40, 60]),
        max_elms=80,
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
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='elm_class/bce_loss/train',
        fir_bp=fir_bp,
        # training
        max_epochs=500,
        log_freq=100,
        lr=1e-2,
        lr_warmup_epochs=20,
        lr_scheduler_patience=60,
        deepest_layer_lr_factor=0.1,
        weight_decay=1e-5,
        batch_size={0:128, 20:256, 80:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        backbone_model_path=f'multi_256_v15/{job_id}',
        backbone_first_n_layers=3,
        backbone_initial_ratio_lr=1e-2,
        backbone_unfreeze_at_epoch=40,
    )
