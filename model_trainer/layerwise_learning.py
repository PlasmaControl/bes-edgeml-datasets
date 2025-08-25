import os
from numpy import random
from model_trainer.main_multitask_v2 import main

if __name__=='__main__':
    seed = os.environ.get('rand', None)
    if seed is not None: seed = int(seed)
    print(f'RNG seed: {seed}')
    rng = random.default_rng(seed=seed)

    fir_choices = (
        (8, None),
        (None, 200),
        (8, 200),
    )

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v10',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=rng.choice([40, 80]),
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.3,
        fraction_validation=0.15,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None,16,1],
            'conf_onehot': [None,16,4],
        },
        monitor_metric='sum_loss/train',
        fir_bp=fir_choices[rng.choice(len(fir_choices))],
        # training
        max_epochs=500,
        log_freq=100,
        lr=rng.choice([1e-2, 3e-2]),
        lr_warmup_epochs=15,
        deepest_layer_lr_factor=1.,
        weight_decay=rng.choice([1e-2,1e-3,1e-4]),
        batch_size={0:64, 10:128, 30:256, 90:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
    )
