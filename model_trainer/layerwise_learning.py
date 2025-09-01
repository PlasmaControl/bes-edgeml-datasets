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

    # mlp_choices = (
    #     [None,],
    #     [None, 16],
    #     [None, 32],
    # )
    # mlp_layers = mlp_choices[rng.choice(len(mlp_choices))]
    mlp_layers = [None, 32]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v15',
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
        ),
        mlp_tasks={
            'elm_class': mlp_layers+[1,],
            'conf_onehot': mlp_layers+[4,],
        },
        monitor_metric='elm_class/bce_loss/train',
        fir_bp=fir_choices[rng.choice(len(fir_choices))],
        # training
        max_epochs=500,
        log_freq=100,
        lr=1e-2,
        lr_warmup_epochs=20,
        lr_scheduler_patience=100,
        deepest_layer_lr_factor=1.,
        weight_decay=1e-5,
        batch_size={0:128, 20:256, 80:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
    )
