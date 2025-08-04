import os
from numpy import random
from model_trainer.main_multitask import main

if __name__=='__main__':
    seed = int(os.environ.get('rand', 0))
    print(f'RNG seed: {seed}')
    rng = random.default_rng(seed=seed)

    kwargs = {}
    kwargs['feature_model_layers'] = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
    )

    fir_choices = (
        (8, None),
        (8, 150),
        (None, 150),
    )
    fir_choice = rng.choice(len(fir_choices))
    kwargs['fir_bp'] = fir_choices[fir_choice]

    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v5',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        max_elms=rng.choice([40, 60, 80]),
        # model
        use_optimizer='adam',
        mlp_tasks = {
            'elm_class': [None, 32, 1],
        },
        no_bias=rng.choice([True, False]),
        monitor_metric='elm_class/bce_loss/train',
        batch_norm=True,
        # training
        max_epochs=500,
        log_freq=100,
        lr=rng.choice([1e-2, 3e-2]),
        lr_warmup_epochs=15,
        lr_scheduler_patience=80,
        deepest_layer_lr_factor=1.,
        batch_size={0:64, 15:128, 30:256},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        # kwargs
        **kwargs,
    )
