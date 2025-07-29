from numpy import random
from model_trainer.main_multitask import main


if __name__=='__main__':
    rng = random.default_rng()
    kwargs = {}
    kwargs['feature_model_layers'] = (
        {'out_channels': 2, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 2, 'kernel': (4, 1, 1), 'stride': (4, 1, 1), 'bias': True},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
    )
    if rng.choice(2) == 0:
        kwargs['batch_norm'] = True
    else:
        kwargs['dropout'] = rng.choice([0.04, 0.12])
    main(
        # scenario
        signal_window_size=128,
        experiment_name='experiment_128_v1',
        # restart
        restart_trial_name='',
        wandb_id='',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
        # model
        mlp_task_models=None,
        no_bias=rng.choice([True, False]),
        # fir_bp_low=5,
        fir_bp_low=rng.choice([0, 10]),
        # fir_bp_high=250,
        fir_bp_high=rng.choice([0, 75, 250]),
        # training
        max_epochs=400,
        # lr=2e-3,
        lr=rng.choice([3e-4, 1e-3, 3e-3]),
        # lr_warmup_epochs=10,
        lr_warmup_epochs=rng.choice([3, 10]),
        lr_scheduler_patience=80,
        # deepest_layer_lr_factor=0.1,
        deepest_layer_lr_factor=rng.choice([1.0, 0.1]),
        batch_size={0:64, 6:128, 20:256},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        # kwargs
        **kwargs,
    )
