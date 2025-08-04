from numpy import random
from model_trainer.main_multitask import main
import ml_data

if __name__=='__main__':
    # print(f'Linux rand: { 0 }')
    # rng = random.default_rng(seed=0)

    kwargs = {}
    kwargs['feature_model_layers'] = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
    )

    # regularization_choice = rng.choice(3)
    # if regularization_choice == 0:
    kwargs['batch_norm'] = True
    # elif regularization_choice == 1:
    #     kwargs['dropout'] = rng.choice([0.04, 0.12])

    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v4',
        # data
        # elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        elm_data_file=ml_data.small_data_100,
        # max_elms=rng.choice([20, 40]),
        max_elms=40,
        # model,
        use_optimizer='adam',
        mlp_tasks = {
            # 'elm_class': [None, rng.choice([16, 32]), 1],
            'elm_class': [None, 16, 1],
        },
        # no_bias=rng.choice([True, False]),
        no_bias=True,
        # fir_bp_low=rng.choice([0, 10]),
        # fir_bp_high=rng.choice([0, 75, 250]),
        # monitor_metric='elm_class/bce_loss/train',
        # training
        max_epochs=10,
        log_freq=100,
        # lr=rng.choice([1e-2, 3e-2]),
        lr=1e-3,
        lr_warmup_epochs=10,
        lr_scheduler_patience=80,
        deepest_layer_lr_factor=1.,
        # deepest_layer_lr_factor=rng.choice([1.0, 0.2]),
        # batch_size={0:64, 10:128, 20:256},
        batch_size=128,
        num_workers=2,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        # use_wandb=True,
        early_stopping_patience=150,
        # kwargs
        **kwargs,
    )
