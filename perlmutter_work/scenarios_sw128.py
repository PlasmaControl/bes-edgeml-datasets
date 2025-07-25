from model_trainer.main_multitask import main

feature_model_layers = (
    {'out_channels': 2, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': False},
    {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
    {'out_channels': 2, 'kernel': (4, 1, 1), 'stride': (4, 1, 1), 'bias': False},
    {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
    {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
)
main(
    # scenario
    signal_window_size=128,
    experiment_name='experiment_128_v1',
    # restart
    restart_trial_name='',
    wandb_id='',
    # data
    elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_500.hdf5',
    # max_elms=500,
    # model
    feature_model_layers=feature_model_layers,
    mlp_task_models=None,
    no_bias=True,
    fir_bp_low=5,
    fir_bp_high=250,
    dropout=0.04,
    batch_norm=True,
    # training
    max_epochs=400,
    lr=2e-3,
    lr_warmup_epochs=10,
    lr_scheduler_patience=80,
    deepest_layer_lr_factor=0.1,
    early_stopping_patience=250,
    batch_size={0:64, 10:128, 30:256},
    num_workers=8,
    gradient_clip_val=1,
    gradient_clip_algorithm='value',
    use_wandb=True,
)
