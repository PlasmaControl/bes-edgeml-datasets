from main import main

data_file='/Users/drsmith/Documents/repos/bes-ml-data/model_trainer/small_data_100.hdf5'
# data_file='/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/small_data_200.hdf5'

num_workers = 4
max_elms = None
max_epochs = 80


def scenario_128(
        batch_size=128,
        lr=1e-3,
):
    feature_model_layers = (
        {'out_channels': 2, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': False},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
        {'out_channels': 2, 'kernel': (4, 1, 1), 'stride': (4, 1, 1), 'bias': False},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
    )
    mlp_task_models=None
    main(
        signal_window_size=128,
        experiment_name='experiment_128_v1',
        no_bias=True,
        batch_norm=False,
        use_wandb=True,
        # skip_data=True,
        # skip_train=True,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        feature_model_layers=feature_model_layers,
        mlp_task_models=mlp_task_models,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        data_file=data_file,
        max_elms=max_elms,
        max_epochs=max_epochs,
    )

if __name__=='__main__':
    # scenario_128()
    for _ in range(3):
        scenario_128(batch_size=64)
        scenario_128(batch_size=128)
        scenario_128(batch_size=256)
