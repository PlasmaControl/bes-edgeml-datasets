from model_trainer.main_multitask import main
from ml_data import small_data_200


def scenario_128(
        batch_size=256,
        lr=1e-3,
        max_epochs=5,
        max_elms=None,
        use_wandb=False,
        experiment_name='experiment_default',
        data_file=small_data_200,
        **kwargs,
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
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        num_workers=4,
        use_wandb=use_wandb,
        experiment_name=experiment_name,
        feature_model_layers=feature_model_layers,
        mlp_task_models=mlp_task_models,
        batch_size=batch_size,
        lr=lr,
        elm_data_file=data_file,
        max_elms=max_elms,
        max_epochs=max_epochs,
        **kwargs,
    )

if __name__=='__main__':
    scenario_128(max_elms=100, max_epochs=2)
