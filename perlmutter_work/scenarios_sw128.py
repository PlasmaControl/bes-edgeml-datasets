from model_trainer.main_multitask import main
from ml_data import small_data_200

if __name__=='__main__':
    feature_model_layers = (
        {'out_channels': 2, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': False},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
        {'out_channels': 2, 'kernel': (4, 1, 1), 'stride': (4, 1, 1), 'bias': False},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
        {'out_channels': 2, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
    )
    main(
        signal_window_size=128,
        restart_trial_name='',
        wandb_id='',
        experiment_name='experiment_128_v1',
        max_epochs=200,
        lr=1e-3,
        deepest_layer_lr_factor=0.1,
        lr_warmup_epochs=15,
        use_wandb=True,
        no_bias=True,
        elm_data_file=small_data_200,
        batch_size={0:64, 15:128, 30:256},
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        num_workers=4,
        feature_model_layers=feature_model_layers,
        mlp_task_models=None,
        fir_bp_low=5,
        fir_bp_high=250,
    )
