from model_trainer.main_multitask import main
import ml_data

if __name__=='__main__':

    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': False},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': False},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': False},
    )

    mlp_tasks = {'elm_class': [None, 32, 1]}

    main(
        # scenario
        signal_window_size=256,
        # restart
        restart_trial_name='',
        wandb_id='',
        # data
        elm_data_file=ml_data.small_data_100,
        max_elms=30,
        # model
        feature_model_layers=feature_model_layers,
        mlp_tasks=mlp_tasks,
        # training
        max_epochs=2,
        lr=1e-3,
        deepest_layer_lr_factor=0.1,
        batch_size=256,
        num_workers=2,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        # skip_data=True,
        # skip_train=True,
    )