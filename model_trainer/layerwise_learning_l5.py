from model_trainer.main_multitask import main

if __name__=='__main__':
    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v7',
        # data
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/small_data_100.hdf5',
        max_elms=40,
        # model
        use_optimizer='adam',
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
        ),
        mlp_tasks = {
            'elm_class': [None, 32, 1],
        },
        no_bias=False,
        monitor_metric='elm_class/bce_loss/train',
        batch_norm=True,
        # training
        max_epochs=400,
        lr=1e-4,
        lr_warmup_epochs=15,
        lr_scheduler_patience=100,
        deepest_layer_lr_factor=0.2,
        batch_size={0:128, 20:256, 150:512},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=250,
        transfer_model='experiment_256_v6/r41506284_1_2025_08_08_14_45_37/checkpoints/last.ckpt',
        fir_bp=(None,250),
        transfer_max_layer=10,
        transfer_layer_lr_factor=1e-3,
    )
