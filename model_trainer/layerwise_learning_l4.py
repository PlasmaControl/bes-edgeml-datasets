from model_trainer.main_multitask import main

if __name__=='__main__':
    main(
        # scenario
        signal_window_size=256,
        experiment_name='experiment_256_v6',
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
        ),
        mlp_tasks = {
            'elm_class': [None, 32, 1],
        },
        no_bias=False,
        monitor_metric='elm_class/bce_loss/train',
        batch_norm=True,
        fir_bp=(None, 150),
        # training
        max_epochs=500,
        log_freq=100,
        lr=1e-3,
        lr_warmup_epochs=10,
        lr_scheduler_patience=80,
        deepest_layer_lr_factor=0.1,
        batch_size={0:64, 10:128, 20:256},
        num_workers=8,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        use_wandb=True,
        early_stopping_patience=200,
        transfer_model='/global/homes/d/drsmith/scratch-ml/experiment_256_v5/r41396289_52_2025_08_05_11_52_47/checkpoints/last.ckpt',
        transfer_max_layer=7,
        transfer_layer_lr_factor=1e-3,
    )
