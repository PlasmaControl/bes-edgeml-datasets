import os
import pickle
from numpy import random
from model_trainer.main_multitask_v3 import main

if __name__=='__main__':

    seed = int(os.getenv("RAND_SEED"))
    print(f'Linux RAND_SEED: {seed}')
    rng = random.default_rng(seed)

    task_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    fir_choices = (
        (8, None),
        (12, None),
        (None, 100),
        (None, 200),
    )
    fir_bp = fir_choices[task_id%len(fir_choices)]
    weight_decay_choices = (1e-4, 1e-5)
    weight_decay = rng.choice(weight_decay_choices)
    batch_size = int(rng.choice([256,512,1024]))

    bad_elm_files = [
        '/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/count_of_bad_elms_multi_256_v27.pkl',
        '/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/count_of_bad_elms_multi_256_v28.pkl',
    ]
    bad_elm_indices = []
    for bad_elm_file in bad_elm_files:
        with open(bad_elm_file, 'rb') as f:
            bad_elm_data = pickle.load(f)
            bad_elm_indices.extend(bad_elm_data['bad_elms'])

    main(
        # scenario
        signal_window_size=256,
        experiment_name='multi_256_v28',
        trial_name_prefix=f'L3_T{task_id:02d}',
        # data
        seed=seed,
        elm_data_file='/global/homes/d/drsmith/scratch-ml/data/elm_data.20240502.hdf5',
        confinement_data_file='/global/homes/d/drsmith/scratch-ml/data/confinement_data.20240112.hdf5',
        max_elms=240,
        max_confinement_event_length=int(30e3),
        confinement_dataset_factor=0.4,
        fraction_validation=0.1,
        fraction_test=0,
        num_workers=4,
        bad_elm_indices=bad_elm_indices,
        # model
        feature_model_layers = (
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
            {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1,         'bias': True},
            {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias': True},
        ),
        mlp_tasks={
            'elm_class': [None, 32, 1],
            'conf_onehot': [None, 32, 4],
        },
        monitor_metric='sum_score/train',
        fir_bp=fir_bp,
        # training
        max_epochs=201,
        lr=1e-2,
        lr_warmup_epochs=10,
        lr_scheduler_patience=50,
        lr_scheduler_threshold=1e-2,
        weight_decay=weight_decay,
        batch_size=batch_size,
        use_wandb=True,
        early_stopping_patience=125,
    )
