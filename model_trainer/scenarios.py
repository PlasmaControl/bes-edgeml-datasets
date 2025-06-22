from main import main

data_file='/Users/drsmith/Documents/repos/bes-ml-data/model_trainer/small_data_100.hdf5'
# data_file='/global/homes/d/drsmith/ml/bes-edgeml-datasets/model_trainer/small_data_200.hdf5'
num_workers = 1
max_elms = 20
max_epochs = 10


def scenario_128(
        batch_size=128,
        lr=1e-3,
):
    main(
        signal_window_size=128,
        gradient_clip_val=1,
        gradient_clip_algorithm='value',
        no_bias=True,
        batch_norm=False,
        batch_size=batch_size,
        lr=lr,
        num_workers=num_workers,
        data_file=data_file,
        max_elms=max_elms,
        max_epochs=max_epochs,
    )

if __name__=='__main__':
    scenario_128()
