from model_trainer.scenarios import scenario_128

data_file = '/global/homes/d/drsmith/scratch-ml/data/small_data_200.hdf5'

if __name__ == '__main__':
    scenario_128(
        batch_size=128,
        max_epochs=3,
        max_elms=20,
        # use_wandb=True,
        experiment_name='experiment_128_v1',
        data_file=data_file,
    )
