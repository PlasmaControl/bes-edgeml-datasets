from model_trainer.scenarios import scenario_128

if __name__ == '__main__':
    scenario_128(
        batch_size=256,
        max_epochs=6,
        # experiment_name='experiment_128_v1',
        # use_wandb=True,
    )
