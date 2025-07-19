from model_trainer.scenarios import scenario_128

if __name__ == '__main__':
    scenario_128(
        batch_size=256,
        max_epochs=80,
        use_wandb=True,
        experiment_name='experiment_128_v1',
        restart_trial_name='r40923398_2025_07_19_10_00_18',
        wandb_id='lnahvlw1',
    )
