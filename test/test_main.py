import pytest

from model_trainer.main_multitask_v2 import main
import ml_data

@pytest.fixture
def base_training_scenario():
    outputs = main(
        elm_data_file=ml_data.small_data_100,
        max_elms=20,
    )
    return outputs

def test_main(base_training_scenario):
    assert base_training_scenario

def test_restart_with_more_data(base_training_scenario):
    existing_trial_name, _ = base_training_scenario
    print("**** RESTARTING WITH MORE DATA ****")
    new_trial_name, _ = main(
        elm_data_file=ml_data.small_data_100,
        max_elms=40,
        max_epochs=5,
        restart_trial_name=existing_trial_name,
    )
    assert new_trial_name == existing_trial_name

def test_transfer_learning_with_backbone():
    pass

def test_batch_size_stepping():
    main(
        elm_data_file=ml_data.small_data_100,
        max_elms=20,
        max_epochs=5,
        batch_size={0:128, 3:256},
        fir_bp=[10,250],  # also testing FIR
    )
