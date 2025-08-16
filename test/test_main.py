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

def test_batch_size_stepping():
    main(
        elm_data_file=ml_data.small_data_100,
        max_elms=20,
        max_epochs=4,
        batch_size={0:128, 2:256},
        fir_bp=[10,250],  # also testing FIR
    )

def test_restart_with_more_data(base_training_scenario):
    first_run_outputs = base_training_scenario
    print("\n**** RESTARTING WITH MORE DATA ****\n")
    second_run_outputs = main(
        elm_data_file=ml_data.small_data_100,
        max_elms=40,
        max_epochs=4,
        restart_trial_name=first_run_outputs['trial_name'],
    )
    assert second_run_outputs['trial_name'] == first_run_outputs['trial_name']

def test_transfer_learning_with_backbone(base_training_scenario):
    outputs = base_training_scenario
    feature_model_layers = (
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1, 'bias':True},
        {'out_channels': 4, 'kernel': (8, 1, 1), 'stride': (8, 1, 1), 'bias':True},
        {'out_channels': 4, 'kernel': (1, 3, 3), 'stride': 1, 'bias':True},  # new layer
    )
    print("\n**** TRANSFER LEARNING WITH BACKBONE MODEL ****\n")
    new_outputs = main(
        elm_data_file=ml_data.small_data_100,
        max_elms=20,
        feature_model_layers=feature_model_layers,
        backbone_model_path=outputs['trial_path'],
        backbone_first_n_layers=3,
        backbone_initial_ratio_lr=1e-3,
    )
