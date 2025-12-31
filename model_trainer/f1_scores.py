from pathlib import Path
from model_trainer.main_multitask_v3 import Model


if __name__=='__main__':
    files = [
        # Path('experiment_default/r44057567/scores/elm_wise_f1_scores_ep0000.pkl'),
        # Path('experiment_default/r44057567/scores/elm_wise_f1_scores_test.pkl'),
        # Path('/global/homes/d/drsmith/scratch-ml/multi_256_v26/L3_BS512_T11_r43927012_11/scores/elm_wise_f1_scores_ep0175.pkl'),
        Path('/global/homes/d/drsmith/scratch-ml/multi_256_v29/L3__r44436312_45/scores/elm_wise_f1_scores_ep0250.pkl'),
    ]
    for file in files:
        Model.read_elm_scores(file)
