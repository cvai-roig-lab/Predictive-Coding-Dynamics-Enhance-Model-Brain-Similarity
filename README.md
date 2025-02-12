<h1> 
<div class="row">
  <div class="column"> Predictive Coding-Dynamics Enhance Model-Brain Similarity </div>
</div>
</h1>

Code for the paper ["Predictive-Coding-Dynamics-Enhance-Model-Brain-Similarity"](https://arxiv.pdf) presented at ESANN 2025.

## Overview:


## Contents:
* `data`: we have provided sample data we used in the paper for demonstration purposes.
  - `example_eeg`: contains eeg from one subject including eyes opened and closed states.
    - `example_subj_EC_raw.fif.gz`: sample eeg recording for eyes closed resting state.
    - `example_subj_EO_raw.fif.gz`: sample eeg recording for eyes opened resting state.
    - `example_subj_EC_preproc.pickle`: preprocessed eeg data from *example_subj_EC_raw.fif.gz*.
    - `example_subj_EO_preproc.pickle`: preprocessed eeg data from *example_subj_EO_raw.fif.gz*.
  - `example_training_set`: contains the `12-all` training set we used in the paper.
* `models`: contains the corresponding code for each of the 10 models explored in the original paper.
  - `training.ipynb`: contains the code for training and tuning the models.
  - `shap_values.ipynb`: contains the code for computing the shap values.
  - `shap_values.pickle`: is a dict object containing the shap values and fold indexes they were computed on for the specific model.
* `example_eeg_preprocessing.ipynb`: a step by step notebook showcasing how to to run our data prerpocessing pipeline. 
* `example_eeg_feature_extraction.ipynb`: a step by step notebook showcasing how to run our feature extraction pipeline. 
* `example_plot_interpolation_map.ipynb`: a step by step notebook showcasing how to create the regional interpolation plots.
* `example_aggregated_shap_value.ipynb`: a step by step notebook showcasing how to aggregate shap values and create rank orders of the feature groups based on the shap value.
* `example_shap_agreement_metric.ipynb`: a step by step notebook showcasing how to compute the ShapAgreement metric between models.

## Cite
```
@INPROCEEDINGS{10385662,
  author={}
}
```
