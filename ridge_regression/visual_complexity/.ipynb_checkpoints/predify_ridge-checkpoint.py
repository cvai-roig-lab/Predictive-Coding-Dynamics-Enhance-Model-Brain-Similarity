import sys
from pprint import pprint
from torchvision import transforms as T
from torchvision.models import alexnet,vgg16,resnet18
from timm.models import efficientnet_b0
import pickle
import torch
import os
import predify
import argparse

sys.path.append(os.path.abspath('..'))
from ridge_complexity import Ridge_Encoding


parser = argparse.ArgumentParser()
#parser.add_argument('--default_hyp', default=False, type=str)
parser.add_argument('--arch', default='vgg16', type=str)
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--n', default=1, type=int)

args = parser.parse_args()

import pickle
with open('../../feature_extraction/nsd_dict.pkl', 'rb') as f:
    available_paths = pickle.load(f)
print(available_paths)

DEVICE = torch.device("cuda") 

subject = args.subject

image_stimuli = available_paths[f"subj0{subject}_images"]
brain_data = available_paths[f"subj0{subject}_rois"]


fbm_dict = {"zero": 0.0, "one": 0.1}
fts_name = args.arch

for key, val in fbm_dict.items():
    
    print(key)
    
    save_path_predify = f"Tutorial_subj{subject}_increase_fbm/{fts_name}_{key}/ts" + str(args.n) + "/"
    
    #loop_dict = {save_path_predify + '/arr/': save_path_predify.split('/')[-1]}
    loop_dict = {save_path_predify + '/arr/': save_path_predify.split('/')[-3].split('_')[0]}


    for feat_path, model_name in loop_dict.items():
        print(f"Ridge Encoding for {model_name} ...")

        # Call the linear encoding function
        results_df_clip_high,results_df_clip_low = Ridge_Encoding(
        feat_path=feat_path,
        roi_path=brain_data,
        model_name=model_name,
        subject = subject,
        trn_tst_split=0.8,
        n_folds=3,
        n_components=100,
        batch_size=100,
        random_state=42,
        return_correlations=True,
        save_path=f"Tutorial_LE_Results_Ridge_opt/subj{subject}/{fts_name}_{key}" + f"/ts" + str(args.n) + '/'    #n=1: time steps 10
    )
