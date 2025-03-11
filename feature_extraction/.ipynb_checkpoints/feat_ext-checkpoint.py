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
import tqdm

import utils.extraction as ext

sys.path.append(os.path.abspath('..'))
from net2brain.feature_extraction import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--n', default=2, type=int)

args = parser.parse_args()

import pickle
with open('nsd_dict.pkl', 'rb') as f:
    available_paths = pickle.load(f)
print(available_paths)

DEVICE = torch.device("cuda") 

subject = args.subject

image_stimuli = available_paths[f"subj0{subject}_images"]
brain_data = available_paths[f"subj0{subject}_rois"]
text_stimuli = available_paths[f"subj0{subject}"] + "/training_split/training_text/"

#fbm_dict = {"zero": 0.0, "one": 0.1, "two": 0.2, "three": 0.3, "four": 0.4, "five": 0.5}
fbm_dict = {"three": 0.3} #fbm=0.3

for key, val in fbm_dict.items():
    
    if args.arch =='alexnet':
        my_model = alexnet(pretrained=True)
        predify.predify(my_model, "alexnet.toml")
        from palexnet import PAlexNetSeparateHP
        pnet = PAlexNetSeparateHP(my_model, build_graph=True)
        pnet.pcoder1.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_palexnet_e10/new_pnet_pretrained_pc1_010.pth", map_location=DEVICE))
        pnet.pcoder2.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_palexnet_e10/new_pnet_pretrained_pc2_010.pth", map_location=DEVICE))
        pnet.pcoder3.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_palexnet_e10/new_pnet_pretrained_pc3_010.pth", map_location=DEVICE))
        pnet.pcoder4.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_palexnet_e10/new_pnet_pretrained_pc4_010.pth", map_location=DEVICE))
        pnet.pcoder5.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_palexnet_e10/new_pnet_pretrained_pc5_010.pth", map_location=DEVICE))

        
    elif args.arch == 'vgg16_dual':
        my_model = vgg16(pretrained=False)
        predify.predify(my_model, "vgg16_dual.toml")
        from pvgg16 import PVGG16SeparateHP
        pnet = PVGG16SeparateHP(my_model, build_graph=True)
        pnet.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_dual/dual_pvgg16_epoch15.pth", map_location=DEVICE))
    
    elif args.arch == 'vgg16':
        my_model = vgg16(pretrained=True)
        predify.predify(my_model, "vgg16.toml")
        from pvgg16 import PVGG16SeparateHP
        pnet = PVGG16SeparateHP(my_model, build_graph=True)
        pnet.pcoder1.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_imagenet/new_pvgg16_imagenet_pretrained_pc1_pmodule.pth", map_location=torch.device('cpu')))
        pnet.pcoder2.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc2_pmodule.pth", map_location=torch.device('cpu')))
        pnet.pcoder3.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc3_pmodule.pth", map_location=torch.device('cpu')))
        pnet.pcoder4.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc4_pmodule.pth", map_location=torch.device('cpu')))
        pnet.pcoder5.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pvgg16_imagenet/pvgg16_imagenet_pretrained_pc5_pmodule.pth", map_location=torch.device('cpu')))    
    
    elif args.arch == 'efficientnetb0':
        my_model = efficientnet_b0(pretrained=True)
        predify.predify(my_model, "efficient_net.toml")
        from peff_b0_v1 import PEff_b0_v1SeparateHP
        pnet = PEff_b0_v1SeparateHP(my_model, build_graph=True)
        pnet.pcoder1.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pefficientNetB0_imagenet/new_pnet_pretrained_pc1_009.pth", map_location=torch.device('cpu')))
        pnet.pcoder2.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pefficientNetB0_imagenet/new_pnet_pretrained_pc2_009.pth", map_location=torch.device('cpu')))
        pnet.pcoder3.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pefficientNetB0_imagenet/new_pnet_pretrained_pc3_009.pth", map_location=torch.device('cpu')))
        pnet.pcoder4.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pefficientNetB0_imagenet/new_pnet_pretrained_pc4_009.pth", map_location=torch.device('cpu')))
        pnet.pcoder5.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_pefficientNetB0_imagenet/new_pnet_pretrained_pc5_009.pth", map_location=torch.device('cpu')))     

    elif args.arch == 'resnet18':
        my_model = resnet18(pretrained=True)
        predify.predify(my_model, "presnet18_config.toml")
        from presnet18 import PResNet18SeparateHP
    
        pnet = PResNet18SeparateHP(my_model, build_graph=True)

        pnet.pcoder1.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_presnet18/new_pnet_pretrained_pc1_020.pth", map_location=torch.device('cuda')))
        pnet.pcoder2.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_presnet18/new_pnet_pretrained_pc2_020.pth", map_location=torch.device('cuda')))
        pnet.pcoder3.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_presnet18/new_pnet_pretrained_pc3_020.pth", map_location=torch.device('cuda')))
        pnet.pcoder4.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_presnet18/new_pnet_pretrained_pc4_020.pth", map_location=torch.device('cuda')))
        pnet.pcoder5.pmodule.load_state_dict(torch.load("/scratch/nikiguo93/weights/weights_presnet18/new_pnet_pretrained_pc5_020.pth", map_location=torch.device('cuda')))
        
    if args.arch != 'efficientnetb0' :
        hps = [
            
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
        ]
    else:
        hps = [
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
            {'ffm': 0.3, 'fbm': val, 'erm': 0.01},
    ]

    pnet.set_hyperparameters(hps)
    ext.print_hps(pnet)

    pnet.to(torch.device(DEVICE))
    pnet.build_graph = False
    pnet.eval()
    ext.print_hps(pnet)

    fts_name = args.arch
    val_loader = image_stimuli
    print(key)
    
    for filename in tqdm.tqdm(sorted(os.listdir(val_loader))):
        file_path = os.path.join(val_loader, filename)
        if file_path == f"{image_stimuli}noise_ceilin_log.json":
            continue
        dataset = ext.extract_for_timestep_n(pnet, file_path, f"Tutorial_subj{subject}_increase_fbm/{fts_name}_{key}/ts" + str(args.n) + "/", MAX_TIME_STEP=10,n=args.n, DEVICE=DEVICE, save_format=None)
