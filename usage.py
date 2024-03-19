from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
from torchvision.utils import save_image 
from torchvision.io import read_image 
import cv2 
import numpy as np
import torch

# JINLOVESPHO
import argparse
import yaml
from dotmap import DotMap
from maskingdepth import initialize
import wandb
from maskingdepth.eval import visualize, eval_metric, get_eval_dict
from tqdm import tqdm
from maskingdepth import loss
import torch.nn.functional as F
from use_dust3r import use_dust3r


parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./conf/base_train.yaml")

args = parser.parse_args()


if __name__ == '__main__':
    with open(args.conf, 'r') as f:

        conf =  yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = DotMap(conf['Train'])
        device = torch.device("cuda" if train_cfg.use_cuda else "cpu")
        
        # data loader
        train_loader, val_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers)
        
        model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        model = load_model(model_path, device)
          
        # set wandb
        if train_cfg.wandb:
            wandb.init(project = train_cfg.wandb_proj_name,
                        name = train_cfg.model_name,
                        config = conf,
                        dir=train_cfg.wandb_log_path)
        # save configuration (this part activated when do not use wandb)
        else: 
            save_config_folder = os.path.join(train_cfg.log_path, train_cfg.model_name)
            if not os.path.exists(save_config_folder):
                os.makedirs(save_config_folder)
            with open(save_config_folder + '/config.yaml', 'w') as f:
                yaml.dump(conf, f)
            progress = open(save_config_folder + '/progress.txt', 'w')
        
        
        # dust3r inference test 
        # validation code from masking depth
        eval_loss = 0
        eval_error = []
        pred_depths = []
        gt_depths = []
        # JINLOVESPHO
        pred_colors = []
        gt_colors= []

        for i, inputs in enumerate(tqdm(val_loader)):
            
            total_loss = 0
            losses = {}
            
            # multiframe validation
            if train_cfg.data.dataset=='kitti_depth_multiframe':
                for input in inputs:
                    for key, ipt in input.items():
                        if type(ipt) == torch.Tensor:
                            input[key] = ipt.to(device)     # Place current and previous frames on cuda              
                # breakpoint()
                
                # use DUST3R
                pred_depth, pred_depth2 = use_dust3r(model, inputs)
                pred_depth = pred_depth.unsqueeze(dim=0).unsqueeze(dim=0)
                pred_depth = F.interpolate(pred_depth, inputs[0]['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
                gt_depth = inputs[0]['depth_gt']
                gt_color = inputs[0]['color']
                
                if i % 50 == 0:        
                    d1_scaled = pred_depth*50000
                    # d2_scaled = pred_depth2*50000
                    
                    d1_np = d1_scaled.detach().cpu().numpy().astype(np.uint16)
                    # d2_np = d2_scaled.detach().cpu().numpy().astype(np.uint16)
                        
                    cv2.imwrite(f'./out_depth/{i}_singleFrame_depthmap1.png', d1_np)
                    # cv2.imwrite(f'./out_depth/{i}_depthmap2.png', d1_np)

            # breakpoint()
            eval_loss += total_loss
            # pred_depth.squeeze(dim=1)은 tensor 로 (8,H,W) 이고. pred_depths 는 [] 리스트이다.
            # pred_depths.extend( pred_depth )를 해주면 pred_depth 의 8개의 이미지들이 차례로 리스트로 들어가서 리스트 len은 개가 돼
            # 즉 list = [ pred_img1(H,W), pred_img2(H,W), . . . ] 
            pred_depths.extend(pred_depth.squeeze(1).detach().cpu().numpy())
            gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
            # JINLOVESPHO
            # pred_colors.extend(pred_color.cpu().numpy())    # 굳이 color에 대해서는 eval metric 돌릴필요 없는듯.
            # gt_colors.extend(gt_color.cpu().numpy())

        # breakpoint()
        eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
        error_dict = get_eval_dict(eval_error)
        error_dict["val_loss"] = eval_loss / len(val_loader)    
        
        breakpoint()            

        # breakpoint()
        
        epoch=0
        
        if train_cfg.wandb:
            error_dict["epoch"] = (epoch+1)
            wandb.log(error_dict)
                            
        else:
            progress.write(f'########################### (epoch:{epoch+1}) validation ###########################\n') 
            progress.write(f'{error_dict}\n') 
            progress.write(f'####################################################################################\n') 

    
    
    
    
        
        
        
        
        
        
        breakpoint()
        
        
