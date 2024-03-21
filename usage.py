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
import wandb
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from dotmap import DotMap
from maskingdepth import initialize
from maskingdepth.eval import visualize, eval_metric, get_eval_dict
from tqdm import tqdm
from maskingdepth import loss
from use_dust3r_multi import use_dust3r_multi
from use_dust3r_single import use_dust3r_single
from noise import image_corruption


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
            
            print(f'=============== is_multi_frame: {train_cfg.dust3r_is_multi} =============== ',  )
            print(f'=============== is_noise: {train_cfg.dust3r_is_noise} =============== ',  )
            if train_cfg.dust3r_is_noise: 
                print(f'=============== noise_type: {train_cfg.dust3r_noise_type} =============== ',  )
                print(f'=============== noise_degree: {train_cfg.dust3r_noise_degree}      =============== ',  )
            
            # add noise
            if train_cfg.dust3r_is_noise:
                
                # noise type and degree
                noise_type = train_cfg.dust3r_noise_type
                noise_degree = train_cfg.dust3r_noise_degree
                
                # tensor to numpy
                img1_np = inputs[0]['color'].squeeze(dim=0).detach().cpu().numpy()
                img2_np = inputs[1]['color'].squeeze(dim=0).detach().cpu().numpy()
                
                # numpy rescale 
                img1_np = img1_np.transpose(1,2,0) * 255
                img2_np = img2_np.transpose(1,2,0) * 255
                img1_np = img1_np.astype('uint8')
                img2_np = img2_np.astype('uint8')
                
                # add noise
                img1_corrupt = image_corruption(img1_np, noise_type, noise_degree)
                img2_corrupt = image_corruption(img2_np, noise_type, noise_degree)
                
                # numpy to tensor
                img1_t = torch.from_numpy(img1_corrupt)
                img2_t = torch.from_numpy(img2_corrupt)
                
                # tensor rescale
                inputs[0]['color'] = img1_t.permute(2,0,1).unsqueeze(dim=0).float() / 255
                inputs[1]['color'] = img2_t.permute(2,0,1).unsqueeze(dim=0).float() / 255
                
            # use DUST3R multi
            if train_cfg.dust3r_is_multi == True :
                pred_depth1, pred_depth2 = use_dust3r_multi(model, inputs)      
            # use DUST3R single
            else:
                pred_depth1, pred_depth2 = use_dust3r_single(model, inputs)

            show_depth = pred_depth1
            pred_depth1 = pred_depth1.unsqueeze(dim=0).unsqueeze(dim=0)
            pred_depth1 = F.interpolate(pred_depth1, inputs[0]['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
            gt_depth = inputs[0]['depth_gt']
            gt_color = inputs[0]['color']   

            # save image every 100 iter
            if i % 100 == 0:
                depth1_scaled = show_depth*50000
                depth1_np = depth1_scaled.detach().cpu().numpy().astype(np.uint16)
                
                if train_cfg.dust3r_is_multi == True:
                    cv2.imwrite(f'./out_depth/noise/multiFrame/{i}_multiFrame_noise({noise_type}{noise_degree})_depthmap.png', depth1_np)
                    save_image(inputs[0]['color'], f'./out_depth/noise/multiFrame/{i}_multiFrame_noise({noise_type}{noise_degree})_origImg.png')
                else:
                    cv2.imwrite(f'./out_depth/noise/singleFrame/{i}_singleFrame_noise({noise_type}{noise_degree})_depthmap.png', depth1_np)
                    save_image(inputs[0]['color'], f'./out_depth/noise/singleFrame/{i}_singleFrame_noise({noise_type}{noise_degree})_origImg.png')
                
            eval_loss += total_loss
            # pred_depth.squeeze(dim=1)은 tensor 로 (8,H,W) 이고. pred_depths 는 [] 리스트이다.
            # pred_depths.extend( pred_depth )를 해주면 pred_depth 의 8개의 이미지들이 차례로 리스트로 들어가서 리스트 len은 개가 돼
            # 즉 list = [ pred_img1(H,W), pred_img2(H,W), . . . ] 
            pred_depths.extend(pred_depth1.squeeze(1).detach().cpu().numpy())
            gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
            # JINLOVESPHO
            # pred_colors.extend(pred_color.cpu().numpy())    # 굳이 color에 대해서는 eval metric 돌릴필요 없는듯.
            # gt_colors.extend(gt_color.cpu().numpy())

        # breakpoint()
        eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
        error_dict = get_eval_dict(eval_error)
        error_dict["val_loss"] = eval_loss / len(val_loader)    

        epoch=0
        
        if train_cfg.wandb:
            error_dict["epoch"] = (epoch+1)
            wandb.log(error_dict)
                            
        else:
            progress.write(f'########################### (epoch:{epoch+1}) validation ###########################\n') 
            progress.write(f'{error_dict}\n') 
            progress.write(f'####################################################################################\n') 

    
    
    
    
        
        
        
        
        
        
        breakpoint()
        
        
