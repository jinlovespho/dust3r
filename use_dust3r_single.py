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



def use_dust3r_single(model, inputs):

        in_b, in_c, in_h, in_w = inputs[0]['color'].shape
        
        device = 'cuda'
        batch_size = in_b
        schedule = 'cosine'
        lr = 0.01
        niter = 300

        # data dir
        data_dir1 = '/mnt/ssd2/dataset/kitti_raw/2011_09_26/2011_09_26_drive_0079_sync/image_02/data'
        os.chdir(data_dir1)
        img_lst = os.listdir()
        img_lst.sort()
        os.chdir('/home/kwangrok/Downloads/VS_CODE/my_github/monodepth/dust3r')

        # inputs[0]['color'].shape = (1,3,144,512) reshaped.

        # load_images can take a list of images or a directory
        images = load_images([f'{data_dir1}/{img_lst[0]}', f'{data_dir1}/{img_lst[20]}'], size=512)
        
        # Multi view Single view 여기서 바꿔주면 된다.
        images[0]['img'] = inputs[0]['color'] 
        images[1]['img'] = inputs[0]['color'] 
        
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

        # next we'll use the global_aligner to align the predictions
        # depending on your task, you may be fine with the raw output and not need it
        # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
        # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        d1=scene.get_depthmaps()[0]
        d2=scene.get_depthmaps()[1]

        return d1, d2

        breakpoint()
        
        d1_scaled = d1*50000
        d2_scaled = d2*50000

        d1_np = d1_scaled.detach().cpu().numpy().astype(np.uint16)
        d2_np = d1_scaled.detach().cpu().numpy().astype(np.uint16)

        cv2.imwrite('d1.png', d1_np)
        cv2.imwrite('d2.png', d2_np)

        breakpoint()

        # save_image(d1, 'dmap1.png', normalize=False)
        # save_image(d1, 'dmap1_n.png', normalize=True)
        # save_image(dmap2, 'dmap2_n.png', normalize=True)
        # save_image(dmap11, 'dmap11_n.png', normalize=True)
        # save_image(dmap22, 'dmap22_n.png', normalize=True)

        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        # visualize reconstruction
        scene.show()

        # find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # visualize a few matches
        import numpy as np
        from matplotlib import pyplot as pl
        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)