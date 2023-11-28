import random

import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points

import sys
sys.path.append(".")

from lib.io_helper import load_image_as_tensor


def chamfer_distance_rgb(region_1, region_2):
    """
        compute chamfer distance in RGB space
        region_1 and region_2 should be in shape (N, 3)
    """
    loss, _ = chamfer_distance(region_1.unsqueeze(0), region_2.unsqueeze(0))

    return loss

# def compute_local_energy(
#         pred_path, targ_path, 
#         instance_mask_path, generation_mask_path, all_instance_mask_path, 
#         device, alpha=10, num_sample=5000
#     ):
    
#     pred = load_image_as_tensor(pred_path, device).float()
#     targ = load_image_as_tensor(targ_path, device).float()
#     instance_mask = load_image_as_tensor(instance_mask_path, device) # 0 - num_inst
#     all_instance_mask = load_image_as_tensor(all_instance_mask_path, device) # 0 - num_inst
#     generation_mask = load_image_as_tensor(generation_mask_path, device).float() / 255 # 0 - 1

#     all_instance_ids = torch.unique(all_instance_mask).cpu().numpy().tolist()

#     energy = 0
#     count = 0
#     for instance_id in all_instance_ids:
#         cur_instance_mask = instance_mask == instance_id
#         if instance_id == 0: continue # invalid / background
#         if cur_instance_mask.sum() == 0: continue # no such instance in view
        
#         generation_instance_mask = cur_instance_mask * generation_mask
#         exist_instance_mask = cur_instance_mask * (1 - generation_mask)

#         indices = torch.arange(generation_instance_mask.view(-1).shape[0]).reshape(generation_instance_mask.shape)
#         pred_indices = indices[generation_instance_mask == 1]
#         targ_indices = indices[exist_instance_mask == 1]

#         pred_sample_ids = random.choices(list(range(pred_indices.shape[0])), k=num_sample)
#         pred_indices = pred_indices[pred_sample_ids]

#         targ_sample_ids = random.choices(list(range(targ_indices.shape[0])), k=num_sample)
#         targ_indices = targ_indices[targ_sample_ids]

#         pred_instance = pred.permute(1, 2, 0).reshape(-1, 3)[pred_indices]
#         targ_instance = targ.permute(1, 2, 0).reshape(-1, 3)[targ_indices]

#         new_dist = chamfer_distance_rgb(pred_instance, targ_instance) / num_sample

#         # test CD within GT region
#         num_targ = targ_indices.shape[0]
#         num_selected_targ = num_targ // 2

#         indices_for_random = random.choices(list(range(num_targ)), k=num_sample*2)
#         selected_indices = indices_for_random[:num_sample]
#         not_selected_indices = indices_for_random[num_sample:]

#         targ_instance_1 = targ_instance[selected_indices]
#         targ_instance_2 = targ_instance[not_selected_indices]

#         old_dist = chamfer_distance_rgb(targ_instance_1, targ_instance_2) / num_sample

#         # E = E_new - \alpha * E_old
#         energy += new_dist - old_dist
#         count += 1

#         print(new_dist, old_dist)
    
#     energy /= count

#     return energy

def compute_local_energy(
        pred_path, targ_path, 
        instance_mask_path, generation_mask_path, all_instance_mask_path, 
        device, alpha=10, num_sample=5000
    ):
    
    source = load_image_as_tensor(targ_path, device).float() # GT
    target = load_image_as_tensor(pred_path, device).float() # prediction
    instance_mask = load_image_as_tensor(instance_mask_path, device) # 0 - num_inst
    all_instance_mask = load_image_as_tensor(all_instance_mask_path, device) # 0 - num_inst
    generation_mask = load_image_as_tensor(generation_mask_path, device).float() / 255 # 0 - 1

    all_instance_ids = torch.unique(all_instance_mask).cpu().numpy().tolist()

    energy = 0
    count = 0
    for instance_id in all_instance_ids:
        cur_instance_mask = instance_mask == instance_id
        if instance_id == 0: continue # invalid / background
        if cur_instance_mask.sum() == 0: continue # no such instance in view
        
        generation_instance_mask = cur_instance_mask * generation_mask
        exist_instance_mask = cur_instance_mask * (1 - generation_mask)

        indices = torch.arange(generation_instance_mask.view(-1).shape[0]).reshape(generation_instance_mask.shape)
        target_indices = indices[generation_instance_mask == 1]
        source_indices = indices[exist_instance_mask == 1]

        target_instance = target.permute(1, 2, 0).reshape(-1, 3)[target_indices] # N_T, 3
        source_instance = source.permute(1, 2, 0).reshape(-1, 3)[source_indices] # N_S, 3

        t2s = knn_points(target_instance.unsqueeze(0), source_instance.unsqueeze(0)).dists[0, :, 0]

        coherence = t2s.mean()

        energy += coherence
        count += 1

        print(instance_id, coherence)

        # empty_target = torch.zeros_like(target.permute(1, 2, 0).reshape(-1, 3))
        # empty_target.fill_(255)
        # empty_target[target_indices] = target_instance
        # empty_target = empty_target.reshape(768, 768, 3)

        # empty_target = Image.fromarray(empty_target.cpu().numpy().astype(np.uint8))
        # empty_target.save("{}_target.png".format(instance_id))

        # empty_source = torch.zeros_like(source.permute(1, 2, 0).reshape(-1, 3))
        # empty_source.fill_(255)
        # empty_source[source_indices] = source_instance
        # empty_source = empty_source.reshape(768, 768, 3)

        # empty_source = Image.fromarray(empty_source.cpu().numpy().astype(np.uint8))
        # empty_source.save("{}_source.png".format(instance_id))

        empty_coherent = torch.zeros(generation_instance_mask.shape[1], generation_instance_mask.shape[2]).to(generation_instance_mask.device).reshape(-1)
        empty_coherent[target_indices] = t2s
        empty_coherent = empty_coherent.reshape(768, 768)

        plt.imshow(empty_coherent.cpu().numpy().astype(np.uint8), cmap='viridis')
        plt.colorbar()
        plt.axis("off")
        plt.show()
        plt.savefig("{}_heatmap.png".format(instance_id),bbox_inches='tight')
        plt.clf()

        # empty_coherent = Image.fromarray(empty_coherent.cpu().numpy().astype(np.uint8))
        # empty_coherent.save("{}_heatmap.png".format(instance_id))


    
    energy /= count

    return energy