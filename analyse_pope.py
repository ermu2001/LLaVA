import torch
from dataclasses import dataclass
import copy

from attention_analyse import (
    visualize_spatial_energy,
    visualize_1d_energy,
    transpose_list,
    make_image_grid,
    load_image,
    Analyser,
)
device = 'cuda:1'

def replace_batch_zero_with_nonzero_min(energy, nonzero_min):
    energy = torch.where(energy == 0, torch.finfo(energy.dtype).max, energy)
    energy = torch.where(energy == torch.finfo(energy.dtype).max, nonzero_min, energy)
    return energy

def analyse_gap(analyser, queries, images, start, gap, mode='average'):
    for query, imgs in zip(queries, images):
        print('=' * 200)
        current = start
        while current - gap >= 0:
            print('-' * 200)
            print(query)
            print(imgs)
            current = current - gap
            deep_layer = current + gap
            shallow_layer = current
            # analyser.analyse(query, imgs, deep_layer, shallow_layer, mode=mode)
            
            vis_energies, all_energies = analyser.analyse(query, imgs, deep_layer, shallow_layer, mode=mode)
            
            layers_instances_vis_pils, layers_instances_text_pils = [], []
            for vis_energy, energy in zip(vis_energies, all_energies):
                # fill with the none zero min
                temp = torch.where(energy == 0, torch.finfo(energy.dtype).max, energy)

                nonzero_min = temp.min(1, keepdim=True)[0]
                energy = replace_batch_zero_with_nonzero_min(energy, nonzero_min)
                vis_energy = replace_batch_zero_with_nonzero_min(vis_energy, nonzero_min)
                energy = energy.log()
                vis_energy = vis_energy.log()
                mean, std = energy.mean(1, keepdim=True), energy.std(1, keepdim=True)
                min_, max_ = mean-3*std, mean+3*std
                # min_, _ = torch.min(energy, keepdim=True, dim=1)
                # max_, _ = torch.max(energy, keepdim=True, dim=1)
                energy = (energy - min_) / (max_ - min_)
                vis_energy = (vis_energy - min_) / (max_ - min_) 
                energy = energy.clip(0, 1)
                vis_energy = vis_energy.clip(0, 1)
                
                layer_vis_pils = visualize_spatial_energy(vis_energy)
                layer_text_pils = visualize_1d_energy(energy) 
                layers_instances_vis_pils.append(layer_vis_pils)
                layers_instances_text_pils.append(layer_text_pils)
            
            instances_layers_vis_pils = transpose_list(layers_instances_vis_pils)
            instances_layers_text_pils = transpose_list(layers_instances_text_pils)
            
            # display(make_image_grid(instances_layers_vis_pils[0], resize=128)) # index by batch
            # display(make_image_grid( instances_layers_text_pils[0], cols=1, resize=(2000, 20)))
            return instances_layers_vis_pils, instances_layers_text_pils

def main():
    @dataclass
    class ARGS():
        model_path="liuhaotian/llava-v1.5-7b"
        model_base=None
        model_name=None
        device=device
        multi_modal=True
        load_8bit=False
        load_4bit=False
    args = ARGS() 
    print(args.model_path)
    analyser = Analyser(args)
    analyser.load_from_args()
    model, tokenizer, image_processor = analyser.model, analyser.tokenizer, analyser.image_processor

    temp_tokenizer = copy.deepcopy(tokenizer)
    temp_tokenizer.add_tokens("<image>")
    image_token_id = temp_tokenizer.added_tokens_encoder['<image>']
    analyser = Analyser(args)
    analyser.set_modules(model, temp_tokenizer, image_processor)


    queries = [
        ['Base on this input image, tell me who is the author of the painting?', 'The author of the painting is Leonardo Da Vinci.', 'Leonardo Da Vinci'], # Model Output
        ['Base on this input image, tell me who is the author of the painting?', 'The author of the painting is Claude Monet.', 'Claude Monet'], # Injected Halu
        # ['Base on this input image, tell me who is the author of the painting?', 'The painting is the famous Monalisa, and the author is Da Vinci', 'Da Vinci '], # Made up Output    
    ]
    images = [['https://llava-vl.github.io/static/images/monalisa.jpg'],] * len(queries)
    images = [list(map(load_image, i) ) for i in images]
    

    instances_layers_vis_pils, instances_layers_text_pils = analyse_gap(analyser, queries, images, 32, 2, mode='average')