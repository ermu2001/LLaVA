
import os
from typing import Union, List

import PIL.Image
from PIL import Image
import PIL.ImageOps
import requests


import io
import itertools
import math
import numpy as np
import requests
import torch
from torchvision.transforms.functional import to_pil_image

import transformers
from dataclasses import dataclass

from llava.model.builder import load_pretrained_model
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria





def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.

    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image



def make_image_grid(images: List[PIL.Image.Image], cols: int=10,rows: int=None, resize: int = None) -> PIL.Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    if rows is None:
        rows = math.ceil(len(images) / cols)
    # assert len(images) == rows * cols

    if resize is not None:
        if isinstance(resize, int):
            images = [img.resize((resize, resize), PIL.Image.NEAREST) for img in images]
        elif isinstance(resize, tuple):
            images = [img.resize((resize), PIL.Image.NEAREST) for img in images]
            
    w, h = images[0].size
    grid = PIL.Image.new("RGB", size=(cols * w, rows * h), color=(255, 255, 255))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def transpose_list(l):
    return list(map(list, zip(*l)))

def visualize_spatial_energy(energy, min_, max_, shape=None):
    b, seq_len = energy.shape
    if shape is None:
        l = math.isqrt(seq_len)
        shape = (l, l)
        
    if math.prod(shape) != seq_len:
        raise ValueError('')

    energy = (energy - min_) * 255 / (max_ - min_)
    energy = energy.clip(0, 255)
    energy = energy.reshape(b, *shape)
    # energy = energy.expand(2, -1, -1).unsqueeze(1)
    # print(energy.shape)
    
    energy_map_pils = [to_pil_image(e) for e in energy]
    
    return energy_map_pils

def visualize_1d_energy(energy, min_, max_):

    b, seq_len = energy.shape
        
    energy = (energy - min_) * 255 / (max_ - min_)
    energy = energy.clip(0, 255)
    energy = energy.reshape(b, 1, seq_len)
    
    energy_map_pils = [to_pil_image(e) for e in energy]
    
    return energy_map_pils


class Analyser():
    is_multimodal=True
    def __init__(self, args):
        self.set_args(args)
        
    def set_args(self, args):
        self.args = args
        model_path = args.model_path
        if model_path.endswith("/"):
            model_path = model_path[:-1]

        if args.model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = args.model_name
        self.device = self.args.device
    
    def load_from_args(self):
        model_path = self.args.model_path
        model_base = self.args.model_base
        load_8bit = self.args.load_8bit
        load_4bit = self.args.load_4bit
        model_name = self.model_name
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = 'llava' in self.model_name.lower()

    def set_modules(self, model=None, tokenizer=None, image_processor=None):
        self.model = model if model is not None else self.model
        self.tokenizer = tokenizer if tokenizer is not None else self.tokenizer
        self.image_processor = image_processor if image_processor is not None else self.image_processor
    
    @torch.inference_mode()
    def analyse_attention(self, start_energy, attentions):
        from_seq_len = start_energy.shape[2]
        # energy_transforme_matrix = torch.diag(torch.ones(to_seq_len))
        # energy_transforme_matrix = torch.unsqueeze(energy_transforme_matrix, 0)
        # print(energy_transforme_matrix.shape)
        # return
        energies = []
        energy = start_energy
        for layer_attention in attentions:
            layer_attention = layer_attention.mean(1) # mean over multi heads
            if layer_attention.shape[1] != from_seq_len:
                raise ValueError('')
            
            # print(layer_attention.shape)
            # print(layer_attention.sum(2)) # all one
            # print(energy.shape)
            # print(energy.sum(1))
            energy = energy @ layer_attention
            energies.append(energy)
        return energies

    @torch.inference_mode()
    def run4attention(self, input_ids, images):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        images = process_images(images, image_processor, model.config)
        if type(images) is list:
            images = [image.to(self.model.device, dtype=torch.float16) for image in images]
        else:
            images = images.to(self.model.device, dtype=torch.float16)
    
        lm_model_out = model(
            input_ids=input_ids,
            images=images,
            output_attentions=True,
        )
        lm_attentions = lm_model_out.attentions
        
        vision_tower = model.get_vision_tower().vision_tower
        vision_model_out = vision_tower(images, output_attentions=True)
        vision_attentions = vision_model_out.attentions
        return lm_attentions, vision_attentions

    @torch.inference_mode()
    def generate(self, query, images, **params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        
        # using conversation_lib to preprocess the text input
        conv = conversation_lib.conv_llava_v1.copy()
        user_input, assistant_output, target_text = query
        conv.append_message(conv.roles[0], (user_input, images[0]))
        # conv.append_message(conv.roles[1], assistant_output)
        prompt = conv.get_prompt()
        ori_prompt = prompt
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", "</s>")
        do_sample = True if temperature > 0.001 else False

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)

        return model.generate(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        )
    
    def construct_target_indexs(self, input_ids, response_token_ids):
        b, seq_len = input_ids.shape
        select_feature = self.model.get_vision_tower().select_feature
        num_vis_tokens = self.model.get_vision_tower().num_patches if select_feature != 'cls_patch' else self.model.get_vision_tower().num_patches + 1
        batch_token_index, vis_token_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)
        vis_token_index = vis_token_index
        assert torch.all(batch_token_index == torch.arange(b)), 'only support each instance containing one image '
        print(vis_token_index)
        # under a batch style
        input_ids_unsqueezed = input_ids.unsqueeze(2)
        response_token_ids_unsqueezed = response_token_ids.unsqueeze(1)
        batch_indexs, token_indexs = torch.any(input_ids_unsqueezed == response_token_ids_unsqueezed, dim=2).nonzero(as_tuple=True)
        token_indexs = torch.where(token_indexs > vis_token_index, token_indexs + num_vis_tokens - 1, token_indexs)
        from_indexs = (batch_indexs, token_indexs)
    
        to_indexs_vis = (torch.repeat_interleave(batch_token_index, num_vis_tokens), torch.arange(num_vis_tokens).unsqueeze(0).add(vis_token_index.repeat_interleave(b)).mT.flatten())
        # print( (input_ids == input_ids).nonzero(as_tuple=True))
        # print( (input_ids == input_ids).shape)
        batch_indexs, token_indexs = (input_ids == input_ids).nonzero(as_tuple=True)
        token_indexs = torch.where(token_indexs > vis_token_index, token_indexs + num_vis_tokens - 1, token_indexs)
        to_indexs_text = (batch_indexs, token_indexs)
        
        return from_indexs, to_indexs_vis, to_indexs_text

    # only single round analyse, supports 
    def analyse(self, query, images, deep_layer=-1, shallow_layer=0, mode='multiple'):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        
        # using conversation_lib to preprocess the text input
        conv = conversation_lib.conv_llava_v1.copy()
        user_input, assistant_output, target_text = query
        conv.append_message(conv.roles[0], (user_input, images[0]))
        conv.append_message(conv.roles[1], assistant_output)
        prompt = conv.get_prompt()
        print('model input:', prompt)
    
        # 
        response_token_ids = tokenizer(target_text, return_tensors='pt', add_special_tokens=False).input_ids
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        
        # torch.cuda.empty_cache()
        lm_attentions, vision_attentions = self.run4attention(
            input_ids.to(device),
            images,
        )
        
        # lm_attentions = [attention.to(dtype=torch.float32) for attention in lm_attentions]
        # vision_attentions = [attention.to(dtype=torch.float32) for attention in vision_attentions]
        attentions = list(itertools.chain(lm_attentions, vision_attentions))
        
        first_attention = lm_attentions[0]
        b, first_num_heads, from_seq_len, to_seq_len = first_attention.shape
        
        print(to_seq_len)
        
        from_indexs, to_indexs_vis, to_indexs_text = self.construct_target_indexs(input_ids, response_token_ids)
        
        start_energy = torch.zeros((b, to_seq_len)).to(first_attention.device, first_attention.dtype)
        start_energy[from_indexs] = 1
        start_energy = start_energy.unsqueeze(1) # for batch process
        start_energy = start_energy / start_energy.sum(2, keepdim=True)

        assert deep_layer > shallow_layer or deep_layer + shallow_layer < 0, ''
        print(f'attention visualizing in layers:deep_layer{deep_layer} - shallow_layer{shallow_layer}')

        if mode == 'multiple':
            lm_attentions = lm_attentions[deep_layer:shallow_layer:-1]
        elif mode == 'average':
            lm_attentions = lm_attentions[deep_layer:shallow_layer:-1]
            lm_attentions = [torch.stack(lm_attentions).mean(0)]
            
        energies = self.analyse_attention(start_energy, lm_attentions)
        layers_instances_vis_pils = []
        layers_instances_text_pils = []
        for i, energy in enumerate(energies):
            energy = energy[:, 0, :]
            # print("sum to one", energy.sum(1)) # sum to one
            mean, std = energy.mean(1, keepdim=True), energy.std(1, keepdim=True)
            min_, max_ = mean-3*std, mean+3*std
            # print(min_, max_)
            num_vis_tok = (to_indexs_vis[0]==0).sum()
            vis_energy = [energy[i, to_indexs_vis[1][i * num_vis_tok],...] for i in range(b)]
            vis_energy = energy[:, to_indexs_vis[1]]
            text_energy = energy
            layer_vis_pils = visualize_spatial_energy(vis_energy, min_=min_, max_=max_)
            layer_text_pils = visualize_1d_energy(text_energy, min_=min_, max_=max_)
            layers_instances_vis_pils.append(layer_vis_pils)
            layers_instances_text_pils.append(layer_text_pils)
        
        instances_layers_vis_pils = transpose_list(layers_instances_vis_pils)
        instances_layers_text_pils = transpose_list(layers_instances_text_pils)
        
        display(make_image_grid(instances_layers_vis_pils[0], resize=128))
        display(make_image_grid( instances_layers_text_pils[0], cols=1, resize=(2000, 20)))
        return energies