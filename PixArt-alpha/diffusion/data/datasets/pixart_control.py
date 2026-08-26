import os
import random
import ast
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torch.utils.data import Dataset
from diffusers.utils.torch_utils import randn_tensor
from torchvision import transforms as T
from diffusion.data.builder import get_data_path, DATASETS

import json, time

_RLC = Path(__file__).resolve().parents[4]
if str(_RLC) not in sys.path:
    sys.path.insert(0, str(_RLC))
from long_prompt_segmentation import (  # noqa: E402
    parse_long_prompt_bracket_segments,
    compose_two_part_prompt,
    process_long_prompt_segmented,
)


def reconstruct_prompt_from_tags(long_prompt_text, tags):
    """Reconstruct the "first sentence + selected segment" prompt from the selected [] tag list."""
    if not tags:
        return None
    first, segs = parse_long_prompt_bracket_segments(long_prompt_text)
    for seg in segs:
        if sorted(seg['tags']) == sorted(tags):
            return compose_two_part_prompt(first, seg)
    return None


def reconstruct_prompt_legacy_sentence_split(long_prompt_text, sentence_1based_idx):
    """Legacy sentence join when random_idx_list.txt contains only numbers (split by periods)."""
    sentences = [s.strip() for s in long_prompt_text.split('.') if s.strip()]
    if len(sentences) < 2 or not (1 <= sentence_1based_idx <= len(sentences) - 1):
        return None
    first_sentence = sentences[0]
    random_sentence = sentences[sentence_1based_idx]
    if not first_sentence.endswith('.'):
        first_sentence = f"{first_sentence}."
    if not random_sentence.endswith('.'):
        random_sentence = f"{random_sentence}."
    return f"{first_sentence} {random_sentence}"


@DATASETS.register_module()
class InternalDataControl(Dataset):
    def __init__(self,
                 root,
                 image_list_json='data_info.json',
                 transform=None,
                 resolution=256,
                 sample_subset=None,
                 load_vae_feat=False,
                 input_size=32,
                 patch_size=2,
                 mask_ratio=0.0,
                 load_mask_index=False,
                 train_ratio=1.0,
                 mode='train',
                 long_prompt_ratio=None,
                 use_long_prompt=None,
                 prompt_feature_dir='prompt_feature',
                 prompt_feature_long_dir='prompt_feature_long',
                 controlnet_modality='hed',  # New param: control modality type, default 'hed'
                 controlnet_feat_dir=None,  # New param: custom feature directory name; if None, use default naming
                 controlnet_feat_dir_2=None,  # Second-branch control feature directory (alongside condition / c, for dual-branch VAE fusion training)
                 t5_feat_dir=None,  # Compat: when long_prompt_ratio in {0,1} only, map to the corresponding single path; for mixed training use prompt_feature_dir / prompt_feature_long_dir
                 metadata_json_subdir='partition_filter',
                 **kwargs):
        self.root = get_data_path(root)
        self.transform = transform
        self.load_vae_feat = load_vae_feat
        self.ori_imgs_nums = 0
        self.resolution = resolution
        self.N = int(resolution // (input_size // patch_size))
        self.mask_ratio = mask_ratio
        self.load_mask_index = load_mask_index
        legacy_use_long = kwargs.pop('use_long_prompt', None)
        if use_long_prompt is None:
            use_long_prompt = legacy_use_long
        if long_prompt_ratio is not None:
            long_prompt_ratio = float(long_prompt_ratio)
        elif use_long_prompt is not None:
            long_prompt_ratio = 1.0 if use_long_prompt else 0.0
        else:
            long_prompt_ratio = 0.0
        if not (0.0 <= long_prompt_ratio <= 1.0):
            raise ValueError(f'long_prompt_ratio must be in [0, 1], got {long_prompt_ratio}')
        self.long_prompt_ratio = long_prompt_ratio
        self.prompt_feature_dir = prompt_feature_dir
        self.prompt_feature_long_dir = prompt_feature_long_dir
        if t5_feat_dir is not None:
            if 0.0 < long_prompt_ratio < 1.0:
                print('[WARN] t5_feat_dir is ignored when 0 < long_prompt_ratio < 1; '
                      'use prompt_feature_dir and prompt_feature_long_dir under InternData.')
            elif long_prompt_ratio >= 1.0:
                self.prompt_feature_long_dir = t5_feat_dir
            else:
                self.prompt_feature_dir = t5_feat_dir
        self.controlnet_modality = controlnet_modality.lower()  # Modality type: 'hed', 'canny', 'depth', 'pose', 'seg', 'normal', 'openpose', 'lineart', 'scribble', 'mlsd', 'anime', 'custom', etc.
        self.controlnet_feat_dir = controlnet_feat_dir  # Custom feature directory name
        self.controlnet_feat_dir_2 = controlnet_feat_dir_2  # Second-branch feature directory (optional)
        self.t5_feat_dir = t5_feat_dir  # Custom T5 feature directory name
        self.zero_text_training = bool(kwargs.get('zero_text_training', False))
        self.meta_data_clean = []
        self.img_samples = []
        self._need_short = (long_prompt_ratio < 1.0) and (not self.zero_text_training)
        self._need_long = (long_prompt_ratio > 0.0) and (not self.zero_text_training)
        self.txt_feat_samples_short = [] if self._need_short else None
        self.txt_feat_samples_long = [] if self._need_long else None
        self.prompt_samples_short = [] if self._need_short else None
        self.prompt_samples_long = [] if self._need_long else None
        self.vae_feat_samples = []
        self.control_feat_samples = []  # First-branch control feature paths (condition)
        self.control_feat_samples_2 = []  # Second-branch control feature paths (condition2)
        self.random_idx_samples = []  # For long branch: SegFace mask region index; short branch overrides to -1 in getdata
        self.random_tags_samples = []  # For long branch; short branch overrides to [] in getdata
        self.random_idx_dict = {}  # path -> legacy mask_idx（int）
        self.random_tags_dict = {}  # path -> selected segment tag list (matches random_idx_list.txt)
        
        # Determine feature directory name from modality type
        if self.controlnet_feat_dir is None:
            # Default naming: {modality}_feature_{resolution}
            self.controlnet_feat_dir = f'{self.controlnet_modality}_feature_{resolution}'
        print(f"[INFO] ControlNet modality: {self.controlnet_modality}, feature directory: {self.controlnet_feat_dir}")
        if self.controlnet_feat_dir_2:
            print(f"[INFO] ControlNet second branch (condition2): {self.controlnet_feat_dir_2}")
        if self.zero_text_training:
            print("[INFO] zero_text_training enabled in dataset: skip loading prompt_feature*.npz files")

        _ds_root = os.path.dirname(self.root)
        # random_idx_list.txt lives alongside long features (prompt_feature_long_dir)
        random_idx_file = None
        if self._need_long:
            long_file = os.path.join(self.root, self.prompt_feature_long_dir, 'random_idx_list.txt')
            if os.path.exists(long_file):
                random_idx_file = long_file
        
        if random_idx_file is not None and os.path.exists(random_idx_file):
            print(f"[DEBUG Dataset Init] Loading random_idx_list from: {random_idx_file}")
            try:
                with open(random_idx_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if ':' not in line:
                            continue
                        path, rest = line.split(':', 1)
                        path = path.strip()
                        rest = rest.strip()
                        if not rest:
                            continue
                        if rest.startswith('['):
                            try:
                                tags = ast.literal_eval(rest)
                                self.random_tags_dict[path] = tags
                                self.random_idx_dict[path] = -1
                            except Exception:
                                self.random_idx_dict[path] = -1
                        else:
                            try:
                                self.random_idx_dict[path] = int(rest)
                            except ValueError:
                                self.random_idx_dict[path] = -1
                print(f"[DEBUG Dataset Init] Loaded {len(self.random_idx_dict)} random_idx entries from file "
                      f"({len(self.random_tags_dict)} with bracket tags)")
            except Exception as e:
                print(f"[WARNING] Failed to load random_idx_list.txt: {e}. Will use random generation instead.")
                self.random_idx_dict = {}
                self.random_tags_dict = {}

        image_list_json = image_list_json if isinstance(image_list_json, list) else [image_list_json]
        for json_file in image_list_json:
            # Detect metadata.json vs data_info.json format
            if json_file == 'metadata.json' or 'metadata' in json_file.lower():
                # Use metadata.json directly: data_root/metadata.json
                metadata_path = os.path.join(os.path.dirname(self.root), json_file)
                meta_data = self.load_json(metadata_path)
                
                # Convert format: keep only samples with long_prompt; map gt_image->path
                meta_data_clean = [
                    {
                        'path': item['gt_image'],
                        'prompt': item.get('prompt', ''),
                        'long_prompt': item['long_prompt'],
                        'height': resolution,
                        'width': resolution,
                        'ratio': 1.0
                    }
                    for item in meta_data if item.get('long_prompt')
                ]
                self.ori_imgs_nums += len(meta_data)
            else:
                # data_info.json: default InternData/partition_filter/; or dataset root (sibling of CelebA-HQ-img)
                if metadata_json_subdir == 'dataset_root':
                    meta_json_path = os.path.join(_ds_root, json_file)
                else:
                    meta_json_path = os.path.join(self.root, metadata_json_subdir, json_file)
                print(f"[INFO] Loading metadata: {meta_json_path} (metadata_json_subdir={metadata_json_subdir!r})")
                meta_data = self.load_json(meta_json_path)
                meta_data_clean = [item for item in meta_data if item.get('ratio', 1.0) <= 4]
                self.ori_imgs_nums += len(meta_data)
            # Debug: check whether the first sample has long_prompt
            if len(meta_data_clean) > 0:
                first_item = meta_data_clean[0]
                has_long_prompt = 'long_prompt' in first_item and first_item.get('long_prompt')
                print(f"[DEBUG Dataset Init] First sample has long_prompt: {has_long_prompt}, "
                      f"long_prompt_ratio={self.long_prompt_ratio}")
                if has_long_prompt:
                    long_prompt_sample = first_item['long_prompt']
                    sentences = [s.strip() for s in long_prompt_sample.split('.') if s.strip()]
                    print(f"[DEBUG Dataset Init] Sample long_prompt has {len(sentences)} sentences")
            self.meta_data_clean.extend(meta_data_clean)
            # For metadata.json format, paths are relative to dataset_train (e.g. celebA/xxx.png)
            # so use dataset_train as the root directory, not InternImgs
            # Build image paths uniformly as data_root + path
            # Do not use InternImgs (that directory does not exist)
            data_root = os.path.dirname(self.root)  # dataset_train
            self.img_samples.extend([os.path.join(data_root, item['path']) for item in meta_data_clean])
            short_dir_path = os.path.join(self.root, self.prompt_feature_dir)
            long_dir_path = os.path.join(self.root, self.prompt_feature_long_dir)
            if self._need_short:
                print(f"[DEBUG Dataset Init] Short T5 features: {short_dir_path}")
                if not os.path.exists(short_dir_path):
                    print(f"[WARNING] Short T5 feature directory does not exist: {short_dir_path}")
            if self._need_long:
                print(f"[DEBUG Dataset Init] Long T5 features: {long_dir_path}")
                if not os.path.exists(long_dir_path):
                    print(f"[WARNING] Long T5 feature directory does not exist: {long_dir_path}")
            # Determine feature file extension from modality (from kwargs or default)
            controlnet_feat_ext = kwargs.get('controlnet_feat_ext', '.npz')  # default .npz
            for item in meta_data_clean:
                filename = os.path.basename(item['path'])  # Get '13400_original.png' from 'celebA/13400_original.png'
                txt_feat_name = filename.replace('.png', '.npz').replace('.jpg', '.npz')
                vae_feat_name = filename.replace('.png', '.npy').replace('.jpg', '.npy')
                controlnet_feat_name = filename.replace('.png', controlnet_feat_ext).replace('.jpg', controlnet_feat_ext)
                if self._need_short:
                    self.txt_feat_samples_short.append(os.path.join(self.root, self.prompt_feature_dir, txt_feat_name))
                if self._need_long:
                    self.txt_feat_samples_long.append(os.path.join(self.root, self.prompt_feature_long_dir, txt_feat_name))
                self.vae_feat_samples.append(os.path.join(self.root, f'img_vae_features_{resolution}/noflip', vae_feat_name))
                self.control_feat_samples.append(os.path.join(self.root, self.controlnet_feat_dir, controlnet_feat_name))
                if self.controlnet_feat_dir_2:
                    self.control_feat_samples_2.append(os.path.join(self.root, self.controlnet_feat_dir_2, controlnet_feat_name))
            if self._need_long:
                processed_prompts = []
                random_idx_samples = []
                random_tags_samples = []
                long_prompt_count = 0
                valid_random_idx_count = 0
                loaded_from_file_count = 0
                for item in meta_data_clean:
                    long_prompt = item.get('long_prompt', None)
                    item_path = item.get('path', '')
                    file_tags = self.random_tags_dict.get(item_path)
                    file_mask_idx = self.random_idx_dict.get(item_path)

                    if long_prompt:
                        long_prompt_count += 1
                        if file_tags is not None:
                            loaded_from_file_count += 1
                            pr = reconstruct_prompt_from_tags(long_prompt, file_tags)
                            processed_prompts.append(pr if pr else item.get('prompt', ''))
                            random_idx_samples.append(-1)
                            random_tags_samples.append(list(file_tags))
                        elif file_mask_idx is not None:
                            loaded_from_file_count += 1
                            pr = reconstruct_prompt_legacy_sentence_split(long_prompt, file_mask_idx)
                            if pr is None:
                                pr = item.get('prompt', '')
                            processed_prompts.append(pr)
                            random_idx_samples.append(max(0, file_mask_idx - 1))
                            random_tags_samples.append([])
                            valid_random_idx_count += 1
                        else:
                            pp, _tags = process_long_prompt_segmented(long_prompt, item_path)
                            processed_prompts.append(pp if pp else item.get('prompt', ''))
                            random_idx_samples.append(-1)
                            random_tags_samples.append(list(_tags) if _tags else [])
                    else:
                        processed_prompts.append(item.get('prompt', ''))
                        random_idx_samples.append(-1)
                        random_tags_samples.append([])
                self.prompt_samples_long.extend(processed_prompts)
                self.random_idx_samples.extend(random_idx_samples)
                self.random_tags_samples.extend(random_tags_samples)
                print(f"[DEBUG Dataset Init] long branch prompts: total samples: {len(meta_data_clean)}, "
                      f"long_prompt found: {long_prompt_count}, valid random_idx: {valid_random_idx_count}, "
                      f"loaded from file: {loaded_from_file_count}, "
                      f"first 10 random_tags_samples: {random_tags_samples[:10]}")
            if self._need_short:
                self.prompt_samples_short.extend([item['prompt'] for item in meta_data_clean])
                if not self._need_long:
                    self.random_idx_samples.extend([-1] * len(meta_data_clean))
                    self.random_tags_samples.extend([[]] * len(meta_data_clean))

        total_sample = len(self.img_samples)
        used_sample_num = int(total_sample * train_ratio)
        print("using mode", mode)
        if mode == 'train':
            self.img_samples = self.img_samples[:used_sample_num]
            if self.txt_feat_samples_short is not None:
                self.txt_feat_samples_short = self.txt_feat_samples_short[:used_sample_num]
            if self.txt_feat_samples_long is not None:
                self.txt_feat_samples_long = self.txt_feat_samples_long[:used_sample_num]
            self.vae_feat_samples = self.vae_feat_samples[:used_sample_num]
            self.control_feat_samples = self.control_feat_samples[:used_sample_num]
            if self.control_feat_samples_2:
                self.control_feat_samples_2 = self.control_feat_samples_2[:used_sample_num]
            if self.prompt_samples_short is not None:
                self.prompt_samples_short = self.prompt_samples_short[:used_sample_num]
            if self.prompt_samples_long is not None:
                self.prompt_samples_long = self.prompt_samples_long[:used_sample_num]
            self.random_idx_samples = self.random_idx_samples[:used_sample_num]
            self.random_tags_samples = self.random_tags_samples[:used_sample_num]
            self.meta_data_clean = self.meta_data_clean[:used_sample_num]  # Keep meta_data_clean sliced in sync
        else:
            self.img_samples = self.img_samples[-used_sample_num:]
            if self.txt_feat_samples_short is not None:
                self.txt_feat_samples_short = self.txt_feat_samples_short[-used_sample_num:]
            if self.txt_feat_samples_long is not None:
                self.txt_feat_samples_long = self.txt_feat_samples_long[-used_sample_num:]
            self.vae_feat_samples = self.vae_feat_samples[-used_sample_num:]
            self.control_feat_samples = self.control_feat_samples[-used_sample_num:]
            if self.control_feat_samples_2:
                self.control_feat_samples_2 = self.control_feat_samples_2[-used_sample_num:]
            if self.prompt_samples_short is not None:
                self.prompt_samples_short = self.prompt_samples_short[-used_sample_num:]
            if self.prompt_samples_long is not None:
                self.prompt_samples_long = self.prompt_samples_long[-used_sample_num:]
            self.random_idx_samples = self.random_idx_samples[-used_sample_num:]
            self.random_tags_samples = self.random_tags_samples[-used_sample_num:]
            self.meta_data_clean = self.meta_data_clean[-used_sample_num:]  # Keep meta_data_clean sliced in sync

        # Set loader and extensions
        if load_vae_feat:
            self.transform = None
            self.loader = self.vae_feat_loader
        else:
            self.loader = default_loader

        if sample_subset is not None:
            self.sample_subset(sample_subset)  # sample dataset for local debug

    def getdata(self, index):
        img_path = self.img_samples[index]
        npy_path = self.vae_feat_samples[index]
        controlnet_feat_path = self.control_feat_samples[index]
        if self.zero_text_training:
            use_long = False
        elif self._need_short and self._need_long:
            use_long = random.random() < self.long_prompt_ratio
        elif self._need_long:
            use_long = True
        else:
            use_long = False
        npz_path = None
        if self.zero_text_training:
            prompt = ""
            random_idx = -1
            random_tags = []
        elif use_long:
            npz_path = self.txt_feat_samples_long[index]
            prompt = self.prompt_samples_long[index]
            random_idx = self.random_idx_samples[index] if index < len(self.random_idx_samples) else -1
            random_tags = list(self.random_tags_samples[index]) if index < len(self.random_tags_samples) else []
        else:
            npz_path = self.txt_feat_samples_short[index]
            prompt = self.prompt_samples_short[index]
            random_idx = -1
            random_tags = []
        # only trained on single-scale 1024 res data
        data_info = {'img_hw': torch.tensor([1024., 1024.], dtype=torch.float32), 'aspect_ratio': torch.tensor(1.)}

        if self.load_vae_feat:
            img = self.loader(npy_path)
        else:
            img = self.loader(img_path)
        
        # Choose loader by file extension
        if controlnet_feat_path.endswith('.npz'):
            controlnet_fea = self.vae_feat_loader_npz(controlnet_feat_path)
        elif controlnet_feat_path.endswith('.npy'):
            controlnet_fea = self.vae_feat_loader(controlnet_feat_path)
        else:
            # Try auto-detecting format
            try:
                controlnet_fea = self.vae_feat_loader_npz(controlnet_feat_path)
            except:
                controlnet_fea = self.vae_feat_loader(controlnet_feat_path)
        
        if self.zero_text_training:
            # Placeholder tensors only for DataLoader collation; training side overrides with a global fixed T5 feature.
            txt_fea = torch.zeros((1, 1, 4096), dtype=torch.float32)
            attention_mask = torch.ones((1, 1, 1), dtype=torch.float32)
        else:
            txt_info = np.load(npz_path)
            txt_fea = torch.from_numpy(txt_info['caption_feature'])
            attention_mask = torch.ones(1, 1, txt_fea.shape[1])
            if 'attention_mask' in txt_info.keys():
                attention_mask = torch.from_numpy(txt_info['attention_mask'])[None]

        if self.transform:
            img = self.transform(img)

        data_info['condition'] = controlnet_fea
        if self.control_feat_samples_2:
            controlnet_feat_path_2 = self.control_feat_samples_2[index]
            if controlnet_feat_path_2.endswith('.npz'):
                controlnet_fea2 = self.vae_feat_loader_npz(controlnet_feat_path_2)
            elif controlnet_feat_path_2.endswith('.npy'):
                controlnet_fea2 = self.vae_feat_loader(controlnet_feat_path_2)
            else:
                try:
                    controlnet_fea2 = self.vae_feat_loader_npz(controlnet_feat_path_2)
                except Exception:
                    controlnet_fea2 = self.vae_feat_loader(controlnet_feat_path_2)
            data_info['condition2'] = controlnet_fea2
        data_info['prompt'] = prompt
        data_info['random_idx'] = random_idx  # Keep only fixed-length/scalar fields to avoid DataLoader collate failure on variable-length lists
        # Use a delimiter-joined string to avoid variable-length list collation issues.
        # Training side parses this field back to a tag list.
        data_info['random_tags'] = "|".join(
            [str(t).strip() for t in random_tags if str(t).strip()]
        )
        data_info['img_path'] = img_path  # Save image path for mask generation
        # Save original path field (relative path in metadata, e.g. "celebA/4983_original.png")
        path_value = None
        if index < len(self.meta_data_clean):
            path_value = self.meta_data_clean[index].get('path', None)
        
        # If path is None, try extracting relative path from img_path
        if path_value is None or not isinstance(path_value, str):
            # Extract relative path from img_path
            if 'dataset_train' in img_path:
                path_value = img_path.split('dataset_train/')[-1]
            elif 'celebA' in img_path:
                path_value = 'celebA/' + os.path.basename(img_path)
            # Debug: if path is None, print a warning
            if path_value is None and index == 0 and not hasattr(self, '_path_warning_logged'):
                print(f"[DEBUG Dataset] Warning: path is None for index {index}, img_path={img_path}, meta_data_clean length={len(self.meta_data_clean)}")
                if index < len(self.meta_data_clean):
                    print(f"[DEBUG Dataset] meta_data_clean[{index}] keys: {list(self.meta_data_clean[index].keys())}")
                self._path_warning_logged = True
        
        data_info['path'] = path_value
        
        # Debug: print only for the first sample
        if index == 0 and not hasattr(self, '_debug_logged'):
            print(f"[DEBUG Dataset] Sample 0: random_tags={list(random_tags) if isinstance(random_tags, (list, tuple)) else []}, random_idx={random_idx}, img_path exists={os.path.exists(img_path)}, long_prompt_ratio={self.long_prompt_ratio}")
            print(f"[DEBUG Dataset] random_idx_samples length: {len(self.random_idx_samples)}, total samples: {len(self.img_samples)}")
            if index < len(self.random_tags_samples):
                print(f"[DEBUG Dataset] First 5 random_tags_samples: {self.random_tags_samples[:5]}")
            self._debug_logged = True
        
        return img, txt_fea, attention_mask, data_info

    def __getitem__(self, idx):
        for i in range(20):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                import traceback
                print(f"Error details: {str(e)}")
                print(f"Attempt {i+1}/20, index: {idx}")
                print(f"VAE path: {self.vae_feat_samples[idx] if idx < len(self.vae_feat_samples) else 'N/A'}")
                _ts = self.txt_feat_samples_short[idx] if self.txt_feat_samples_short else None
                _tl = self.txt_feat_samples_long[idx] if self.txt_feat_samples_long else None
                print(f"TXT path (short/long): {_ts!r} / {_tl!r}")
                print(f"ControlNet {self.controlnet_modality} path: {self.control_feat_samples[idx] if idx < len(self.control_feat_samples) else 'N/A'}")
                traceback.print_exc()
                idx = np.random.randint(len(self))
        raise RuntimeError('Too many bad data.')

    def get_data_info(self, idx):
        data_info = self.meta_data_clean[idx]
        return {'height': data_info['height'], 'width': data_info['width']}

    @staticmethod
    def vae_feat_loader(path):
        # [mean, std]
        mean, std = torch.from_numpy(np.load(path)).chunk(2)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    @staticmethod
    def vae_feat_loader_npz(path):
        # [mean, std] - Original logic: same as vae_feat_loader
        # HED features are saved as mean and std concatenated
        hed_feat = torch.from_numpy(np.load(path)['arr_0'])
        # Handle batch dimension if present
        if hed_feat.dim() == 4 and hed_feat.shape[0] == 1:
            hed_feat = hed_feat.squeeze(0)
        # Split into mean and std (chunk along channel dimension) - Original logic
        mean, std = hed_feat.chunk(2, dim=0)
        sample = randn_tensor(mean.shape, generator=None, device=mean.device, dtype=mean.dtype)
        return mean + std * sample

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            meta_data = json.load(f)

        return meta_data

    def sample_subset(self, ratio):
        sampled_idx = random.sample(list(range(len(self))), int(len(self) * ratio))
        self.img_samples = [self.img_samples[i] for i in sampled_idx]

    def __len__(self):
        return len(self.img_samples)

    def __getattr__(self, name):
        if name == "set_epoch":
            return lambda epoch: None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


DATASETS.register_module(name='InternalDataHed', module=InternalDataControl)
