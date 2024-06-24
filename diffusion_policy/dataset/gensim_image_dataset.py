from typing import Dict
import torch
import numpy as np
import copy
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class GensimImageDataset(BaseImageDataset):
    def __init__(self,
        data_path="", 
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        high_level: bool=True
    ):
        
        super().__init__()
        from gen_diversity.dataset import GensimDataset, GENSIM_ROOT
        data_path = os.path.join(GENSIM_ROOT, "data", "train")

        if high_level:
            seq_length = 2
        else:
            seq_length = 20
        try:
            self._dataset = GensimDataset.load(data_path, seq_length, high_level=high_level)
        except FileNotFoundError:
            self._dataset = GensimDataset.make(data_path, seq_length, high_level=high_level)
        
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        return self

    def get_normalizer(self, mode='limits', **kwargs):
        samples = torch.randint(len(self._dataset), (1000,))
        samples = self._dataset[samples.tolist()]
        data = {
            'action': samples['action'],
            'state': samples['state']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self._dataset)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self._dataset[idx]
        image = data["image"] / 255
        data = {
            "obs": {
                "image": image[:1],
                "state": data["state"][:1]
            },
            "action": data["action"]
        }
        return data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
