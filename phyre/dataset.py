from torch.utils.data import Dataset
from pathlib import Path
import phyre.fsvisit as fsvisit
import hickle as hkl
import torch
import torch.nn.functional as F
import phyre.vis
import einops

class PhyreVideoDataset(Dataset):
    def _add_hkl_to_list(self, fpath: Path, _):
        if fpath.suffix == ".hkl":
            self.video_paths.append(fpath)

    def __init__(self, base_path):
        self.video_paths = []
        fsvisit.FSVisitor(
            file_callback=self._add_hkl_to_list
        ).go(Path(base_path))
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_phyre = hkl.load(self.video_paths[idx]).astype(int)
        video_phyre = einops.rearrange(video_phyre, "t w h -> w h t")
        video_rgb = phyre.vis.observations_to_float_rgb(video_phyre)
        video_rgb = einops.rearrange(video_rgb, "w h t c -> t c w h")
        video_rgb_tensor = torch.Tensor(video_rgb)
        video_rgb_tensor = F.max_pool2d(video_rgb_tensor, kernel_size=(2,2), stride=(2,2))
        return video_rgb_tensor
