# eval_patch_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io

class EvalPatchDataset(Dataset):
    """
    评测用滑窗 Dataset（不做整图重建，只切 patch）
    依赖外部提供的全局变量/函数：DATASET, DATA_FOLDER, DSM_FOLDER, ERODED_FOLDER, convert_from_color
    """
    def __init__(self, ids, window_size, stride, dataset_name,
                 data_folder_tmpl, dsm_folder_tmpl, eroded_folder_tmpl,
                 convert_from_color_fn):
        self.ids = list(ids)
        self.win = window_size  # (H, W)
        self.stride = stride
        self.dataset_name = dataset_name
        self.DATA_FOLDER = data_folder_tmpl
        self.DSM_FOLDER = dsm_folder_tmpl
        self.ERODED_FOLDER = eroded_folder_tmpl
        self.convert_from_color = convert_from_color_fn

        self.meta = []  # (id, x, y, h, w)
        # 仅探尺寸生成索引
        for _id in self.ids:
            if self.dataset_name == 'Potsdam':
                _img = io.imread(self.DATA_FOLDER.format(_id))[:, :, :3]
            else:
                _img = io.imread(self.DATA_FOLDER.format(_id))
            H, W = _img.shape[:2]
            h, w = self.win
            for x in range(0, H - h + 1, self.stride):
                for y in range(0, W - w + 1, self.stride):
                    self.meta.append((_id, x, y, h, w))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        _id, x, y, h, w = self.meta[i]

        if self.dataset_name == 'Potsdam':
            rgb = io.imread(self.DATA_FOLDER.format(_id))[:, :, :3].astype('float32') / 255.0
        else:
            rgb = io.imread(self.DATA_FOLDER.format(_id)).astype('float32') / 255.0

        dsm = io.imread(self.DSM_FOLDER.format(_id)).astype('float32')
        mn, mx = float(np.min(dsm)), float(np.max(dsm))
        if mx > mn:
            dsm = (dsm - mn) / (mx - mn)

        gt_e = self.convert_from_color(io.imread(self.ERODED_FOLDER.format(_id))).astype('int64')

        rgb_patch = np.copy(rgb[x:x+h, y:y+w]).transpose(2, 0, 1)   # [3,h,w]
        dsm_patch = np.copy(dsm[x:x+h, y:y+w])[None, ...]           # [1,h,w]
        gt_patch  = np.copy(gt_e[x:x+h, y:y+w])                     # [h,w]
        img4 = np.concatenate([rgb_patch, dsm_patch], axis=0).astype('float32')  # [4,h,w]

        return torch.from_numpy(img4), torch.from_numpy(gt_patch)
