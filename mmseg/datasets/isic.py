from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class ISICDataset(CustomDataset):
    CLASSES = ["background","bing",]
    PALETTE = [[0,0,0],[255,255,255]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png',
                         split=split,reduce_zero_label=False, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None