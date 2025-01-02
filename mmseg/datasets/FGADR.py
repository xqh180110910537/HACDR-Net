from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class FGADRDataset(CustomDataset):
    # CLASSES = ["background","EX","HE","SE",]
    # PALETTE = [[0,0,0],[255,0,0],[255,255,0],[0,255,0]]
    CLASSES = ["background", "EX","MA", "SE", "HE"]
    PALETTE = [[0, 0, 0], [255, 0, 0],[255, 255, 0],[255, 255, 255], [0, 255, 0]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split,reduce_zero_label=False, **kwargs)

        assert osp.exists(self.img_dir) and self.split is not None
