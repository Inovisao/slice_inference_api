from typing import Tuple
from cv2.typing import MatLike
import cv2
import os
from dataclasses import dataclass

from config.settings import DatasetConfig, SlicingConfig


class Sahi:
    def __init__(self, DatasetConfig: DatasetConfig, SlicingConfig: SlicingConfig):
        self.data_config = DatasetConfig
        self.slicing_config = SlicingConfig
        
    def define_stride(self):
        stride = []
        
        o_p = self.slicing_config.overlap_percentage
        overlap_y, overlap_x = int(float(self.slicing_config.tile_size[0]) * o_p), int(float(self.slicing_config.tile_size[1]) * o_p)
        
        stride[0] = self.slicing_config.tile_size[0] - overlap_y # y
        stride[1] = self.slicing_config.tile_size[1] - overlap_x # x
        return stride, overlap_x, overlap_y
    
    def img_count(dataset_path: str) -> int:
        number_images = 0
        if os.path.exists(dataset):
            dataset = os.listdir(dataset_path)
            for _ in dataset:
                if _.endswith(("png", "jpg", "jpeg")):
                    n_images += 1
        return number_images
    
    def mount_matrix(self, dataset_path: str):
        overlap_px = (self.define_stride()[1], self.define_stride[2])
        dataset_json ={
            "dataset_metadata" : {
                "tile_size": self.slicing_config.tile_size,
                "overlap_px": overlap_px,
                "slicing_mode": "sahi",
                "tiles": {
                    "row_index": -1,
                    "column_index": -1,
                }
            }
        }
    
    def slice_image(self, image: MatLike, output_path: str):
        config = self.config
        
    
    def apply_slicing(self, dataset_path: str, output_path: str):
        number_images = self.img_count(dataset_path)
        if os.path.exists(dataset):
            dataset = os.listdir(dataset_path)
            for img_name, i in dataset:
                if img_name.endswith(("png", "jpg", "jpeg")):
                    img_path = os.path.join(dataset_path, img_name)
                    img = cv2.imread(img_path)
                    self.slice_image(img, output_path)
                    print(f"Image {img_path} processed!  {i} / {number_images}")
        print(f"{number_images} imagens salvas em {output_path}")
                
                
                
            
            
    
    