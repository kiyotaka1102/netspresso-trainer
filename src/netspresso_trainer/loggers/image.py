# Copyright (C) 2024 Nota Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import PIL.Image as Image


class ImageSaver:
    def __init__(self, model, result_dir, save_best_only: Optional[bool]=None) -> None:
        super(ImageSaver, self).__init__()
        self.model = model
        self.save_dir: Path = Path(result_dir) / "result_image"
        self.save_dir.mkdir(exist_ok=True)
        self.save_best_only = save_best_only

    def save_ndarray_as_image(self, image_array: np.ndarray, filename: Union[str, Path], dataformats: Literal['HWC', 'CHW'] = 'HWC'):
        assert image_array.ndim == 3
        if dataformats != 'HWC' and dataformats == 'CHW':
            image_array = image_array.transpose((1, 2, 0))

        # HWC
        assert image_array.shape[-1] in [1, 3]
        Image.fromarray(image_array.astype(np.uint8)).save(filename)
        return True
    # def save_ndarray_as_image(self, image_array: np.ndarray, filename: Union[str, Path], dataformats: Literal['HWC', 'CHW'] = 'HWC'):
    #     assert image_array.ndim == 3, f"Expected 3D array, got {image_array.ndim}D"

    #     # 1. Convert CHW to HWC
    #     if dataformats == 'CHW':
    #         image_array = image_array.transpose((1, 2, 0))

    #     img = np.array(image_array)
    #     h, w, c = img.shape
    #     assert c in [1, 3], f"Expected 1 or 3 channels, got {c}"

    #     # DEBUG: In ra giá trị để kiểm tra
    #     print(f"[SAVE DEBUG] {filename}")
    #     print(f"   shape: {img.shape}, dtype: {img.dtype}")
    #     print(f"   min: {img.min():.4f}, max: {img.max():.4f}")

    #     # 2. Denormalize
    #     if img.dtype in [np.float32, np.float64]:
    #         # Case 1: [0, 1]
    #         if img.max() <= 1.0 and img.min() >= 0.0:
    #             img = img * 255.0
    #         # Case 2: [-1, 1]
    #         elif img.min() >= -1.0 and img.max() <= 1.0:
    #             img = (img + 1.0) * 127.5
    #         # Case 3: ImageNet normalized (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    #         elif img.min() < -0.5 or img.max() > 1.5:  # Heuristic
    #             mean = np.array([0.485, 0.456, 0.406], dtype=img.dtype)
    #             std = np.array([0.229, 0.224, 0.225], dtype=img.dtype)
    #             if c == 3:
    #                 mean = mean.reshape(1, 1, 3)
    #                 std = std.reshape(1, 1, 3)
    #             img = (img * std) + mean  # denormalize
    #             img = img * 255.0
    #         else:
    #             img = img * 255.0  # fallback

    #         img = np.clip(img, 0, 255)

    #     img = img.astype(np.uint8)

    #     # 3. Grayscale → RGB
    #     if c == 1:
    #         img = np.repeat(img, 3, axis=-1)

    #     # 4. Save
    #     Image.fromarray(img).save(filename)
    #     print(f"   → Saved: {filename}")
    #     return True
    def save_result(self, image_dict: Dict, prefix, epoch):
        prefix_dir: Path = self.save_dir / prefix
        prefix_dir.mkdir(exist_ok=True)

        for k, v_list in image_dict.items():
            for idx, v in enumerate(v_list):
                assert isinstance(v, np.ndarray)
                if epoch is None:
                    self.save_ndarray_as_image(v, f"{prefix_dir}/{idx:03d}_{k}.png", dataformats='HWC')
                elif self.save_best_only:
                    self.save_ndarray_as_image(v, f"{prefix_dir}/best_{idx:03d}_{k}.png", dataformats='HWC')
                else:
                    self.save_ndarray_as_image(v, f"{prefix_dir}/{epoch:04d}_{idx:03d}_{k}.png", dataformats='HWC')

    def __call__(
        self,
        prefix: Literal['training', 'validation', 'evaluation', 'inference'],
        epoch: Optional[int] = None,
        images: Optional[List] = None,
        **kwargs
    ):
        if images is not None:
            self.save_result(images, prefix=prefix, epoch=epoch)
