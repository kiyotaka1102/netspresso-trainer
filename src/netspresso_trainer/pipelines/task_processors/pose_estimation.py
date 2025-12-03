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

from typing import Literal, Optional

import numpy as np
import torch

from netspresso_trainer.models.utils import set_training_targets

from ...utils.protocols import ProcessorStepOut
from .base import BaseTaskProcessor


class PoseEstimationProcessor(BaseTaskProcessor):
    def __init__(self, conf, postprocessor, devices, **kwargs):
        super(PoseEstimationProcessor, self).__init__(conf, postprocessor, devices, **kwargs)

    def train_step(self, train_model, batch, optimizer, loss_factory, metric_factory):
        train_model.train()
        images, keypoints = batch['pixel_values'], batch['keypoints']

        if isinstance(images, torch.Tensor) and images.ndim == 4:  # [batch_size, C, H, W]
            images = images
        else:
            images = torch.stack(images, dim=0)
        
        if isinstance(keypoints, torch.Tensor):
            # Nếu đã là tensor (không phải list)
            if keypoints.dim() == 4 and keypoints.shape[0] == 1:
                keypoints = keypoints.squeeze(0)
        elif isinstance(keypoints, (list, tuple)):
            # Mỗi phần tử có shape [1, 4, 3]
            keypoints = [kp.squeeze(0) if kp.dim() == 3 and kp.shape[0] == 1 else kp for kp in keypoints]
            keypoints = torch.stack(keypoints, dim=0)  # [8, 4, 3]
        else:
            raise TypeError(f"Unexpected keypoints type: {type(keypoints)}")

        keypoints = keypoints.squeeze(1)
        # print(keypoints)
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            train_model = set_training_targets(train_model, target)
            out = train_model(images)
            loss_factory.calc(out, target, phase='train')

        loss_factory.backward(self.grad_scaler)
        if self.max_norm:
            # Unscales the gradients of optimizer's assigned parameters in-place
            self.grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), self.max_norm)
        # optimizer's gradients are already unscaled, so scaler.step doesn't unscale them,
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()

        pred = self.postprocessor(out)

        keypoints = keypoints.detach().cpu().numpy()
        if self.conf.distributed:
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                pred = np.concatenate(gathered_pred, axis=0)
                keypoints = np.concatenate(gathered_labels, axis=0)

        step_out = ProcessorStepOut.empty()
        if self.single_gpu_or_rank_zero:
            step_out['pred'] = list(pred)
            step_out['target'] = list(keypoints)

        return step_out

    def valid_step(self, eval_model, batch, loss_factory, metric_factory):
        eval_model.eval()
        name = batch['name']
        indices, images, keypoints = batch['indices'], batch['pixel_values'], batch['keypoints']

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        elif indices.ndim == 0:
            indices = indices.unsqueeze(0)
            
        if not isinstance(images, torch.Tensor) or images.ndim == 3:
            images = torch.stack(images, dim=0) if isinstance(images, (list, tuple)) else images.unsqueeze(0)
            
        if not isinstance(keypoints, torch.Tensor) or keypoints.ndim == 3:
            keypoints = torch.stack(keypoints, dim=0) if isinstance(keypoints, (list, tuple)) else keypoints.unsqueeze(0)
        keypoints = keypoints.squeeze(1)
        # print(keypoints.shape)
        images = images.to(self.devices)
        target = {'keypoints': keypoints.to(self.devices)}

        out = eval_model(images)
        loss_factory.calc(out, target, phase='valid')

        pred = self.postprocessor(out)
        # print(pred)
        indices = indices.numpy()
        keypoints = keypoints.detach().cpu().numpy()
        if self.conf.distributed:
            name = np.array(name)[indices != -1].tolist()
            pred = pred[indices != -1]
            keypoints = keypoints[indices != -1]

            gathered_name = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]
            gathered_labels = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(name, gathered_name if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(keypoints, gathered_labels if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                name = sum(gathered_name, [])
                pred = np.concatenate(gathered_pred, axis=0)
                keypoints = np.concatenate(gathered_labels, axis=0)

        step_out = ProcessorStepOut.empty()
        if self.single_gpu_or_rank_zero:
            step_out['name'] = name
            step_out['pred'] = list(pred)
            step_out['target'] = list(keypoints)

        return step_out

    def test_step(self, test_model, batch):
        test_model.eval()
        name = batch['name']
        indices, images = batch['indices'], batch['pixel_values']
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        elif indices.ndim == 0:  # scalar tensor
            indices = indices.unsqueeze(0)
            
        if not isinstance(images, torch.Tensor) or images.ndim == 3:
            images = torch.stack(images, dim=0) if isinstance(images, (list, tuple)) else images.unsqueeze(0)
        images = images.to(self.devices)

        out = test_model(images)

        pred = self.postprocessor(out)

        indices = indices.numpy()
        if self.conf.distributed:
            name = np.array(name)[indices != -1].tolist()
            pred = pred[indices != -1]

            gathered_name = [None for _ in range(torch.distributed.get_world_size())]
            gathered_pred = [None for _ in range(torch.distributed.get_world_size())]

            torch.distributed.gather_object(name, gathered_name if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.gather_object(pred, gathered_pred if torch.distributed.get_rank() == 0 else None, dst=0)
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                gathered_pred = sum(gathered_pred, [])
                name = sum(gathered_name, [])
                pred = gathered_pred

        step_out = ProcessorStepOut.empty()
        if self.single_gpu_or_rank_zero:
            step_out['name'] = name
            step_out['pred'] = list(pred)

        return step_out

    def get_metric_with_all_outputs(self, outputs, phase: Literal['train', 'valid'], metric_factory):
        if self.single_gpu_or_rank_zero:
            # pred = np.concatenate([output['pred']for output in outputs], axis=0)
            pred = np.stack(outputs['pred'], axis=0)
            # keypoints = np.concatenate([output['target']for output in outputs], axis=0)
            keypoints = np.stack(outputs['target'], axis=0)
            metric_factory.update(pred, keypoints, phase=phase)

    def get_predictions(self, results, class_map):
        pass
