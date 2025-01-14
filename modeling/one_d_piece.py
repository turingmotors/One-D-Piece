"""This file contains the model definition of TiTok.

Code Copyright (2024) Turing Inc. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

import torch

from modeling.titok import TiTok, PretrainedTokenizer


class OneDPiece(TiTok):
    def apply_tail_token_drop(self, x: torch.Tensor):
        """
        Apply Tail Token Drop to the tensor.
        """
        # x: B x 12 x 1 x num_latent_tokens
        _, _, _, num_tokens = x.shape
        # take index randomly
        end_indices = torch.randint(1, num_tokens, (1,)).to(x.device)
        # apply tail token drop
        x = x[:, :, :, :end_indices]
        return x

    def decode(self, z_quantized, length=None):
        if self.training:
            # MARK: Apply Tail Token Drop in training
            z_quantized = self.apply_tail_token_drop(z_quantized)
        return super().decode(z_quantized, length=length)
