"""This file contains a class to evalute the reconstruction results.

Original code Copyright (2024) Bytedance Ltd. and/or its affiliates
Modified code Copyright (2024) Turing Inc. and/or its affiliates

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

import warnings

from typing import Optional, Mapping, Text
import numpy as np
from scipy import linalg
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import torchvision
from transformers import CLIPProcessor, CLIPModel

from .inception import get_inception_model


def get_covariance(sigma: torch.Tensor, total: torch.Tensor, num_examples: int) -> torch.Tensor:
    """Computes covariance of the input tensor.

    Args:
        sigma: A torch.Tensor, sum of outer products of input features.
        total: A torch.Tensor, sum of all input features.
        num_examples: An integer, number of examples in the input tensor.
    Returns:
        A torch.Tensor, covariance of the input tensor.
    """
    if num_examples == 0:
        return torch.zeros_like(sigma)

    sub_matrix = torch.outer(total, total)
    sub_matrix = sub_matrix / num_examples

    return (sigma - sub_matrix) / (num_examples - 1)


class VQGANEvaluator:
    def __init__(
        self,
        device,
        enable_rfid: bool = True,
        enable_inception_score: bool = True,
        enable_codebook_usage_measure: bool = False,
        enable_codebook_entropy_measure: bool = False,
        num_codebook_entries: int = 1024
    ):
        """Initializes VQGAN Evaluator.

        Args:
            device: The device to use for evaluation.
            enable_rfid: A boolean, whether enabling rFID score.
            enable_inception_score: A boolean, whether enabling Inception Score.
            enable_codebook_usage_measure: A boolean, whether enabling codebook usage measure.
            enable_codebook_entropy_measure: A boolean, whether enabling codebook entropy measure.
            num_codebook_entries: An integer, the number of codebook entries.
        """
        self._device = device

        self._enable_rfid = enable_rfid
        self._enable_inception_score = enable_inception_score
        self._enable_codebook_usage_measure = enable_codebook_usage_measure
        self._enable_codebook_entropy_measure = enable_codebook_entropy_measure
        self._num_codebook_entries = num_codebook_entries

        # Variables related to Inception score and rFID.
        self._inception_model = None
        self._is_num_features = 0
        self._rfid_num_features = 0
        if self._enable_inception_score or self._enable_rfid:
            self._rfid_num_features = 2048
            self._is_num_features = 1008
            self._inception_model = get_inception_model().to(self._device)
            self._inception_model.eval()
        self._is_eps = 1e-16
        self._rfid_eps = 1e-6

        self.reset_metrics()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        self._is_prob_total = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._is_total_kl_d = torch.zeros(
            self._is_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_real_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_real_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )
        self._rfid_fake_sigma = torch.zeros(
            (self._rfid_num_features, self._rfid_num_features),
            dtype=torch.float64, device=self._device
        )
        self._rfid_fake_total = torch.zeros(
            self._rfid_num_features, dtype=torch.float64, device=self._device
        )

        self._set_of_codebook_indices = set()
        self._codebook_frequencies = torch.zeros((self._num_codebook_entries), dtype=torch.float64, device=self._device)
        self._position_codebook_frequencies = torch.zeros((256, self._num_codebook_entries), dtype=torch.float64, device=self._device)

    def update(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        codebook_indices: Optional[torch.Tensor] = None
    ):
        """Updates the metrics with the given images.

        Args:
            real_images: A torch.Tensor, the real images.
            fake_images: A torch.Tensor, the fake images.
            codebook_indices: A torch.Tensor, the indices of the codebooks for each image.

        Raises:
            ValueError: If the fake images is not in RGB (3 channel).
            ValueError: If the fake and real images have different shape.
        """

        batch_size = real_images.shape[0]
        self._num_examples += batch_size
        self._num_updates += 1

        if self._enable_inception_score or self._enable_rfid:
            # Quantize to uint8 as a real image.
            fake_inception_images = (fake_images * 255).to(torch.uint8)
            features_fake = self._inception_model(fake_inception_images)
            inception_logits_fake = features_fake["logits_unbiased"]
            inception_probabilities_fake = F.softmax(inception_logits_fake, dim=-1)
        
        if self._enable_inception_score:
            probabiliies_sum = torch.sum(inception_probabilities_fake, 0, dtype=torch.float64)

            log_prob = torch.log(inception_probabilities_fake + self._is_eps)
            if log_prob.dtype != inception_probabilities_fake.dtype:
                log_prob = log_prob.to(inception_probabilities_fake)
            kl_sum = torch.sum(inception_probabilities_fake * log_prob, 0, dtype=torch.float64)

            self._is_prob_total += probabiliies_sum
            self._is_total_kl_d += kl_sum

        if self._enable_rfid:
            real_inception_images = (real_images * 255).to(torch.uint8)
            features_real = self._inception_model(real_inception_images)
            if (features_real['2048'].shape[0] != features_fake['2048'].shape[0] or
                features_real['2048'].shape[1] != features_fake['2048'].shape[1]):
                raise ValueError(f"Number of features should be equal for real and fake.")

            for f_real, f_fake in zip(features_real['2048'], features_fake['2048']):
                self._rfid_real_total += f_real
                self._rfid_fake_total += f_fake

                self._rfid_real_sigma += torch.outer(f_real, f_real)
                self._rfid_fake_sigma += torch.outer(f_fake, f_fake)

        if self._enable_codebook_usage_measure:
            self._set_of_codebook_indices |= set(torch.unique(codebook_indices, sorted=False).tolist())

        if self._enable_codebook_entropy_measure:
            entries, counts = torch.unique(codebook_indices, sorted=False, return_counts=True)
            self._codebook_frequencies.index_add_(0, entries.int(), counts.double())
            # codebook indices: (B 1 L)
            for i in range(codebook_indices.shape[2]):
                # codebook_indices_i: (B 1)
                codebook_indices_i = codebook_indices[:, :, i]
                # entries: (self._num_codebook_entries,), counts: (self._num_codebook_entries,)
                entries, counts = torch.unique(codebook_indices_i, sorted=False, return_counts=True)
                self._position_codebook_frequencies[i].index_add_(0, entries.int(), counts.double())


    def result(self) -> Mapping[Text, torch.Tensor]:
        """Returns the evaluation result."""
        eval_score = {}

        if self._num_examples < 1:
            raise ValueError("No examples to evaluate.")
        
        if self._enable_inception_score:
            mean_probs = self._is_prob_total / self._num_examples
            log_mean_probs = torch.log(mean_probs + self._is_eps)
            if log_mean_probs.dtype != self._is_prob_total.dtype:
                log_mean_probs = log_mean_probs.to(self._is_prob_total)
            excess_entropy = self._is_prob_total * log_mean_probs
            avg_kl_d = torch.sum(self._is_total_kl_d - excess_entropy) / self._num_examples

            inception_score = torch.exp(avg_kl_d).item()
            eval_score["InceptionScore"] = inception_score

        if self._enable_rfid:
            mu_real = self._rfid_real_total / self._num_examples
            mu_fake = self._rfid_fake_total / self._num_examples
            sigma_real = get_covariance(self._rfid_real_sigma, self._rfid_real_total, self._num_examples)
            sigma_fake = get_covariance(self._rfid_fake_sigma, self._rfid_fake_total, self._num_examples)

            mu_real, mu_fake = mu_real.cpu(), mu_fake.cpu()
            sigma_real, sigma_fake = sigma_real.cpu(), sigma_fake.cpu()

            diff = mu_real - mu_fake

            # Product might be almost singular.
            covmean, _ = linalg.sqrtm(sigma_real.mm(sigma_fake).numpy(), disp=False)
            # Numerical error might give slight imaginary component.
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    m = np.max(np.abs(covmean.imag))
                    raise ValueError("Imaginary component {}".format(m))
                covmean = covmean.real

            tr_covmean = np.trace(covmean)

            if not np.isfinite(covmean).all():
                tr_covmean = np.sum(np.sqrt((
                    (np.diag(sigma_real) * self._rfid_eps) * (np.diag(sigma_fake) * self._rfid_eps))
                    / (self._rfid_eps * self._rfid_eps)
                ))

            rfid = float(diff.dot(diff).item() + torch.trace(sigma_real) + torch.trace(sigma_fake) 
                - 2 * tr_covmean
            )
            if torch.isnan(torch.tensor(rfid)) or torch.isinf(torch.tensor(rfid)):
                warnings.warn("The product of covariance of train and test features is out of bounds.")

            eval_score["rFID"] = rfid

        if self._enable_codebook_usage_measure:
            usage = float(len(self._set_of_codebook_indices)) / self._num_codebook_entries
            eval_score["CodebookUsage"] = usage

        if self._enable_codebook_entropy_measure:
            probs = self._codebook_frequencies / self._codebook_frequencies.sum()
            entropy = (-torch.log2(probs + 1e-8) * probs).sum()
            eval_score["CodebookEntropy"] = entropy

            # minimum frequency
            min_freq = torch.min(self._codebook_frequencies)
            eval_score["CodebookMinFreq"] = min_freq

            # maximum frequency
            max_freq = torch.max(self._codebook_frequencies)
            eval_score["CodebookMaxFreq"] = max_freq

            # evaluation of entropy for each position (0...len)
            entropy_per_position = self._position_codebook_frequencies / self._position_codebook_frequencies.sum(dim=1, keepdim=True)
            entropy_per_position = (-torch.log2(entropy_per_position + 1e-8) * entropy_per_position).sum(dim=1)
            # tolist and make more readable
            entropy_per_position = entropy_per_position.tolist()
            entropy_per_position = "\n".join([f"{i:04d}: {v}" for i, v in enumerate(entropy_per_position)])
            eval_score["CodebookEntropyPerPosition"] = entropy_per_position

        return eval_score


class DepthEstimationQualityEvaluator:
    def __init__(self, device):
        """Initializes ProgressiveQualityEvaluator Evaluator."""
        self._device = device
        self.reset_metrics()

        # Load depth estimation model and processor
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(self._device).eval()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        # Metric storage
        self._depth_cosine_similarity = []
        self._depth_l1_loss = []
        self._depth_l2_loss = []

    def depth_estimation(self, images):
        """Performs depth estimation on a batch of images and returns raw depth values."""
        inputs = self.image_processor(images=images, return_tensors="pt").to(self._device)
        outputs = self.depth_model(**inputs)
        return outputs.predicted_depth  # Return raw depth predictions

    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """Updates the metrics with the given batch of images."""
        batch_size, num_channels, height, width = real_images.shape
        assert num_channels == 3, "The input images should be in RGB format."
        assert height == width == 256, "The input images should be 256x256."
        
        self._num_examples += batch_size
        self._num_updates += 1

        # Convert real and fake images to PIL format for depth estimation
        real_pil_images = [Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) for img in real_images]
        fake_pil_images = [Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)) for img in fake_images]

        # Perform depth estimation for the entire batch of real and fake images
        real_depths = self.depth_estimation(real_pil_images)
        fake_depths = self.depth_estimation(fake_pil_images)

        # Compare raw depth values without resizing or normalization
        for real_depth, fake_depth in zip(real_depths, fake_depths):
            # Compute depth-based similarity metrics directly on raw depth predictions
            cossim = F.cosine_similarity(real_depth.view(1, -1), fake_depth.view(1, -1)).item()
            l1_loss = F.l1_loss(real_depth, fake_depth).item()
            l2_loss = F.mse_loss(real_depth, fake_depth).item()

            self._depth_cosine_similarity.append(cossim)
            self._depth_l1_loss.append(l1_loss)
            self._depth_l2_loss.append(l2_loss)

    def result(self) -> dict:
        """Returns the evaluation result."""
        eval_score = {}

        # Calculate averages for depth-based metrics
        eval_score["DepthCosineSimilarity"] = np.mean(self._depth_cosine_similarity)
        eval_score["DepthL1Loss"] = np.mean(self._depth_l1_loss)
        eval_score["DepthL2Loss"] = np.mean(self._depth_l2_loss)

        return eval_score


class ImageClassificationQualityEvaluator:
    def __init__(self, device):
        """Initializes ImageClassificationQualityEvaluator."""
        self._device = device
        self.reset_metrics()

        # Load ConvNeXt model for image classification
        self.convnext = torchvision.models.convnext_base(weights=torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1).to(self._device).eval()
        self.transform = torchvision.models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()
        self.to_pil = torchvision.transforms.ToPILImage()

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        # Metric storage
        self._acc_at_1 = []
        self._acc_at_5 = []

    @torch.no_grad()
    def update(self, labels: torch.Tensor, fake_images: torch.Tensor):
        """Updates the metrics with the given batch of images."""
        batch_size, num_channels, height, width = fake_images.shape
        assert num_channels == 3, "The input images should be in RGB format."
        assert height == width == 256, "The input images should be 256x256."
        
        self._num_examples += batch_size
        self._num_updates += 1

        # Convert to PIL images
        fake_pil_images = [self.to_pil(img) for img in fake_images]

        # Apply transformations and move to device
        fake_images_tensor = torch.stack([self.transform(img) for img in fake_pil_images]).to(self._device)

        # Perform inference
        fake_results = self.convnext(fake_images_tensor)

        # Compute metrics
        for label, fake_result in zip(labels, fake_results):
            fake_top1 = fake_result.argmax().item()
            acc_at_1 = int(label == fake_top1)

            fake_top5 = fake_result.topk(5).indices.tolist()
            acc_at_5 = int(label in fake_top5)

            self._acc_at_1.append(float(acc_at_1))
            self._acc_at_5.append(float(acc_at_5))

    def result(self) -> dict:
        """Returns the evaluation result."""
        eval_score = {
            "acc@1": np.mean(self._acc_at_1),
            "acc@5": np.mean(self._acc_at_5)
        }
        return eval_score


class SemanticQualityEvaluator:
    def __init__(self, device):
        """Initializes SemanticQualityEvaluator."""
        self._device = device
        self.reset_metrics()

        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self._device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def reset_metrics(self):
        """Resets all metrics."""
        self._num_examples = 0
        self._num_updates = 0

        # Metric storage
        self._cosine_similarities = []

    @torch.no_grad()
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor):
        """Updates the metrics with the given batch of images."""
        batch_size, num_channels, height, width = real_images.shape
        assert num_channels == 3, "The input images should be in RGB format."
        assert height == width == 256, "The input images should be 256x256."
        
        self._num_examples += batch_size
        self._num_updates += 1

        # Convert to PIL images
        real_pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in real_images]
        fake_pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for img in fake_images]

        # Prepare inputs for CLIP
        real_inputs = self.clip_processor(images=real_pil_images, return_tensors="pt", padding=True).to(self._device)
        fake_inputs = self.clip_processor(images=fake_pil_images, return_tensors="pt", padding=True).to(self._device)

        # Get embeddings
        real_embeddings = self.clip_model.get_image_features(**real_inputs)
        fake_embeddings = self.clip_model.get_image_features(**fake_inputs)

        # Normalize embeddings
        real_embeddings = real_embeddings / real_embeddings.norm(dim=-1, keepdim=True)
        fake_embeddings = fake_embeddings / fake_embeddings.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(real_embeddings, fake_embeddings, dim=-1)
        self._cosine_similarities.extend(cosine_sim.cpu().tolist())

    def result(self) -> dict:
        """Returns the evaluation result."""
        eval_score = {
            "mean_cosine_similarity": np.mean(self._cosine_similarities),
            "std_cosine_similarity": np.std(self._cosine_similarities),
        }
        return eval_score