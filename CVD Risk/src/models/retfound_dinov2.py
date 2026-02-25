import torch
# neural network module for classification head
import torch.nn as nn

# timm is used to create the DINOv2 ViT architecture (same as official RETFound repo)
import timm

# huggingface_hub to download the .pth weights from the gated repo
from huggingface_hub import hf_hub_download

# import project abstract class
from .base import BaseClassifier, PreprocessingConfig, IMAGENET_NORM


# create the class
class RETFoundDINOv2(BaseClassifier):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()

        # Download the .pth weights from HuggingFace (auto-cached after first download)
        weights_path = hf_hub_download(
            repo_id="YukunZhou/RETFound_dinov2_meh",
            filename="RETFound_dinov2_meh.pth",
        )

        # Create DINOv2 ViT-Large/14 architecture and load retinal weights
        # num_classes=0 removes the default head (we add our own)
        # this matches the official RETFound repo (models_vit.py lines 79-86)
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=224,
            checkpoint_path=weights_path,
        )

        # feature dim from the model (should be 1024 for ViT-Large)
        self._feature_dim = self.model.num_features

        # classification head (linear layer)
        self._head = nn.Linear(self._feature_dim, 1)

    # required properties for the abstract class

    @property
    def backbone(self) -> nn.Module:
        """Return the feature extractor (everything except the classification head)."""
        return self.model

    @property
    def head(self) -> nn.Module:
        """Return the classification layer."""
        return self._head

    @property
    def feature_dim(self) -> int:
        """Return the feature dimension of the backbone."""
        return self._feature_dim

    @property
    def preprocessing_config(self) -> PreprocessingConfig:
        """Declare input size and normalisation for ViT-Large/14."""
        return PreprocessingConfig(
            input_size=224,
            mean=IMAGENET_NORM["mean"],
            std=IMAGENET_NORM["std"],
            crop_strategy="center",
        )

    # required methods

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector from input images. """
        
        # timm's forward_features returns the backbone output
        features = self.model.forward_features(x)

        # if it returns (B, num_tokens, dim), extract the CLS token
        if features.dim() == 3:
            features = features[:, 0]

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: backbone features -> classification head.   """

        features = self.get_features(x)
        output = self._head(features)
        self._validate_output_shape(output, x.size(0))
        return output