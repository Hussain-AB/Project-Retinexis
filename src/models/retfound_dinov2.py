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
    def __init__(self, pretrained=True):  # review: removed **kwargs — factory validates kwargs, swallowing them hides typos
        super().__init__()

        # Create DINOv2 ViT-Large/14 architecture (no weights yet)
        # num_classes=0 removes the default head (we add our own)
        self.model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=518,  # must match checkpoint (518/14=37 patches → 1370 tokens)
        )

        # Load RETFound retinal weights if pretrained
        # The checkpoint stores weights under a "teacher" key (DINOv2
        # self-supervised format), so we extract and strip the prefix.
        if pretrained:
            weights_path = hf_hub_download(
                repo_id="YukunZhou/RETFound_dinov2_meh",
                filename="RETFound_dinov2_meh.pth",
            )
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)

            # Handle nested checkpoint formats
            if "teacher" in ckpt:
                state_dict = ckpt["teacher"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt

            # Strip common key prefixes (e.g. "backbone.", "module.")
            cleaned = {}
            print("State dict: ", state_dict)
            for k, v in state_dict.items():
                for prefix in ("backbone.", "module.", "encoder."):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                cleaned[k] = v

            print("Cleaned state dict: ", cleaned)

            msg = self.model.load_state_dict(cleaned, strict=False)
            print(f"  RETFound-DINOv2 weights loaded: {len(cleaned)} keys, "
                  f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")

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
            input_size=518,  # matches checkpoint positional embedding (518×518)
            mean=IMAGENET_NORM["mean"],
            std=IMAGENET_NORM["std"],
            crop_strategy="pad_to_square",  # review: was "center" which isn't a valid strategy in transforms.py
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
