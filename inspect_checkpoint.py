import torch
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Faisasa12/Project-Retinexis",
    filename="multimodal_models/hypertension_combined/efficientnet/best.pt"
)

ckpt = torch.load(path, map_location="cpu", weights_only=False)

print("=" * 60)
print("CHECKPOINT KEYS:")
print("=" * 60)
for k in ckpt.keys():
    print(f"  {k}")

print()
print("=" * 60)
print("METADATA ENCODER:")
print("=" * 60)
if "metadata_encoder" in ckpt:
    enc = ckpt["metadata_encoder"]
    print(f"  metadata_columns: {enc.get('metadata_columns')}")
    print(f"  continuous_cols:  {enc.get('continuous_cols')}")
    print(f"  categorical_cols: {enc.get('categorical_cols')}")
    print(f"  means:            {enc.get('means')}")
    print(f"  stds:             {enc.get('stds')}")
    print(f"  category_maps:    {enc.get('category_maps')}")
    print(f"  output_dim:       {enc.get('output_dim')}")
else:
    print("  NOT FOUND")

print()
print("=" * 60)
print("METADATA HEAD STATE DICT:")
print("=" * 60)
if "metadata_head_state_dict" in ckpt:
    for k, v in ckpt["metadata_head_state_dict"].items():
        print(f"  {k}: {tuple(v.shape)}")
else:
    print("  NOT FOUND")
