from transformers import Qwen2_5_VLForConditionalGeneration
import torch
import torch.nn as nn
import argparse

class LayerNorm(nn.Module):
    """T5-style LayerNorm over the channel dimension (No bias and no subtraction of mean)."""
    def __init__(self, n_channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_channels, 1, 1))

    def forward(self, x: torch.Tensor):
        # x is a feature map of shape: batch_size x n_channels x h x w
        var = x.square().mean(dim=1, keepdim=True)
        out = x * (var + 1e-8).rsqrt()
        out = out * self.scale
        return out

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Load and modify Qwen2.5-VL model with heatmap and prompt encoder')
parser.add_argument('--stage1_model', type=str, required=True, help='Path to the model trained in Stage 1')
parser.add_argument('--model_init', type=str, required=True, help='Path to the model_init.pt file')
parser.add_argument('--output', type=str, required=True, help='Output directory to save the modified model')
args = parser.parse_args()

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.stage1_model,
    trust_remote_code=True
)


with open(args.model_init, "rb") as f:
    state_dict = torch.load(f, map_location='cpu')

for key in state_dict:
    if isinstance(state_dict[key], torch.Tensor):
        state_dict[key] = state_dict[key].to(torch.bfloat16)

heatmap_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("heatmap") and isinstance(value, torch.Tensor):
        heatmap_state_dict[key] = value.to(torch.bfloat16)
missing, unexpected = model.heatmap.load_state_dict(
    {k.replace("heatmap.", ""): v for k, v in heatmap_state_dict.items()},
    strict=False
)


prompt_encoder_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("prompt_encoder") and isinstance(value, torch.Tensor):
        prompt_encoder_state_dict[key] = value.to(torch.bfloat16)
missing, unexpected = model.prompt_encoder.load_state_dict(
    {k.replace("prompt_encoder.", ""): v for k, v in prompt_encoder_state_dict.items()},
    strict=False
)

model.save_pretrained(args.output)

