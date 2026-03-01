from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from collections import defaultdict
import numpy as np
import torch.nn.functional as F



def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_text, default_num=0.0, clamp01=True):
    semantic_alignment, aesthetic, plausibility, overall = [], [], [], []

    # number pattern: supports integers, decimals, and scientific notation
    num_pat = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'

    fields = [
        (r'semantic\s*alignment', semantic_alignment),
        (r'aesthetic',            aesthetic),
        (r'plausibility',         plausibility),
        (r'overall\s*impression', overall),
    ]

    for text in output_text:
        # prefer the content inside <answer>...</answer>; fall back to full text
        m_block = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL | re.IGNORECASE)
        block = m_block.group(1).strip() if m_block else text

        for label, bucket in fields:
            m_score = re.search(fr'{label}\s*score\s*:\s*{num_pat}', block, flags=re.IGNORECASE)
            if m_score:
                val = float(m_score.group(1))
                if clamp01:
                    val = max(0.0, min(1.0, val))
                bucket.append(val)
            else:
                bucket.append(default_num)
    # B = len(output_text)
    # means = [ (semantic_alignment[i] + aesthetic[i] + plausibility[i] + overall[i]) / 4.0
    #           for i in range(B) ]
    all_scores = {'semantic_alignment': semantic_alignment,
                  'aesthetic':aesthetic,
                  'plausibility': plausibility,
                  'overall': overall}
    def combine_scores(all_scores):
        
        sa = np.asarray(all_scores["semantic_alignment"], dtype=np.float32)
        ae = np.asarray(all_scores["aesthetic"],            dtype=np.float32)
        pl = np.asarray(all_scores["plausibility"],         dtype=np.float32)
        ov = np.asarray(all_scores["overall"],              dtype=np.float32)

        weighted = sa * 0.8 + ae * 0.1 + pl * 0.1
        return weighted
    combined = combine_scores(all_scores)
    return combined, all_scores


from pathlib import Path
class DoctorScorer_hp(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "",  # The model path after reinforcement learning fine-tuning
            torch_dtype=self.dtype,
            device_map=None,
        ).to(self.device)

        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained("", use_fast=True) # The model path after reinforcement learning fine-tuning
        self.task = ("Given a caption and an image generated based on this caption, please analyze the provided image in detail. Evaluate it on various dimensions including Semantic Alignment (How well the image content corresponds to the caption), Aesthetics (composition, color usage, and overall artistic quality), Plausibility (realism and attention to detail), and Overall Impression (General subjective assessment of the image's quality). For each evaluation dimension, provide a score between 0-1. Within the <answer> and </answer> tags, summarize your assessment in the following format: \"Semantic Alignment score: ...\nMisalignment Locations: ...\nAesthetic score: ...\nPlausibility score: ...\nArtifact Locations: ...\nOverall Impression score: ...\". No additional text is allowed in the answer section.\n\n Your actual evaluation should be based on the quality of the provided image.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]")

    @torch.no_grad()
    def __call__(self, prompt, images):

        def clean_prompt(p: str) -> str:
            # strip whitespace, then remove trailing terminal punctuation if present
            # covers ASCII and common full-width punctuation
            return re.sub(r'[\s\.\,\!\?\:\;…、，。！？；：]+$', '', str(p).strip())

        images_512 = [image.resize((512,512), resample=Image.BICUBIC) for image in images]
        
        # images_base64 = [pil_image_to_base64(image) for image in images_512]
        messages=[]
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        prompts_clean = [clean_prompt(p) for p in prompts]
        for i, image in enumerate(images_512):
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.task.format(prompt=prompts_clean[i])},
                    ],
                },
            ])

        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        texts = texts + '<think>\n...</think>\n<answer>'
        image_inputs, video_inputs = process_vision_info(messages)
        outputs_texts = [None] * len(texts)
        heatmaps ={"heatmap": []} 

        for i in range(len(texts)):
            single_inputs = self.processor(
            text=[texts[i]],
            images=[image_inputs[i]] if isinstance(image_inputs, list) else image_inputs,
            videos=[video_inputs[i]] if isinstance(video_inputs, list) else video_inputs,
            padding=False,                 # <<< NO PADDING
            return_tensors="pt",
        ).to(self.device)
            single_inputs["return_dict_in_generate"] = True
            single_inputs["output_hidden_states"] = True

            with torch.no_grad():
                image_embeds = self.model.visual(
                    single_inputs["pixel_values"].to(self.model.device), 
                    grid_thw=single_inputs["image_grid_thw"].to(self.model.device)
                )
                out = self.model.generate(**single_inputs, max_new_tokens=2048,use_cache=True )
                attn = single_inputs["attention_mask"]           # [1, T]
                L = int(attn.sum().item())
                seqs = out.sequences if hasattr(out, "sequences") else out
                trimmed = seqs[0, L:]                            # [T_new]

                outputs_texts[i] = self.processor.batch_decode(
                    [trimmed.detach().cpu().tolist()],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]        

                final_layer_hidden_states = []
                if hasattr(out, "hidden_states") and len(out.hidden_states) > 1:
                    for step_states in out.hidden_states[1:]:
                        # step_states is a tuple of layers; take last layer: [bs, 1, hidden]
                        final_layer_output = step_states[-1]
                        final_layer_hidden_states.append(final_layer_output)
                all_generated_hidden_states = torch.cat(final_layer_hidden_states, dim=1)
                true_generated = seqs[:, single_inputs["input_ids"].shape[1]:]             # [1, T_new]
                cfg = self.model.config
                mis_id = getattr(cfg, "misalignment_token_id", None)
                art_id = getattr(cfg, "artifact_token_id", None)

                # Masks over generated positions (skip first token to mirror your snippet)
                misalignment_mask = true_generated[:, 1:] == mis_id
                artifact_mask = true_generated[:, 1:] == art_id
                last_hidden_state = self.model.text_hidden_fcs[0](all_generated_hidden_states)

                
                # Prepare image embeddings
                image_embedding = self.model.image_hidden_fcs[0](image_embeds.unsqueeze(0))
                image_embedding = image_embedding.transpose(1, 2).view(1, -1, 18, 18)

                def _blank_heatmap(h=512, w=512):
                    # Always return a tensor, on CPU float32 (easy to save/export)
                    return torch.zeros((h, w), dtype=torch.float32)

                def _predict_heatmap_from_mask(token_mask):
                    # token_mask shape: [1, T_new-1]
                    if token_mask is None or not token_mask.any():
                        return _blank_heatmap()
                    token_embeds = last_hidden_state[token_mask].unsqueeze(1)  # [K, 1, D]
                    # keep only one token to prompt the segmenter (like your code)
                    token_embeds = token_embeds[:1, ...] if token_embeds.shape[0] > 1 else token_embeds
                    # prompt encoder expects (points/boxes/masks/text_embeds)
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=None, boxes=None, masks=None, text_embeds=token_embeds
                    )

                    # segmentation head -> low-res mask
                    low_res_masks = self.model.heatmap(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings.to(image_embedding.dtype),
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    fused_pred = self.model.sigmoid(low_res_masks)  # [1, 1, H, W]
                    return fused_pred[0, 0]
                def to_64x64(hm: torch.Tensor) -> torch.Tensor:
                    # hm: [H, W] (e.g., 288×288), stays on same device/dtype
                    x = hm.unsqueeze(0).unsqueeze(0)                 # [1,1,H,W]
                    x = F.adaptive_avg_pool2d(x, (64, 64))           # [1,1,64,64]
                    return x.squeeze(0).squeeze(0)    
                mis_hm = _predict_heatmap_from_mask(misalignment_mask)
                art_hm = _predict_heatmap_from_mask(artifact_mask)
                mis_hm_64 = to_64x64(mis_hm)
                art_hm_64 = to_64x64(art_hm)
                combined_64 = (mis_hm_64 + art_hm_64)/2.0


                heatmaps["heatmap"].append(combined_64)


        rewards,all_scores = extract_scores(outputs_texts)

        return rewards, heatmaps

# Usage example
def main():
    scorer = DoctorScorer_hp(
        device="cuda",
        dtype=torch.bfloat16
    )
    images=[
    "nasa.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]

    print(scorer(None, pil_images))

if __name__ == "__main__":
    main()