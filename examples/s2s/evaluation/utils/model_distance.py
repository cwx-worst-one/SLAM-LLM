import torch

ckpt_a = '/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t_1.00-t2t_1.00-total_steps_230000-lora_r_32/s2s_epoch_3_step_73644/model.pt'  
ckpt_b = '/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t_1.00-t2t_0.75-total_steps_204000-lora_r_32/s2s_epoch_3_step_65940/model.pt'

model_a = torch.load(ckpt_a)
model_b = torch.load(ckpt_b)

# Calculate the distance between the two models
distance = 0.0
for key in model_a:
    if 'weight' in key or 'bias' in key:
        distance += torch.norm(model_a[key] - model_b[key]).item()

print(f"Parameter distance: {distance:.4f}")