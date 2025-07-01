import torch
in_path = "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t-qwen2.5-7b-instruct_enrich_new-total_steps120000-lora_r_32-ds_zero2/s2s_epoch_2_step_12797/global_step51000/mp_rank_00_model_states.pt"
out_path = "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t-qwen2.5-7b-instruct_enrich_new-total_steps120000-lora_r_32-ds_zero2/s2s_epoch_2_step_12797/global_step51000"
weight_dict = torch.load(in_path)["module"]
torch.save(weight_dict, f"{out_path}/model.pt")
print("[Finish]")