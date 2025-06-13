import torch
in_path = "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz4-lr1e-4-qwen2.5-7b-instruct-freeze_llm-s2t-deepspeed_zero0-total_steps150000/s2s_epoch_5_step_14392/global_step129000/mp_rank_00_model_states.pt"
out_path = "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz4-lr1e-4-qwen2.5-7b-instruct-freeze_llm-s2t-deepspeed_zero0-total_steps150000/s2s_epoch_5_step_14392/global_step129000"
weight_dict = torch.load(in_path)["module"]
torch.save(weight_dict, f"{out_path}/model.pt")
print("[Finish]")