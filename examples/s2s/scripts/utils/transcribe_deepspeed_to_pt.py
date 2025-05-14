import torch
in_path = "/valleblob/v-wenxichen/exp/s2s-interleave/debug/s2s_epoch_1_step_10/global_step10/mp_rank_00_model_states.pt"
out_path = "/valleblob/v-wenxichen/exp/s2s-interleave/debug/s2s_epoch_1_step_10/global_step10"
weight_dict = torch.load(in_path)["module"]
torch.save(weight_dict, f"{out_path}/model.pt")
print("[Finish]")