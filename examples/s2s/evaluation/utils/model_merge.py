import torch
from collections import OrderedDict

# === Configuration ===
pt_file_paths = [
    "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t_1.00-t2t_0.50-total_steps_180000-lora_r_32/s2s_epoch_4_step_3351/model.pt",
    "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t_1.00-t2t_0.75-total_steps_204000-lora_r_32/s2s_epoch_3_step_65940/model.pt",
    "/valleblob/v-wenxichen/exp/s2s-interleave/gpu4-btz3-lr1e-4-qwen2.5-7b-instruct-s2t_1.00-t2t_1.00-total_steps_230000-lora_r_32/s2s_epoch_3_step_73644/model.pt"
]
output_path = "/valleblob/v-wenxichen/exp/s2s-interleave/merge_models/A5_A6_A7/model.pt"
# ====================

def average_models(model_paths):
    assert len(model_paths) > 0, "Model paths list cannot be empty."

    # Load the first model to initialize the averaging
    avg_state_dict = torch.load(model_paths[0], map_location="cpu")
    if isinstance(avg_state_dict, dict) and "state_dict" in avg_state_dict:
        avg_state_dict = avg_state_dict["state_dict"]  # Adapt to Lightning and other structures

    # Initialize a new OrderedDict to hold the summed parameters
    summed_params = OrderedDict()
    for key in avg_state_dict.keys():
        summed_params[key] = avg_state_dict[key].clone()

    # Add parameters from subsequent models
    for path in model_paths[1:]:
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        for key in summed_params:
            summed_params[key] += state_dict[key]

    # Average the summed parameters
    for key in summed_params:
        summed_params[key] /= len(model_paths)

    return summed_params

def save_averaged_model(state_dict, output_path):
    torch.save(state_dict, output_path)
    print(f"✅ Averaged model saved to {output_path}")

if __name__ == "__main__":
    averaged_state = average_models(pt_file_paths)
    save_averaged_model(averaged_state, output_path)
