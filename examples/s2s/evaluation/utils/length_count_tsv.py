# 文件路径请直接在这里修改
input_file_path = "/home/wenxi/mydisk/exp/standard_qa_eval/web_qa/gpu4-btz1-lr1e-4-interleave_text12_audio36-Qwen2.5-7b-Instruct-lora-audio_embed_only-freeze_llm-s2t-whisper_large-v3-s2t_0.75-t2t_0.25/pred_text"

def compute_average_word_count(file_path):
    total_words = 0
    total_lines = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '\t' not in line:
                continue  # 忽略格式不对的行
            parts = line.strip().split('\t', 1)
            if len(parts) < 2:
                continue
            text = parts[1]
            word_count = len(text.strip().split())
            total_words += word_count
            total_lines += 1

    if total_lines == 0:
        print("⚠️ 文件中没有有效行。")
        return

    avg_words = total_words / total_lines
    print(f"📊 平均单词数: {avg_words:.2f} （基于 {total_lines} 条样本）")

# 执行主函数
compute_average_word_count(input_file_path)
