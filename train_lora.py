import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"  # 限制 CPU 线程数，防止 Win7 卡死

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from modules.bio_processing.fetcher import fetch_proteins
from modules.bio_processing.parser import parse_fasta

# 1. 加载基础模型
model_id = "Qwen/Qwen1.5-0.5B-Chat"
print("[OSE-Train] 正在加载模型 (首次运行可能需要网络)...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# 2. 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)
peft_model = get_peft_model(model, lora_config)

# 3. 获取真实的 50 条蛋白质数据！
print("[OSE-Train] 正在获取真实生信数据...")
fasta_text = fetch_proteins(limit=50)  # 拉取 50 条
seqs = parse_fasta(fasta_text)
if not seqs:
    raise ValueError("未获取到序列，请检查网络！")

print(f"[OSE-Train] 成功获取 {len(seqs)} 条真实序列，开始构建训练语料...")

# 4. 构建真实数据集
train_data = []
for seq in seqs[:50]:  # 取前50条
    # 构造对话格式
    text = f"<|im_start|>system\n你是一个专业的生信AI助手。<|im_end|>\n<|im_start|>user\n请分析序列 {seq[:20]} 的物理特性。<|im_end|>\n<|im_start|>assistant\n该序列属于高度保守的蛋白质结构，具备良好的水溶性。<|im_end|>"
    inputs = tokenizer(
        text, return_tensors="pt", max_length=128, truncation=True, padding="max_length"
    )
    train_data.append((inputs["input_ids"], inputs["attention_mask"]))

optimizer = torch.optim.AdamW(peft_model.parameters(), lr=1e-4)

# 5. 地狱级 CPU 训练开始
epochs = 10  # 准备跑 10 个 Epoch，Win7 可能会很慢，做好心理准备
print(f"[OSE-Train] 开始在 Win7 CPU 上进行微调，共 {epochs} 个 Epoch...")

for epoch in range(epochs):
    total_loss = 0
    for input_ids, attention_mask in train_data:
        optimizer.zero_grad()
        outputs = peft_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_data)
    print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

print("✅ 真实数据微调完成！")
peft_model.save_pretrained("ose_lora_weights_real")
print("✅ 权重已保存至 'ose_lora_weights_real' 文件夹")
