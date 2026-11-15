import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 0. 防御配置与编码锁死
# ==========================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 强制把标准输出和标准错误设为 UTF-8，遇到无法解码的非法字符直接忽略，绝不让它崩溃
sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

# 拦截 Python 级别的 print，全部导向安全区
import builtins
_original_print = builtins.print
def _safe_print(*args, **kwargs):
    kwargs['file'] = sys.stderr
    _original_print(*args, **kwargs)
builtins.print = _safe_print

import contextlib
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from modules.bio_processing.fetcher import fetch_proteins
from modules.bio_processing.parser import parse_fasta
from modules.phys_sim.accelerator import PhysSimAccelerator
from modules.light_ai.inference import LightAIModel
from modules.bio_processing.processor import BioProcessor

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. 初始化引擎与专属大脑
# ==========================================
server = Server("omni-science-engine")
bio = BioProcessor()
phys = PhysSimAccelerator()
ai = LightAIModel()

peft_model = None
tokenizer = None

print("正在加载微调后的专属 AI 大脑 (Win7 CPU 极速模式)...")
try:
    base_model_id = "Qwen/Qwen1.5-0.5B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, local_files_only=False)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    
    if os.path.exists("./ose_lora_weights"):
        peft_model = PeftModel.from_pretrained(base_model, "./ose_lora_weights")
        peft_model.eval()
        print("✅ 专属大脑 (LoRA) 挂载成功！")
    else:
        print("⚠️ 未找到 ose_lora_weights 文件夹。")
        peft_model = base_model
except Exception as e:
    print(f"❌ 模型加载失败: {e}")

# ==========================================
# 2. 定义 MCP 工具接口
# ==========================================
@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="predict_protein",
            description="拉取真实基因序列，运行物理模拟，并用微调的 AI 大脑预测置信度",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "拉取的序列数量"}
                },
                "required": ["limit"],
            },
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    if name == "predict_protein" and arguments:
        try:
            limit = arguments.get("limit", 3)
            
            # 强制屏蔽这期间所有的 C 语言底层输出，防止破坏 MCP 管道
            with contextlib.redirect_stdout(open(os.devnull, 'w')):
                fasta_text = fetch_proteins(limit=limit)
                seqs = parse_fasta(fasta_text)
                
                if not seqs:
                    return [TextContent(type="text", text="未获取到序列")]
                    
                target_seq = seqs[0][:20]
                    
                bio_feats = bio.process(seqs[0])
                states = phys.simulate(bio_feats, steps=10)
                score = ai.train_and_predict(states)
                
                llm_analysis = ""
                if peft_model and tokenizer:
                    prompt = f"<|im_start|>system\n你是一个专业的生信AI助手。<|im_end|>\n<|im_start|>user\n请分析序列 {target_seq} 的预测结果。置信度为 {score:.4f}。<|im_end|>\n<|im_start|>assistant\n"
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = peft_model.generate(**inputs, max_new_tokens=40, temperature=0.7, do_sample=True)
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    clean_reply = generated_text.split('assistant\n')[-1].strip()
                    llm_analysis = f"\n\n[AI Brain 分析]: {clean_reply}"

            # 正常返回结果
            return [TextContent(
                type="text", 
                text=f"OSE predict success!\nSequence: {target_seq}...\nConfidence: {score:.4f}{llm_analysis}"
            )]
            
        except Exception as e:
            return [TextContent(type="text", text=f"运行失败: {str(e)}")]

    raise ValueError(f"未知工具: {name}")

# ==========================================
# 3. 启动服务
# ==========================================
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    asyncio.run(main())