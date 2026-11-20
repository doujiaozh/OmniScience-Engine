import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
import time
from modules.bio_processing.fetcher import fetch_proteins
from modules.bio_processing.parser import parse_fasta
from modules.phys_sim.accelerator import PhysSimAccelerator
from modules.light_ai.inference import LightAIModel
from modules.bio_processing.processor import BioProcessor

# 初始化模块
bio = BioProcessor()
phys = PhysSimAccelerator()
ai = LightAIModel()


def run_pipeline(limit):
    logs = []
    t0 = time.time()

    logs.append(f"启动 OSE 流水线 (CNCB数据源, 限制: {limit}条)...")
    try:
        fasta_text = fetch_proteins(limit=int(limit))
        logs.append("数据下载完成。")

        seqs = parse_fasta(fasta_text)
        logs.append(f"成功解析 {len(seqs)} 条序列。")

        target_seq = seqs[0]
        logs.append(f"提取目标序列 (前50字符): {target_seq[:50]}...")

        # 生信处理
        bio_feats = bio.process(target_seq)

        # 物理模拟 (简化展示)
        states = phys.simulate(bio_feats, steps=10)
        logs.append("物理动力学模拟完成 (阻尼简谐振子)。")

        # AI 训练与推理
        score = ai.train_and_predict(states)
        logs.append(f"AI 模型梯度下降训练完成！预测置信度: {score:.4f}")

        elapsed = round(time.time() - t0, 2)
        logs.append(f"全流程耗时: {elapsed} 秒")

        return "\n".join(logs), target_seq
    except Exception as e:
        logs.append(f"运行出错: {str(e)}")
        return "\n".join(logs), ""


# 构建炫酷的网页界面
with gr.Blocks(title="OmniScience-Engine", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# 🧬 OmniScience-Engine (OSE)")
    gr.Markdown(
        "**Win7 极客手搓的轻量级 AI4Science 框架**\n融合生信数据、物理模拟与轻量化AI。"
    )

    with gr.Row():
        with gr.Column():
            limit_input = gr.Slider(
                minimum=1, maximum=10, value=3, step=1, label="拉取蛋白质序列数量"
            )
            run_btn = gr.Button("🚀 启动端到端流水线", variant="primary")
        with gr.Column():
            output_log = gr.Textbox(label="执行日志", lines=10, interactive=False)
            output_seq = gr.Textbox(
                label="真实解析的序列 (Example)", lines=3, interactive=False
            )

    run_btn.click(fn=run_pipeline, inputs=limit_input, outputs=[output_log, output_seq])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
