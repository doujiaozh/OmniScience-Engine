mcp_server.py
import asyncio
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from modules.bio_processing.fetcher import fetch_proteins
from modules.bio_processing.parser import parse_fasta
from modules.phys_sim.accelerator import PhysSimAccelerator
from modules.light_ai.inference import LightAIModel
from modules.bio_processing.processor import BioProcessor

server = Server("omni-science-engine")

# 初始化引擎
bio = BioProcessor()
phys = PhysSimAccelerator()
ai = LightAIModel()

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
                Tool(
                                name="predict_protein",
                                            description="拉取真实基因序列并预测其物理特性 (OSE Engine)",
                                                        inputSchema={
                                                                            "type": "object",
                                                                                            "properties": {
                                                                                                                    "limit": {"type": "integer", "description": "拉取序列数量"}
                                                                                            },
                                                                                                            "required": ["limit"],
                                                        },
                )
    ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        if name == "predict_protein" and arguments:
                limit = arguments.get("limit", 3)
                        # 调用你的核心引擎
                                fasta_text = fetch_proteins(limit=limit)
                                        seqs = parse_fasta(fasta_text)
                                                
                                                        if not seqs:
                                                                    return [TextContent(type="text", text="未获取到序列")]
                                                                                
                                                                                        bio_feats = bio.process(seqs[0])
                                                                                                states = phys.simulate(bio_feats, steps=10)
                                                                                                        score = ai.train_and_predict(states)
                                                                                                                
                                                                                                                        return [TextContent(
                                                                                                                                        type="text", 
                                                                                                                                                    text=f"OSE预测成功！目标序列: {seqs[0][:20]}...，AI预测置信度: {score:.4f}"
                                                                                                                        )]
                                                                                                                            raise ValueError(f"未知工具: {name}")

                                                                                                                            async def main():
                                                                                                                                async with stdio_server() as (read_stream, write_stream):
                                                                                                                                        await server.run(
                                                                                                                                                        read_stream,
                                                                                                                                                                    write_stream,
                                                                                                                                                                                InitializationOptions(
                                                                                                                                                                                                    server_name="omni-science-engine",
                                                                                                                                                                                                                    server_version="0.1.0",
                                                                                                                                                                                                                                    capabilities=server.get_capabilities(
                                                                                                                                                                                                                                                            notification_options=NotificationOptions(),
                                                                                                                                                                                                                                                                                experimental_capabilities={},
                                                                                                                                                                                                                                    ),
                                                                                                                                                                                ),
                                                                                                                                        )

                                                                                                                                        if __name__ == "__main__":
                                                                                                                                            asyncio.run(main())
                                                                                                                                                                                                                                    )
                                                                                                                                                                                )
                                                                                            }
                                                        }
                )
    ]