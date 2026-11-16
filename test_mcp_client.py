import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    print("正在启动 OSE MCP Server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ 连接成功！")

            tools = await session.list_tools()
            print("获取到工具列表:", [t.name for t in tools.tools])

            print("\n🤖 正在模拟大模型调用 'predict_protein' (限制 3 条)...")
            result = await session.call_tool("predict_protein", arguments={"limit": 3})
            
            for content in result.content:
                print("\n🎯 最终返回给大模型的结果:\n", content.text)

if __name__ == "__main__":
    asyncio.run(main())