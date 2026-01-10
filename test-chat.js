// test-chat.js
// 测试聊天功能是否正常

import * as dotenv from "dotenv";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

dotenv.config();

async function testChat() {
  console.log("\n🧪 测试 Gemini 聊天功能...\n");

  try {
    // 初始化模型
    const llm = new ChatGoogleGenerativeAI({
      modelName: "gemini-2.5-flash",
      temperature: 0.7,
      apiKey: process.env.GOOGLE_API_KEY,
      maxRetries: 2,
      timeout: 30000,
    });

    console.log("✅ 模型初始化成功");
    console.log("📤 发送测试消息: '你好吗?'");

    const startTime = Date.now();
    
    // 测试简单对话
    const response = await llm.invoke("你好吗?");
    
    const endTime = Date.now();
    const duration = ((endTime - startTime) / 1000).toFixed(2);

    console.log(`\n✅ 响应成功! (耗时: ${duration}秒)`);
    console.log(`\n📝 AI 回复:\n${response.content}\n`);

    // 测试第二条消息
    console.log("📤 发送第二条测试消息: '介绍一下自己'");
    
    const startTime2 = Date.now();
    const response2 = await llm.invoke("用一句话介绍一下自己");
    const endTime2 = Date.now();
    const duration2 = ((endTime2 - startTime2) / 1000).toFixed(2);

    console.log(`\n✅ 响应成功! (耗时: ${duration2}秒)`);
    console.log(`\n📝 AI 回复:\n${response2.content}\n`);

    console.log("🎉 所有测试通过!");
    console.log("\n💡 提示:");
    console.log("  - 如果响应时间过长（>10秒），可能是网络问题");
    console.log("  - 如果遇到配额错误，请等待配额重置");
    console.log("  - 免费层级限制: 每分钟 15 次请求\n");

  } catch (error) {
    console.error("\n❌ 测试失败!");
    console.error("错误类型:", error.name);
    console.error("错误信息:", error.message);

    if (error.message.includes("timeout")) {
      console.log("\n💡 超时问题:");
      console.log("  - 检查网络连接");
      console.log("  - 可能是 Gemini API 响应慢");
      console.log("  - 尝试增加 timeout 设置");
    } else if (error.message.includes("API key")) {
      console.log("\n💡 API 密钥问题:");
      console.log("  - 检查 .env 文件中的 GOOGLE_API_KEY");
      console.log("  - 确保密钥有效且未过期");
    } else if (error.message.includes("429") || error.message.includes("quota")) {
      console.log("\n💡 配额问题:");
      console.log("  - 超出免费层级限制");
      console.log("  - 每分钟最多 15 次请求");
      console.log("  - 等待 1 分钟后重试");
    } else if (error.message.includes("404")) {
      console.log("\n💡 模型不存在:");
      console.log("  - 模型名称可能不正确");
      console.log("  - 尝试使用: gemini-1.5-flash 或 gemini-1.5-pro");
    }

    console.log("\n📋 完整错误信息:");
    console.error(error);
    
    process.exit(1);
  }
}

testChat();