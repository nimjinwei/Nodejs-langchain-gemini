// test-api-key.js
// 用于测试 Gemini API 密钥是否有效

import * as dotenv from "dotenv";
dotenv.config();

async function testGeminiAPIKey() {
  const apiKey = process.env.GOOGLE_API_KEY;

  console.log("\n🔍 开始测试 Gemini API 密钥...\n");

  // 检查1: 密钥是否配置
  if (!apiKey || apiKey === "your_gemini_api_key_here") {
    console.error("❌ 错误: API 密钥未配置!");
    console.log("\n📝 请在 .env 文件中设置 GOOGLE_API_KEY");
    console.log("获取密钥: https://aistudio.google.com/app/apikey\n");
    process.exit(1);
  }

  console.log("✅ 步骤 1: API 密钥已配置");
  console.log(`   密钥格式: ${apiKey.substring(0, 10)}...${apiKey.substring(apiKey.length - 4)}`);

  // 检查2: 密钥格式
  if (!apiKey.startsWith("AIzaSy") || apiKey.length < 30) {
    console.error("\n❌ 错误: API 密钥格式不正确!");
    console.log("Gemini API 密钥应该以 'AIzaSy' 开头，长度约为 39 字符\n");
    process.exit(1);
  }

  console.log("✅ 步骤 2: 密钥格式正确");

  // 检查3: 列出可用模型
  console.log("\n🌐 步骤 3: 获取可用模型列表...");

  try {
    const modelsResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1/models?key=${apiKey}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!modelsResponse.ok) {
      const errorData = await modelsResponse.json();
      console.error("\n❌ 获取模型列表失败!");
      console.error("状态码:", modelsResponse.status);
      console.error("错误信息:", JSON.stringify(errorData, null, 2));
      process.exit(1);
    }

    const modelsData = await modelsResponse.json();
    console.log("✅ 成功获取模型列表");
    
    // 显示可用的生成模型
    const generativeModels = modelsData.models.filter(m => 
      m.supportedGenerationMethods?.includes('generateContent')
    );
    
    console.log("\n📋 可用的生成模型:");
    generativeModels.forEach(model => {
      console.log(`   • ${model.name.split('/')[1]}`);
    });

    // 选择一个可用模型进行测试
    const testModel = generativeModels[0]?.name || 'models/gemini-1.5-flash';
    console.log(`\n🧪 使用模型进行测试: ${testModel}`);

    // 检查4: 测试 API 调用
    console.log("\n🌐 步骤 4: 测试 API 调用...");

    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1/${testModel}:generateContent?key=${apiKey}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [
                {
                  text: "Say hello in one word",
                },
              ],
            },
          ],
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      console.error("\n❌ API 调用失败!");
      console.error("状态码:", response.status);
      console.error("错误信息:", JSON.stringify(data, null, 2));

      // 常见错误处理
      if (response.status === 400) {
        console.log("\n💡 可能的原因:");
        console.log("  - API 密钥格式错误");
        console.log("  - 请求参数不正确");
      } else if (response.status === 403) {
        console.log("\n💡 可能的原因:");
        console.log("  - API 密钥无效或已过期");
        console.log("  - API 未启用，请访问 https://console.cloud.google.com/apis/");
        console.log("  - 需要在 Google Cloud Console 启用 'Generative Language API'");
      } else if (response.status === 429) {
        console.log("\n💡 可能的原因:");
        console.log("  - 超出免费配额限制");
        console.log("  - 请求过于频繁");
        console.log("  - 免费层级限制: 每分钟 15 次请求，每天 1500 次请求");
      }

      process.exit(1);
    }

    console.log("✅ API 调用成功!");
    
    // 显示响应内容
    if (data.candidates && data.candidates[0]?.content?.parts) {
      const responseText = data.candidates[0].content.parts[0].text;
      console.log("\n📝 AI 响应:", responseText);
    }

    // 检查5: 配额信息
    console.log("\n📈 步骤 5: 配额状态");
    console.log("\n📋 免费层级限制:");
    console.log("  • 每分钟请求数 (RPM): 15");
    console.log("  • 每天请求数 (RPD): 1,500");
    console.log("  • 每分钟 Tokens: 32,000");

    console.log("\n✅ 所有测试通过!");
    console.log("\n🎉 你的 Gemini API 密钥可以正常使用!");
    console.log("\n💡 建议使用的模型:");
    generativeModels.slice(0, 3).forEach(model => {
      console.log(`  • ${model.name.split('/')[1]}`);
    });
    
    console.log("\n📚 更多信息:");
    console.log("  - API 文档: https://ai.google.dev/docs");
    console.log("  - 配额监控: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas");

  } catch (error) {
    console.error("\n❌ 测试失败!");
    console.error("错误:", error.message);
    
    if (error.message.includes("fetch")) {
      console.log("\n💡 可能的原因:");
      console.log("  - 网络连接问题");
      console.log("  - 防火墙或代理阻止了请求");
      console.log("  - DNS 解析问题");
    }
    
    process.exit(1);
  }
}

// 运行测试
testGeminiAPIKey().catch(console.error);