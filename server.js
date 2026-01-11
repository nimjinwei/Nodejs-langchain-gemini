import express from "express";
import cors from "cors";
import multer from "multer";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ConversationChain } from "langchain/chains";
import { BufferMemory } from "langchain/memory";
import { PromptTemplate } from "@langchain/core/prompts";
import * as dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

// Dynamic import for pdf-parse (lazy load)
let pdfParse = null;
async function loadPdfParser() {
  if (!pdfParse) {
    try {
      const module = await import('pdf-parse/lib/pdf-parse.js');
      pdfParse = module.default;
      console.log('✅ PDF Parser loaded successfully');
    } catch (error) {
      console.warn('⚠️  PDF Parser load failed');
      console.warn('   Try: npm install pdf-parse');
    }
  }
  return pdfParse;
}

const app = express();

// CORS configuration for remote environments
app.use(cors({
  origin: '*', // Allow all origins (dev)
  methods: ['GET', 'POST', 'DELETE'],
  credentials: true
}));

app.use(express.json());
app.use(express.static("public"));

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server Error:', err);
  res.status(500).json({ 
    error: err.message || 'Internal Server Error',
    details: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});

// File upload configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = "uploads";
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir);
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    if (file.mimetype === "application/pdf") {
      cb(null, true);
    } else {
      cb(new Error("Only PDF files are supported"));
    }
  },
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
});

// RAG System Class
class RAGSystem {
  constructor() {
    this.llm = new ChatGoogleGenerativeAI({
      modelName: "gemini-2.5-flash",
      temperature: 0.7,
      apiKey: process.env.GOOGLE_API_KEY,
    });

    this.embeddings = new GoogleGenerativeAIEmbeddings({
      modelName: "embedding-001",
      apiKey: process.env.GOOGLE_API_KEY,
    });

    this.memory = new BufferMemory({
      returnMessages: true,
      memoryKey: "history",
    });

    this.vectorStore = null;
    this.isInitialized = false;
    this.currentPdfInfo = null;
  }

  async ingestDocuments(documents, metadata = {}) {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await textSplitter.createDocuments(documents);
    
    // Add metadata
    splitDocs.forEach(doc => {
      doc.metadata = { ...doc.metadata, ...metadata };
    });

    this.vectorStore = await HNSWLib.fromDocuments(splitDocs, this.embeddings);
    this.isInitialized = true;
  }

  async ingestPDF(pdfBuffer, filename) {
    try {
      console.log('📄 Starting PDF parsing...');
      
      // Load parser dynamically
      const parser = await loadPdfParser();
      if (!parser) {
        throw new Error('PDF Parser library not installed');
      }
      
      // Parse PDF
      const data = await parser(pdfBuffer);
      const text = data.text;
      
      if (!text || text.trim().length === 0) {
        throw new Error("No extractable text found in PDF (might be scanned/image-based)");
      }

      console.log(`📖 PDF has ${data.numpages} pages`);
      console.log(`📝 Extracted text length: ${text.length} chars`);

      // Store PDF Info
      this.currentPdfInfo = {
        filename: filename,
        pages: data.numpages,
        textLength: text.length,
        uploadedAt: new Date().toISOString(),
      };

      // Split and Index
      await this.ingestDocuments([text], {
        source: filename,
        type: "pdf",
      });

      console.log('✅ PDF processing complete');

      return {
        success: true,
        info: this.currentPdfInfo,
        preview: text.substring(0, 500) + "...",
      };
    } catch (error) {
      console.error('❌ PDF Processing Error:', error);
      throw new Error(`PDF Processing failed: ${error.message}`);
    }
  }

  async queryWithRAG(query) {
    if (!this.vectorStore) {
      throw new Error("Please upload a PDF or load documents first!");
    }

    const relevantDocs = await this.vectorStore.similaritySearch(query, 3);
    const context = relevantDocs
      .map((doc, i) => `[Fragment ${i + 1}]\n${doc.pageContent}`)
      .join("\n\n");

    const prompt = `You are a professional document assistant. Answer the user's question based on the following context extracted from a PDF document.
If the context does not contain relevant information, please explicitly state so.

Document Context:
${context}

User Question: ${query}

Please provide an accurate and detailed answer in English:`;

    const response = await this.llm.invoke(prompt);
    return {
      answer: response.content,
      sources: relevantDocs.length,
      pdfInfo: this.currentPdfInfo,
    };
  }

  async summarizeText(text) {
    const summarizePrompt = `Please generate a concise summary (3-5 sentences) for the following text:

Text:
${text}

Summary (in English):`;

    const response = await this.llm.invoke(summarizePrompt);
    return response.content;
  }

  async summarizePDF() {
    if (!this.currentPdfInfo) {
      throw new Error("Please upload a PDF file first");
    }

    // Get all document fragments
    const allDocs = await this.vectorStore.similaritySearch("", 10);
    const fullText = allDocs.map(doc => doc.pageContent).join("\n\n");

    const summarizePrompt = `Please generate a comprehensive summary (5-8 sentences) for the following PDF document content, including:
1. Main topic
2. Key points
3. Important conclusions or findings

PDF Content:
${fullText.substring(0, 8000)}

Summary (in English):`;

    const response = await this.llm.invoke(summarizePrompt);
    return {
      summary: response.content,
      pdfInfo: this.currentPdfInfo,
    };
  }

  async chatWithMemory(userInput) {
    const startTime = Date.now();
    
    try {
      console.log('  🤖 Generating chat response (with memory)...');
      
      // Get history - with timeout protection
      let history = [];
      try {
        const memoryVars = await Promise.race([
          this.memory.loadMemoryVariables({}),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Memory load timeout')), 5000)
          )
        ]);
        history = memoryVars.history || [];
      } catch (memError) {
        console.warn('  ⚠️  Failed to load memory, using empty memory:', memError.message);
        history = [];
      }
      
      console.log(`  💭 Current History: ${history.length} messages`);
      
      // Build prompt with history - Simplified
      let historyText = '';
      if (history.length > 0) {
        // Keep last 3 turns (6 messages) to save tokens
        const recentHistory = history.slice(-6);
        historyText = recentHistory.map(msg => {
          try {
            const role = msg._getType() === 'human' ? 'User' : 'Assistant';
            const content = msg.content?.substring(0, 200) || ''; // Limit length
            return `${role}: ${content}`;
          } catch (e) {
            return '';
          }
        }).filter(Boolean).join('\n');
      }
      
      const prompt = historyText 
        ? `You are a friendly AI assistant. Answer concisely (2-3 sentences).

Recent Conversation:
${historyText}

User: ${userInput}
Assistant:`
        : `You are a friendly AI assistant. Answer concisely (2-3 sentences).

User: ${userInput}
Assistant:`;

      console.log('  🌐 Calling Gemini API...');
      const response = await this.llm.invoke(prompt);
      
      // Save memory asynchronously
      this.memory.saveContext(
        { input: userInput },
        { output: response.content }
      ).catch(err => console.warn('  ⚠️  Failed to save memory:', err.message));
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`  ✅ Chat response success (Time: ${duration}s)`);
      
      return response.content;
    } catch (error) {
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.error(`  ❌ Chat Generation Failed (Time: ${duration}s)`);
      console.error('  Error Details:', error.message);
      
      // Quota issues
      if (error.message.includes('429') || error.message.includes('quota')) {
        throw new Error('API Quota Exceeded. Free tier: 15 req/min. Please wait 1 minute.');
      }
      
      // Timeout issues
      if (error.message.includes('timeout') || error.message.includes('ETIMEDOUT')) {
        throw new Error('API response timeout. Check network or retry later.');
      }
      
      throw error;
    }
  }

  async getMemory() {
    const memoryVariables = await this.memory.loadMemoryVariables({});
    return memoryVariables;
  }

  async clearMemory() {
    await this.memory.clear();
  }

  getCurrentPdfInfo() {
    return this.currentPdfInfo;
  }
}

// Initialize RAG System
const ragSystem = new RAGSystem();

// Check API Key
if (!process.env.GOOGLE_API_KEY || process.env.GOOGLE_API_KEY === 'your_gemini_api_key_here') {
  console.error('⚠️  WARNING: GOOGLE_API_KEY not configured!');
  console.error('Please set your Gemini API key in the .env file');
  console.error('Get Key: https://makersuite.google.com/app/apikey');
}

// Default docs
const defaultDocuments = [
  "LangChain is a framework for developing applications powered by language models.",
  "Gemini is a multimodal large language model developed by Google.",
  "RAG (Retrieval Augmented Generation) enhances LLM responses by retrieving relevant documents.",
];

// API Routes

// Health Check
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    initialized: ragSystem.isInitialized,
    pdfLoaded: ragSystem.currentPdfInfo !== null,
  });
});

// Upload PDF
app.post("/api/upload-pdf", upload.single("pdf"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Please upload a PDF file" });
    }

    const pdfBuffer = fs.readFileSync(req.file.path);
    const result = await ragSystem.ingestPDF(pdfBuffer, req.file.originalname);

    // Delete temp file
    fs.unlinkSync(req.file.path);

    res.json(result);
  } catch (error) {
    if (req.file) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: error.message });
  }
});

// Get PDF Info
app.get("/api/pdf-info", (req, res) => {
  const info = ragSystem.getCurrentPdfInfo();
  if (!info) {
    return res.status(404).json({ error: "No PDF loaded" });
  }
  res.json(info);
});

// PDF Summary
app.post("/api/pdf-summary", async (req, res) => {
  const startTime = Date.now();
  
  try {
    console.log(`\n📋 [${new Date().toISOString()}] Received PDF Summary Request`);
    console.log(`⏱️  Processing...`);
    
    // Set longer timeout for summary
    req.setTimeout(90000); // 90s
    res.setTimeout(90000);
    
    const result = await ragSystem.summarizePDF();
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ PDF Summary Generated (Time: ${duration}s)`);
    
    res.json(result);
  } catch (error) {
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`❌ PDF Summary Error (Time: ${duration}s):`, error.message);
    
    let errorMessage = error.message;
    
    if (error.message.includes('timeout') || error.message.includes('ETIMEDOUT')) {
      errorMessage = 'PDF Summary timed out (Document might be too large). Try specific questions instead.';
    } else if (error.message.includes('quota') || error.message.includes('429')) {
      errorMessage = 'API Quota Exceeded. Please wait 1 minute.';
    }
    
    res.status(500).json({ error: errorMessage });
  }
});

// Load Documents
app.post("/api/documents", async (req, res) => {
  try {
    const { documents } = req.body;
    if (!documents || !Array.isArray(documents)) {
      return res.status(400).json({ error: "Please provide a document array" });
    }
    await ragSystem.ingestDocuments(documents);
    res.json({ success: true, message: "Documents loaded successfully", count: documents.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// RAG Query
app.post("/api/rag/query", async (req, res) => {
  try {
    const { query } = req.body;
    if (!query) {
      return res.status(400).json({ error: "Please provide a query" });
    }
    const result = await ragSystem.queryWithRAG(query);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Text Summarize
app.post("/api/summarize", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) {
      return res.status(400).json({ error: "Please provide text to summarize" });
    }
    const summary = await ragSystem.summarizeText(text);
    res.json({ summary });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Chat (With Memory)
app.post("/api/chat", async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Please provide message content" });
    }
    
    console.log(`\n💬 [${new Date().toISOString()}] Received Chat Message: "${message}"`);
    console.log(`⏱️  Processing...`);
    
    // Set timeout
    req.setTimeout(60000); // 60s
    res.setTimeout(60000);
    
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Server processing timeout')), 55000);
    });
    
    const chatPromise = ragSystem.chatWithMemory(message);
    
    const response = await Promise.race([chatPromise, timeoutPromise]);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ Chat Response Success (Time: ${duration}s)`);
    console.log(`📝 Response Content: ${response.substring(0, 100)}...`);
    
    res.json({ response });
  } catch (error) {
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`❌ Chat Error (Time: ${duration}s):`, error.message);
    console.error('Full Stack:', error.stack);
    
    let errorMessage = error.message;
    let statusCode = 500;
    
    if (error.message.includes('timeout') || error.message.includes('ETIMEDOUT') || error.message.includes('超时')) {
      errorMessage = `Request timed out (${duration}s). Causes:\n1. Gemini API slow\n2. Network issues\n3. Quota full\n\nSuggestion: Wait 1-2 mins and retry`;
      statusCode = 504;
    } else if (error.message.includes('API key')) {
      errorMessage = 'API Key invalid or missing';
      statusCode = 401;
    } else if (error.message.includes('quota') || error.message.includes('429')) {
      errorMessage = 'API Quota Exceeded (15 req/min). Please wait 1 minute.';
      statusCode = 429;
    } else if (error.message.includes('rate limit')) {
      errorMessage = 'Too many requests, please retry later';
      statusCode = 429;
    } else if (error.message.includes('ECONNREFUSED')) {
      errorMessage = 'Cannot connect to Gemini API, check network';
      statusCode = 503;
    }
    
    res.status(statusCode).json({ error: errorMessage });
  }
});

// Get Memory
app.get("/api/memory", async (req, res) => {
  try {
    console.log('\n👁️  View Memory Request');
    
    const memory = await ragSystem.getMemory();
    console.log('Memory Data:', JSON.stringify(memory, null, 2));
    
    const formattedMemory = {
      history: (memory.history || []).map((msg, index) => {
        try {
          return {
            index: index + 1,
            type: msg._getType ? msg._getType() : (msg.type || 'unknown'),
            content: msg.content || msg.text || '',
            timestamp: new Date().toISOString()
          };
        } catch (e) {
          console.error('Format message failed:', e);
          return {
            index: index + 1,
            type: 'error',
            content: 'Cannot parse this message',
            raw: msg
          };
        }
      }),
      total: memory.history?.length || 0
    };
    
    console.log(`✅ Returning ${formattedMemory.total} memory items`);
    res.json(formattedMemory);
  } catch (error) {
    console.error('❌ Get Memory Error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Clear Memory
app.delete("/api/memory", async (req, res) => {
  try {
    await ragSystem.clearMemory();
    res.json({ success: true, message: "Memory cleared" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => {
  const isCodespace = process.env.CODESPACE_NAME;
  const isGitpod = process.env.GITPOD_WORKSPACE_URL;
  
  console.log(`
╔════════════════════════════════════════════════════════╗
║  🚀 LangChain RAG Server Started                        ║
║                                                        ║
║  📍 Local: http://localhost:${PORT}                     ║
║  📍 Network: http://0.0.0.0:${PORT}                     ║`);

  if (isCodespace) {
    console.log(`║  📍 Codespace: https://${process.env.CODESPACE_NAME}-${PORT}.app.github.dev`);
  }
  
  if (isGitpod) {
    console.log(`║  📍 Gitpod: ${isGitpod.replace('https://', `https://${PORT}-`)}`);
  }
  
  console.log(`║                                                        ║
║  ✅ Server Status: Running                              ║
║  ${process.env.GOOGLE_API_KEY && process.env.GOOGLE_API_KEY !== 'your_gemini_api_key_here' ? '✅ API Key: Configured' : '❌ API Key: Not Configured'}                             ║
╚════════════════════════════════════════════════════════╝
  `);
  
  if (!process.env.GOOGLE_API_KEY || process.env.GOOGLE_API_KEY === 'your_gemini_api_key_here') {
    console.log('\n⚠️  Please configure GOOGLE_API_KEY before use\n');
  }
});