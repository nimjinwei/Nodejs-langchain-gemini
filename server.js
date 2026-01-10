import express from "express";
import cors from "cors";
import multer from "multer";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { BufferMemory } from "langchain/memory";
import * as dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

// Dynamic import for pdf-parse (lazy loading to avoid startup errors)
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

// Configure CORS for remote environments
app.use(cors({
  origin: '*', // Allow all origins (dev environment)
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
      
      // Dynamically load PDF parser
      const parser = await loadPdfParser();
      if (!parser) {
        throw new Error('PDF Parser library not installed');
      }
      
      // Parse PDF
      const data = await parser(pdfBuffer);
      const text = data.text;
      
      if (!text || text.trim().length === 0) {
        throw new Error("No extractable text found in PDF. It might be a scanned image PDF.");
      }

      console.log(`📖 PDF has ${data.numpages} pages`);
      console.log(`📝 Extracted text length: ${text.length} chars`);

      // Store PDF info
      this.currentPdfInfo = {
        filename: filename,
        pages: data.numpages,
        textLength: text.length,
        uploadedAt: new Date().toISOString(),
      };

      // Chunk and index PDF text
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
      console.error('❌ PDF processing error:', error);
      throw new Error(`PDF processing failed: ${error.message}`);
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

    // PROMPT UPDATED TO ENGLISH
    const prompt = `You are a professional document assistant. Answer the user's question based on the following context extracted from the PDF document.
If the information is not in the context, please state that clearly.

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
    // PROMPT UPDATED TO ENGLISH
    const summarizePrompt = `Please generate a concise summary (3-5 sentences) for the following text in English:

Text:
${text}

Summary:`;

    const response = await this.llm.invoke(summarizePrompt);
    return response.content;
  }

  async summarizePDF() {
    if (!this.currentPdfInfo) {
      throw new Error("Please upload a PDF file first");
    }

    // Get document fragments
    const allDocs = await this.vectorStore.similaritySearch("", 10);
    const fullText = allDocs.map(doc => doc.pageContent).join("\n\n");

    // PROMPT UPDATED TO ENGLISH
    const summarizePrompt = `Please generate a comprehensive summary (5-8 sentences) for the following PDF content in English, including:
1. Main Document Topic
2. Key Points
3. Important Conclusions or Findings

PDF Content:
${fullText.substring(0, 8000)}

Summary:`;

    const response = await this.llm.invoke(summarizePrompt);
    return {
      summary: response.content,
      pdfInfo: this.currentPdfInfo,
    };
  }

  async chatWithMemory(userInput) {
    const startTime = Date.now();
    
    try {
      console.log('  🤖 Calling Gemini API...');
      
      // PROMPT UPDATED TO ENGLISH
      const prompt = `You are a friendly AI assistant. Please answer the user's question concisely and in English.

User Question: ${userInput}

Your Answer:`;

      const response = await this.llm.invoke(prompt);
      
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.log(`  ✅ Gemini API Response Success (Duration: ${duration}s)`);
      
      return response.content;
    } catch (error) {
      const duration = ((Date.now() - startTime) / 1000).toFixed(2);
      console.error(`  ❌ Gemini API Call Failed (Duration: ${duration}s)`);
      console.error('  Error Details:', error.message);
      
      // Friendly message for quota errors
      if (error.message.includes('429') || error.message.includes('quota')) {
        throw new Error('API quota limit exceeded. Free tier: 15 req/min, 1500/day. Please wait 1 minute and try again.');
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

// Get Current PDF Info
app.get("/api/pdf-info", (req, res) => {
  const info = ragSystem.getCurrentPdfInfo();
  if (!info) {
    return res.status(404).json({ error: "No PDF file loaded" });
  }
  res.json(info);
});

// PDF Summary
app.post("/api/pdf-summary", async (req, res) => {
  try {
    const result = await ragSystem.summarizePDF();
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Load Documents (Manual)
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
      return res.status(400).json({ error: "Please provide query content" });
    }
    const result = await ragSystem.queryWithRAG(query);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Text Summary
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

// Chat (with memory)
app.post("/api/chat", async (req, res) => {
  const startTime = Date.now();
  
  try {
    const { message } = req.body;
    if (!message) {
      return res.status(400).json({ error: "Please provide message content" });
    }
    
    console.log(`\n💬 [${new Date().toISOString()}] Received chat message: "${message}"`);
    console.log(`⏱️  Processing...`);
    
    // Set longer timeout
    req.setTimeout(60000); // 60s
    res.setTimeout(60000);
    
    const response = await ragSystem.chatWithMemory(message);
    
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`✅ Chat response success (Duration: ${duration}s)`);
    console.log(`📝 Response content: ${response.substring(0, 100)}...`);
    
    res.json({ response });
  } catch (error) {
    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.error(`❌ Chat error (Duration: ${duration}s):`, error.message);
    console.error('Full Error:', error);
    
    // User-friendly error messages
    let errorMessage = error.message;
    
    if (error.message.includes('timeout') || error.message.includes('ETIMEDOUT')) {
      errorMessage = 'Request timed out, Gemini API is responding too slowly, please try again later';
    } else if (error.message.includes('API key')) {
      errorMessage = 'API Key invalid or not configured';
    } else if (error.message.includes('quota') || error.message.includes('429')) {
      errorMessage = 'API quota exceeded (15 req/min), please wait 1 minute and try again';
    } else if (error.message.includes('rate limit')) {
      errorMessage = 'Too many requests, please try again later';
    } else if (error.message.includes('ECONNREFUSED')) {
      errorMessage = 'Cannot connect to Gemini API';
    }
    
    res.status(500).json({ error: errorMessage });
  }
});

// Get Memory
app.get("/api/memory", async (req, res) => {
  try {
    const memory = await ragSystem.getMemory();
    res.json(memory);
  } catch (error) {
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