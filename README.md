# Nodejs-langchain-gemini

# 📄 LangChain RAG System with PDF & Gemini

A full-stack web application that enables users to **chat with PDF documents**, generate summaries, and interact with an AI assistant using **Google Gemini Pro** and **LangChain**.

> **Note:** This project has been configured to strictly communicate in **English** (both the User Interface and AI responses).

---

## ✨ Features

- **📤 PDF Upload & Indexing**: Upload PDF files to build a local vector knowledge base.
- **🔍 RAG (Retrieval-Augmented Generation)**: Ask specific questions based *only* on the content of your uploaded PDF.
- **📝 Summarization**:
  - **PDF Summary**: Generate a comprehensive summary of the entire uploaded document.
  - **Text Summary**: Paste long text to get a concise 3-5 sentence summary.
- **💬 AI Chat with Memory**: A general-purpose chatbot that remembers conversation history.
- **⚡ Real-time Feedback**: Visual loading states and connection status indicators.
- **🌐 Environment Aware**: Automatically adjusts API endpoints for Localhost, CodeSpaces, or Replit.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript.
- **Backend**: Node.js, Express.js.
- **AI & LLM**: 
  - [LangChain.js](https://js.langchain.com/) (Framework)
  - [Google Gemini](https://ai.google.dev/) (LLM: `gemini-2.5-flash`, Embeddings: `embedding-001`)
  - [HNSWLib](https://github.com/nmateria/hnswlib) (Vector Store)
- **File Handling**: Multer, PDF-Parse.

---

# How to Use the file 
Step 1:
Create a file .env

port = 3000

GOOGLE_API_KEY= YOUR_API_KEY


Step 2:
npm install

Step 3:
npm start
