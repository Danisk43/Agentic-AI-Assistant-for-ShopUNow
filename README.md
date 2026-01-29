# 🤖 Agentic AI Assistant for ShopUNow

An **Agentic AI-powered support assistant** designed to intelligently handle both **customer** and **employee** queries for a retail platform.  
Built using **LangGraph**, **Retrieval-Augmented Generation (RAG)**, and **ChromaDB**, this system automates query routing, delivers contextual answers, and escalates complex issues to human agents.

📌 **Capstone Project – Analytics Vidhya**  
👤 **Author:** Daniyal Sheikh

---

## 📖 Problem Statement

Retail organizations face increasing volumes of customer and internal employee queries. Traditional support systems are:

- Slow and inefficient  
- Expensive to scale  
- Overloaded with repetitive queries  

This project addresses the need for a **scalable, intelligent AI assistant** that can:
- Automatically route queries
- Provide accurate, department-specific answers
- Escalate critical or negative-sentiment queries to humans  

---

## 🎯 Project Objectives

- Build an **Agentic AI Assistant** tailored for retail operations  
- Support **dual user bases**:
  - Internal employees
  - External customers  
- Leverage **advanced AI techniques**:
  - Agentic workflows
  - Retrieval-Augmented Generation (RAG)
  - Intelligent query routing  

---

## 🏢 Supported Departments

### Internal Departments
- **HR** – Policies, payroll, leave management  
- **IT Support** – System access, VPN, hardware troubleshooting  

### External Departments
- **Billing** – Payments, refunds, invoices  
- **Shipping** – Order tracking, delivery updates, logistics  

---

## 🧠 Knowledge Base

- Synthetic **FAQ datasets** generated using LLMs  
- **15 Q&A pairs per department**  
- **60 total Q&A entries**  
- Stored with **department-level metadata** for accurate retrieval  

---

## 🏗️ System Architecture

**Workflow Overview:**

1. User submits a query  
2. Sentiment analysis + department classification  
3. LangGraph-based router selects the correct path  
4. RAG retrieves relevant answers from ChromaDB  
5. Negative or unknown queries are escalated to human support  

---

## 🔄 Agentic Workflow

- **Sentiment Agent**  
  Detects positive, neutral, or negative sentiment  

- **Classifier Agent**  
  Identifies the relevant department (HR, IT, Billing, Shipping)  

- **Router Agent**  
  Decides between AI response or human escalation  

- **RAG Agent**  
  Generates grounded responses using the knowledge base  

---

## 🛠️ Technology Stack

- **Python** – Core development language  
- **LangGraph** – Agentic routing & state management  
- **LangChain** – LLM orchestration  
- **ChromaDB** – Vector database for RAG  
- **OpenAI GPT-4o-mini** – Reasoning and response generation  
- **Streamlit** – Interactive user interface  

---

## 🚀 Key Features

- ✅ Accurate query routing  
- ✅ Context-aware RAG responses  
- ✅ Sentiment-based escalation  
- ✅ Multi-turn conversational memory  
- ✅ Scalable agentic architecture  

---

## 📊 Results & Highlights

- High-precision routing for both employee and customer queries  
- Relevant, department-specific responses  
- Automatic escalation for negative sentiment queries  
- Smooth multi-turn conversations  

---

## 🔮 Future Scope

- Add more departments and complex workflows  
- Integrate escalation channels (Email / WhatsApp)  
- Deploy as a production-grade API  
- Improve monitoring and analytics  

---

## 📎 Project Reference

This repository is based on the project presentation:  
**“Agentic AI Assistant for ShopUNow”** :contentReference[oaicite:0]{index=0}

---

⭐ If you like this project, feel free to star the repo!
