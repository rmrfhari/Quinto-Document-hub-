# Quinto Document Hub
### AI-Powered Trade Document Validation

Quinto is a full-stack, AI-driven application that automates the classification, extraction, and cross-validation of international trade shipping documents.

It leverages:
- Tesseract OCR  
- Regex-based heuristic extraction  
- Google Gemini LLM  

The system detects discrepancies such as:
- Weight mismatches  
- Missing HS codes  
- Conflicting consignee details  

---

## ✨ Features

### 🔍 Intelligent OCR Pipeline
- Converts PDFs into pre-processed images  
- Extracts raw text using Tesseract OCR  

### 📄 Automated Document Classification
Identifies:
- Commercial Invoices  
- Packing Lists  
- Bills of Lading  
- Certificates of Origin  
- Customs Declarations  

Uses weighted keyword scoring.

### 🧾 Structured Field Extraction
Extracts:
- Dates  
- Weights  
- HS Codes  
- Values  
- Incoterms  

Using targeted regex patterns.

### 🤖 Cross-Document AI Validation
- Powered by **Gemini 1.5 Flash**  
- Detects:
  - Value mismatches  
  - Semantic inconsistencies  
  - Missing required fields  
- Based on a strict Master Registry of trade rules  

### 💻 Modern Web UI
- Built with Tailwind CSS and Alpine.js  
- Features:
  - Drag-and-drop upload  
  - Interactive discrepancy dashboard  

---

## 🏗️ Architecture & File Structure

| File | Description |
|------|------------|
| `main_input.py` | FastAPI backend orchestrator (handles uploads, pipeline, frontend serving) |
| `ocr_engine.py` | PDF → image → text extraction using `pdf2image`, `Pillow`, `pytesseract` |
| `extractor.py` | Text normalization, classification, and structured JSON extraction |
| `field_checker.py` | Gemini API interface for cross-document validation |
| `landingpage.html` | Upload UI (served at `/`) |
| `output.html` | Results dashboard (served at `/output.html`) |

---

## ⚙️ Prerequisites

### System Dependencies

#### 1. Poppler (required for `pdf2image`)
- **Mac (Homebrew):**
  ```bash
  brew install poppler
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install poppler-utils
  ```
- **Windows:**
  - Download Poppler binaries  
  - Add to system PATH  

#### 2. Tesseract OCR (required for `pytesseract`)
- **Mac (Homebrew):**
  ```bash
  brew install tesseract
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get install tesseract-ocr
  ```
- **Windows:**
  - Install Tesseract  
  - Add to system PATH  

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
cd /path/to/your/quinto/repo
```

### 2. Install Python dependencies (critical)
```bash
pip install fastapi uvicorn python-multipart pytesseract pdf2image pillow google-generativeai
```

### 3. Configure Gemini API key
- The key is currently hardcoded in `field_checker.py`
- Replace:
  ```python
  API_KEY = "..."
  ```
- Recommended (production):
  - Use environment variables instead  

---

## 💻 Running the Application

### 1. Start the FastAPI server
```bash
python main_input.py
```

**Alternative (Uvicorn):**
```bash
uvicorn main_input:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Open the web interface
```
http://localhost:8000
```

Upload sample trade PDFs to test the pipeline.

---

## 🛠️ Standalone Testing

Run validation logic without the web server:

```bash
# Single-field check
python field_checker.py

# Full batch cross-document check
python field_checker.py --all
```

---

## 📌 Notes
- Ensure Poppler and Tesseract are correctly added to PATH  
- OCR accuracy depends on document quality  
- Gemini API usage may incur costs  

---
