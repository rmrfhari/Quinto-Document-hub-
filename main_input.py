"""
FastAPI Application
Main API orchestrating file uploads, OCR processing, document extraction, and JSON responses
Endpoints for single/batch PDF processing with complete extraction pipeline
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import uuid

# Import custom modules
from ocr_engine import OCREngine, get_ocr_engine
from extractor import DocumentExtractor, get_extractor, ExtractionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing API",
    description="OCR and document extraction pipeline for international trade documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
ocr_engine = None
document_extractor = None

# In-memory storage for processing results (for demonstration)
# In production, use Redis or database
processing_results = {}


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize OCR engine and extractor on startup"""
    global ocr_engine, document_extractor
    
    try:
        logger.info("Initializing OCR Engine...")
        ocr_engine = get_ocr_engine(dpi=300)
        
        logger.info("Initializing Document Extractor...")
        document_extractor = get_extractor()
        
        logger.info("Application startup completed successfully")
    
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Application shutting down")


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Dict: Status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/info", tags=["Info"])
async def app_info():
    """
    Get application information
    
    Returns:
        Dict: API information and capabilities
    """
    return {
        "name": "Document Processing API",
        "version": "1.0.0",
        "description": "OCR and document extraction for trade documents",
        "supported_document_types": [
            "commercial_invoice",
            "packing_list",
            "bill_of_lading",
            "certificate_of_origin",
            "customs_declaration"
        ],
        "max_file_size_mb": 50,
        "supported_formats": ["PDF"],
        "ocr_languages": ["eng", "fra", "deu", "ita", "spa"]
    }


# ============================================================================
# Single File Processing
# ============================================================================

@app.post("/api/v1/process", tags=["Processing"])
async def process_single_document(
    file: UploadFile = File(...),
    ocr_language: str = "eng",
    enhance_preprocessing: bool = True
):
    """
    Process a single PDF document with complete extraction pipeline
    
    Args:
        file (UploadFile): PDF file to process
        ocr_language (str): OCR language code (default: 'eng')
        enhance_preprocessing (bool): Enable image preprocessing
        
    Returns:
        Dict: Processing result with extracted fields
        
    Raises:
        HTTPException: 400 for invalid file, 500 for processing errors
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Validate file
        if file.content_type != 'application/pdf':
            logger.warning(f"[{request_id}] Invalid file type: {file.content_type}")
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Save uploaded file to temp location
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
        
        logger.info(f"[{request_id}] Received file: {file.filename}")
        
        # Write file to disk
        with open(temp_file_path, "wb") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            file_size_mb = len(contents) / (1024 * 1024)
            logger.info(f"[{request_id}] File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 50:
            os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="File size exceeds 50 MB limit"
            )
        
        # OCR Processing
        logger.info(f"[{request_id}] Starting OCR processing...")
        if enhance_preprocessing:
            raw_text_dict = ocr_engine.extract_text_enhanced(
                temp_file_path,
                lang=ocr_language,
                preprocess=True
            )
        else:
            raw_text_dict = ocr_engine.extract_text(
                temp_file_path,
                lang=ocr_language
            )
        
        # Document Extraction
        logger.info(f"[{request_id}] Starting document extraction...")
        extraction_results = document_extractor.extract_batch(raw_text_dict)
        
        # Prepare response
        response_data = {
            "request_id": request_id,
            "filename": file.filename,
            "processing_status": "completed",
            "ocr_language": ocr_language,
            "total_pages": len(raw_text_dict),
            "extraction_results": [
                document_extractor.to_dict(result)
                for result in extraction_results.values()
            ],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Store results
        processing_results[request_id] = response_data
        
        # Cleanup temp file
        os.remove(temp_file_path)
        
        logger.info(f"[{request_id}] Processing completed successfully")
        
        return JSONResponse(
    content={
        "extraction_results": [
            document_extractor.to_dict(result)
            for result in extraction_results.values()
        ]
    },
    status_code=200
)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[{request_id}] Error processing document: {str(e)}", exc_info=True)
        
        # Cleanup on error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )


# ============================================================================
# Batch Processing
# ============================================================================

@app.post("/api/v1/process-batch", tags=["Processing"])
async def process_batch_documents(
    files: List[UploadFile] = File(...),
    ocr_language: str = "eng"
):
    """
    Process multiple PDF documents in batch
    
    Args:
        files (List[UploadFile]): List of PDF files to process
        ocr_language (str): OCR language code
        
    Returns:
        Dict: Batch processing results
        
    Raises:
        HTTPException: For validation or processing errors
    """
    request_id = str(uuid.uuid4())
    batch_results = []
    errors = []
    
    try:
        logger.info(f"[{request_id}] Starting batch processing for {len(files)} files")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if len(files) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files per batch"
            )
        
        # Process each file
        for idx, file in enumerate(files):
            try:
                logger.info(f"[{request_id}] Processing file {idx + 1}/{len(files)}: {file.filename}")
                
                # Validate file type
                if file.content_type != 'application/pdf':
                    errors.append({
                        "filename": file.filename,
                        "error": "Invalid file type. Only PDF supported."
                    })
                    continue
                
                # Save temp file
                temp_dir = tempfile.gettempdir()
                temp_file_path = os.path.join(
                    temp_dir,
                    f"{request_id}_{idx}_{file.filename}"
                )
                
                with open(temp_file_path, "wb") as temp_file:
                    contents = await file.read()
                    temp_file.write(contents)
                
                # Process
                raw_text_dict = ocr_engine.extract_text(temp_file_path, lang=ocr_language)
                extraction_results = document_extractor.extract_batch(raw_text_dict)
                
                batch_results.append({
                    "filename": file.filename,
                    "status": "success",
                    "pages": len(raw_text_dict),
                    "results": [
                        document_extractor.to_dict(result)
                        for result in extraction_results.values()
                    ]
                })
                
                # Cleanup
                os.remove(temp_file_path)
            
            except Exception as e:
                logger.error(f"[{request_id}] Error processing {file.filename}: {str(e)}")
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        response_data = {
            "request_id": request_id,
            "total_files": len(files),
            "successful": len(batch_results),
            "failed": len(errors),
            "processing_results": batch_results,
            "errors": errors,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        processing_results[request_id] = response_data
        logger.info(f"[{request_id}] Batch processing completed")
        
        return JSONResponse(content=response_data, status_code=200)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[{request_id}] Batch processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing error: {str(e)}"
        )


# ============================================================================
# Results Retrieval
# ============================================================================

@app.get("/api/v1/results/{request_id}", tags=["Results"])
async def get_results(request_id: str):
    """
    Retrieve processing results by request ID
    
    Args:
        request_id (str): Request ID from processing endpoint
        
    Returns:
        Dict: Processing results
        
    Raises:
        HTTPException: 404 if results not found
    """
    if request_id not in processing_results:
        raise HTTPException(
            status_code=404,
            detail=f"No results found for request ID: {request_id}"
        )
    
    logger.info(f"Retrieved results for request: {request_id}")
    return processing_results[request_id]


@app.get("/api/v1/results", tags=["Results"])
async def list_results():
    """
    List all available processing results
    
    Returns:
        Dict: Summary of all results
    """
    return {
        "total_requests": len(processing_results),
        "request_ids": list(processing_results.keys()),
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# Advanced Endpoints
# ============================================================================

@app.post("/api/v1/extract-fields", tags=["Extraction"])
async def extract_specific_fields(
    file: UploadFile = File(...),
    field_names: Optional[List[str]] = None,
    ocr_language: str = "eng"
):
    """
    Extract specific fields from a document
    
    Args:
        file (UploadFile): PDF file
        field_names (Optional[List[str]]): Specific fields to extract
        ocr_language (str): OCR language code
        
    Returns:
        Dict: Extracted fields matching the request
    """
    request_id = str(uuid.uuid4())
    
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
        
        with open(temp_file_path, "wb") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
        
        # OCR and Extraction
        raw_text_dict = ocr_engine.extract_text(temp_file_path, lang=ocr_language)
        extraction_results = document_extractor.extract_batch(raw_text_dict)
        
        # Filter fields if requested
        filtered_results = []
        for result in extraction_results.values():
            result_dict = document_extractor.to_dict(result)
            
            if field_names:
                filtered_fields = {
                    k: v for k, v in result_dict['extracted_fields'].items()
                    if k in field_names
                }
                result_dict['extracted_fields'] = filtered_fields
            
            filtered_results.append(result_dict)
        
        response_data = {
            "request_id": request_id,
            "filename": file.filename,
            "document_type": extraction_results[0].document_type,
            "fields_requested": field_names if field_names else "all",
            "extraction_results": filtered_results,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        os.remove(temp_file_path)
        
        return JSONResponse(content=response_data, status_code=200)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[{request_id}] Error extracting fields: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Document Type Detection
# ============================================================================

@app.post("/api/v1/detect-type", tags=["Detection"])
async def detect_document_type(
    file: UploadFile = File(...),
    ocr_language: str = "eng"
):
    """
    Detect document type without full extraction
    
    Args:
        file (UploadFile): PDF file
        ocr_language (str): OCR language code
        
    Returns:
        Dict: Document type and confidence score
    """
    request_id = str(uuid.uuid4())
    
    try:
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail="Only PDF files supported")
        
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{request_id}_{file.filename}")
        
        with open(temp_file_path, "wb") as temp_file:
            contents = await file.read()
            temp_file.write(contents)
        
        # OCR only
        raw_text_dict = ocr_engine.extract_text(temp_file_path, lang=ocr_language)
        
        # Classify only
        detection_results = []
        for page_num, text in raw_text_dict.items():
            doc_type, confidence = document_extractor.classifier.classify(text)
            detection_results.append({
                "page": page_num,
                "document_type": doc_type.value,
                "confidence": confidence
            })
        
        response_data = {
            "request_id": request_id,
            "filename": file.filename,
            "total_pages": len(raw_text_dict),
            "detection_results": detection_results,
            "primary_type": detection_results[0]['document_type'] if detection_results else "unknown",
            "detection_timestamp": datetime.now().isoformat()
        }
        
        os.remove(temp_file_path)
        
        return JSONResponse(content=response_data, status_code=200)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"[{request_id}] Error detecting document type: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Root and 404 Handling
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API documentation"""
    return {
        "message": "Document Processing API",
        "version": "1.0.0",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json"
    }


@app.get("/docs", tags=["Documentation"])
async def get_docs():
    """API documentation"""
    return {
        "endpoints": {
            "health": "GET /health",
            "info": "GET /info",
            "process": "POST /api/v1/process",
            "process_batch": "POST /api/v1/process-batch",
            "detect_type": "POST /api/v1/detect-type",
            "extract_fields": "POST /api/v1/extract-fields",
            "get_results": "GET /api/v1/results/{request_id}",
            "list_results": "GET /api/v1/results"
        },
        "documentation_url": "http://localhost:8000/docs"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "status_code": 500,
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Run FastAPI application
    # For production, use: gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
