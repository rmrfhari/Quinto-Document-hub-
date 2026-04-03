"""
Document Extractor Module
Classifies document types and extracts structured data into normalized JSON format
Supports: Commercial Invoice, Packing List, Bill of Lading, Certificate of Origin, Customs Declaration
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types"""
    COMMERCIAL_INVOICE = "commercial_invoice"
    PACKING_LIST = "packing_list"
    BILL_OF_LADING = "bill_of_lading"
    CERTIFICATE_OF_ORIGIN = "certificate_of_origin"
    CUSTOMS_DECLARATION = "customs_declaration"
    UNKNOWN = "unknown"


@dataclass
class ExtractionResult:
    """Standardized extraction result"""
    document_type: str
    confidence: float
    extracted_fields: Dict[str, Any]
    raw_text: str
    quality_score: float
    warnings: List[str]
    extraction_timestamp: str


class DocumentClassifier:
    """
    Improved classifier using:
    - Priority keyword rules
    - Weighted scoring
    - Tie-breakers
    - Better thresholds
    """

    # Weighted keywords (importance-based)
    CLASSIFICATION_KEYWORDS = {
        DocumentType.COMMERCIAL_INVOICE: {
            'invoice': 3, 'bill to': 2, 'ship to': 2,
            'total amount': 3, 'amount due': 3,
            'invoice number': 3, 'invoice date': 2,
            'tax': 2, 'payment terms': 2,
            'unit price': 2, 'quantity': 2
        },

        DocumentType.PACKING_LIST: {
            'packing list': 5, 'contents': 2, 'carton': 2,
            'package': 2, 'qty': 2, 'weight': 3,
            'dimension': 3, 'description of goods': 2
        },

        DocumentType.BILL_OF_LADING: {
            'bill of lading': 5, 'b/l': 4, 'bol': 4,
            'carrier': 2, 'vessel': 3,
            'port of loading': 3, 'port of discharge': 3,
            'shipper': 3, 'consignee': 3
        },

        DocumentType.CERTIFICATE_OF_ORIGIN: {
            'certificate of origin': 5,
            'origin certificate': 4,
            'manufactured in': 3,
            'exporter': 3,
            'importer': 3,
            'authorized signature': 2,
            'declaration': 1,  # weak keyword
            'country of origin': 1  # shared keyword → low weight
        },

        DocumentType.CUSTOMS_DECLARATION: {
            'customs declaration': 5,
            'hs code': 4,
            'tariff code': 4,
            'duties': 3,
            'customs value': 3,
            'incoterms': 2,
            'total value': 3,
            'declarant': 3,
            'declaration date': 3,
            'country of origin': 1  # shared keyword → low weight
        }
    }

    def classify(self, raw_text: str) -> Tuple[DocumentType, float]:
        text_lower = raw_text.lower()

        # 🔥 PRIORITY RULES (most important fix)
        if "customs declaration" in text_lower:
            return DocumentType.CUSTOMS_DECLARATION, 0.99

        if "certificate of origin" in text_lower:
            return DocumentType.CERTIFICATE_OF_ORIGIN, 0.99

        if "bill of lading" in text_lower:
            return DocumentType.BILL_OF_LADING, 0.99

        if "packing list" in text_lower:
            return DocumentType.PACKING_LIST, 0.99

        # 🔥 WEIGHTED SCORING
        scores = {}

        for doc_type, keywords in self.CLASSIFICATION_KEYWORDS.items():
            score = 0
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    score += weight
            scores[doc_type] = score

        # Normalize scores (optional but useful)
        max_possible_scores = {
            doc: sum(keywords.values())
            for doc, keywords in self.CLASSIFICATION_KEYWORDS.items()
        }

        normalized_scores = {
            doc: (scores[doc] / max_possible_scores[doc]) if max_possible_scores[doc] > 0 else 0
            for doc in scores
        }

        # Find best match
        best_type = max(normalized_scores, key=normalized_scores.get)
        best_score = normalized_scores[best_type]

        # 🔥 TIE-BREAKER LOGIC
        # Handle confusion between Customs vs Certificate
        if abs(normalized_scores.get(DocumentType.CUSTOMS_DECLARATION, 0) -
               normalized_scores.get(DocumentType.CERTIFICATE_OF_ORIGIN, 0)) < 0.05:

            if "hs code" in text_lower or "tariff code" in text_lower:
                return DocumentType.CUSTOMS_DECLARATION, normalized_scores[DocumentType.CUSTOMS_DECLARATION]

        # 🔥 BETTER THRESHOLD
        if best_score < 0.25:
            return DocumentType.UNKNOWN, best_score

        logger.info(f"Classified document as {best_type.value} with confidence {best_score:.2%}")
        return best_type, best_score


class FieldExtractor:
    """
    Extracts specific fields from classified documents
    """
    
    # Field extraction patterns for each document type
    EXTRACTION_PATTERNS = {
        DocumentType.COMMERCIAL_INVOICE: {
            'invoice_number': r'(?:invoice\s*(?:no|number|#)?\s*[:=]?\s*)([A-Z0-9\-\/]+)',
            'invoice_date': r'(?:invoice\s*date\s*[:=]?\s*)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            'buyer': r'(?:bill\s*to[:\s]+)([\w\s&.,]+?)(?:\n|bill|ship)',
            'seller': r'(?:from|seller|vendor)[:\s]+([\w\s&.,]+?)(?:\n|invoice)',
            'total_amount': r'(?:total\s*(?:amount|due)?\s*[:=]?\s*)([0-9,]+\.?\d*)',
            'currency': r'(?:currency|amount|total)\s*\(?\s*([A-Z]{3})?\s*\)?',
            'tax': r'(?:tax|gst|vat)\s*[:=]?\s*([0-9,]+\.?\d*)',
            'items': r'(?:(?:item|product|description)\s*[#:]?\s*)(.*?)(?:quantity|qty|unit)',
        },
        DocumentType.PACKING_LIST: {
            'packing_list_number': r'(?:packing\s*list\s*(?:no|number)?\s*[:=]?\s*)([A-Z0-9\-\/]+)',
            'contents': r'(?:contents?|description)\s*[:=]?\s*([\w\s,\.]+?)(?:\n|qty|quantity)',
            'total_items': r'(?:total\s*(?:items?|qty|pieces?))\s*[:=]?\s*(\d+)',
            'weight': r'(?:weight|wgt)\s*[:=]?\s*([0-9,]+\.?\d*)\s*(kg|lbs?|mt)?',
            'dimensions': r'(?:dimension|size)\s*[:=]?\s*([\d,\.]+\s*x\s*[\d,\.]+\s*x\s*[\d,\.]+)',
            'date_packed': r'(?:date\s*(?:packed|packing))\s*[:=]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        },
        DocumentType.BILL_OF_LADING: {
            'bol_number': r'(?:bill\s*of\s*lading|b\/l|bol)\s*(?:no|number)?\s*[:=]?\s*([A-Z0-9\-\/]+)',
            'vessel_name': r'(?:vessel|ship|vessel\s*name)\s*[:=]?\s*([\w\s]+?)(?:\n|voyage)',
            'voyage_number': r'(?:voyage|voyage\s*(?:no|number))\s*[:=]?\s*([A-Z0-9\-\/]+)',
            'port_of_loading': r'(?:port\s*of\s*(?:loading|shipment))\s*[:=]?\s*([A-Z\s,]+?)(?:\n|port\s*of)',
            'port_of_discharge': r'(?:port\s*of\s*(?:discharge|destination))\s*[:=]?\s*([A-Z\s,]+?)(?:\n|final)',
            'shipper': r'(?:shipper)[:\s]+([\w\s&.,]+?)(?:\n|consignee)',
            'consignee': r'(?:consignee)[:\s]+([\w\s&.,]+?)(?:\n|notify)',
        },
        DocumentType.CERTIFICATE_OF_ORIGIN: {
            'certificate_number': r'(?:certificate\s*(?:no|number))\s*[:=]?\s*([A-Z0-9\-\/]+)',
            'exporter': r'(?:exporter)[:\s]+([\w\s&.,]+?)(?:\n|importer)',
            'importer': r'(?:importer|buyer)[:\s]+([\w\s&.,]+?)(?:\n|country)',
            'country_of_origin': r'(?:country\s*of\s*origin)\s*[:=]?\s*([A-Z\s,]+?)(?:\n|product)',
            'issue_date': r'(?:issue\s*date|date\s*of\s*issue)\s*[:=]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        },
        DocumentType.CUSTOMS_DECLARATION: {
            'declaration_number': r'(?:declaration|form)\s*(?:no|number)?\s*[:=]?\s*([A-Z0-9\-\/]+)',
            'hs_code': r'(?:hs\s*code|tariff\s*code)\s*[:=]?\s*([0-9]{4,10})',
            'total_value': r'(?:total\s*(?:value|amount))\s*[:=]?\s*([0-9,]+\.?\d*)',
            'declarant_name': r'(?:declarant|declaration\s*by)\s*[:=]?\s*([\w\s&.,]+?)(?:\n|date)',
            'declaration_date': r'(?:declaration\s*date|date)\s*[:=]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        }
    }
    
    def extract_fields(self, raw_text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """
        Extract fields based on document type
        
        Args:
            raw_text (str): Raw OCR text
            doc_type (DocumentType): Classified document type
            
        Returns:
            Dict[str, Any]: Extracted fields with values
        """
        extracted = {}
        
        if doc_type not in self.EXTRACTION_PATTERNS:
            logger.warning(f"No extraction patterns for document type: {doc_type}")
            return extracted
        
        patterns = self.EXTRACTION_PATTERNS[doc_type]
        
        for field_name, pattern in patterns.items():
            try:
                match = re.search(pattern, raw_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if match:
                    value = match.group(1).strip() if match.groups() else match.group(0).strip()
                    extracted[field_name] = self._normalize_field(field_name, value)
                    logger.debug(f"Extracted {field_name}: {extracted[field_name]}")
            
            except Exception as e:
                logger.error(f"Error extracting field {field_name}: {str(e)}")
        
        return extracted
    
    def _normalize_field(self, field_name: str, value: str) -> Any:
        """
        Normalize field values based on type
        
        Args:
            field_name (str): Field name
            value (str): Raw field value
            
        Returns:
            Any: Normalized value
        """
        try:
            # Normalize numbers
            if any(x in field_name for x in ['amount', 'total', 'weight', 'price', 'qty', 'quantity']):
                # Remove currency symbols and convert to float
                value = re.sub(r'[^\d.,\-]', '', value)
                value = value.replace(',', '')
                return float(value) if value else 0.0
            
            # Normalize dates
            if any(x in field_name for x in ['date', 'time']):
                return self._parse_date(value)
            
            # Keep as string
            return value.strip()
        
        except Exception as e:
            logger.warning(f"Error normalizing field {field_name}: {str(e)}")
            return value
    
    def _parse_date(self, date_str: str) -> str:
        """
        Parse and normalize date formats
        
        Args:
            date_str (str): Date string in various formats
            
        Returns:
            str: Normalized ISO format date (YYYY-MM-DD)
        """
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%m/%d/%Y', '%m-%d-%Y',
            '%Y/%m/%d', '%Y-%m-%d', '%d.%m.%Y', '%B %d, %Y',
            '%d %B %Y', '%d %b %Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str.strip(), fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return date_str  # Return original if parsing fails


class DocumentValidator:
    """
    Validates extracted fields against document requirements
    """
    
    # Required fields for each document type
    REQUIRED_FIELDS = {
        DocumentType.COMMERCIAL_INVOICE: [
            'invoice_number', 'invoice_date', 'total_amount'
        ],
        DocumentType.PACKING_LIST: [
            'packing_list_number', 'contents'
        ],
        DocumentType.BILL_OF_LADING: [
            'bol_number', 'shipper', 'consignee'
        ],
        DocumentType.CERTIFICATE_OF_ORIGIN: [
            'certificate_number', 'country_of_origin'
        ],
        DocumentType.CUSTOMS_DECLARATION: [
            'declaration_number', 'total_value'
        ]
    }
    
    def validate(self, doc_type: DocumentType, extracted_fields: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate extracted fields
        
        Args:
            doc_type (DocumentType): Document type
            extracted_fields (Dict): Extracted fields
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_warnings)
        """
        warnings = []
        
        if doc_type not in self.REQUIRED_FIELDS:
            return True, []
        
        required = self.REQUIRED_FIELDS[doc_type]
        missing = [f for f in required if f not in extracted_fields or not extracted_fields[f]]
        
        if missing:
            warnings.append(f"Missing required fields: {', '.join(missing)}")
        
        # Additional validation
        for field, value in extracted_fields.items():
            if value is None or (isinstance(value, str) and len(value) == 0):
                warnings.append(f"Field '{field}' is empty")
        
        is_valid = len([w for w in warnings if w.startswith('Missing required')]) == 0
        return is_valid, warnings


class DocumentExtractor:
    """
    Main extraction orchestrator combining classification, extraction, and validation
    """
    
    def __init__(self):
        self.classifier = DocumentClassifier()
        self.field_extractor = FieldExtractor()
        self.validator = DocumentValidator()
    
    def extract(self, raw_text: str, page_num: int = 0) -> ExtractionResult:
        """
        Complete extraction pipeline
        
        Args:
            raw_text (str): Raw OCR text from document
            page_num (int): Page number for reference
            
        Returns:
            ExtractionResult: Structured extraction result
        """
        try:
            # Step 1: Classify document
            doc_type, classification_confidence = self.classifier.classify(raw_text)
            logger.info(f"Page {page_num}: Classified as {doc_type.value}")
            
            # Step 2: Extract fields
            extracted_fields = self.field_extractor.extract_fields(raw_text, doc_type)
            logger.info(f"Page {page_num}: Extracted {len(extracted_fields)} fields")
            
            # Step 3: Validate
            is_valid, warnings = self.validator.validate(doc_type, extracted_fields)
            
            # Calculate quality score (0-100)
            quality_score = (
                classification_confidence * 50 +
                (len(extracted_fields) / 10) * 30 +
                (100 if is_valid else 50)
            ) / 100 * 100
            quality_score = min(100, max(0, quality_score))
            
            return ExtractionResult(
                document_type=doc_type.value,
                confidence=classification_confidence,
                extracted_fields=extracted_fields,
                raw_text=raw_text[:500] + "..." if len(raw_text) > 500 else raw_text,
                quality_score=quality_score,
                warnings=warnings,
                extraction_timestamp=datetime.now().isoformat()
            )
        
        except Exception as e:
            logger.error(f"Error in extraction: {str(e)}")
            return ExtractionResult(
                document_type=DocumentType.UNKNOWN.value,
                confidence=0.0,
                extracted_fields={},
                raw_text=raw_text[:500],
                quality_score=0.0,
                warnings=[str(e)],
                extraction_timestamp=datetime.now().isoformat()
            )
    
    def extract_batch(self, raw_text_dict: Dict[int, str]) -> Dict[int, ExtractionResult]:
        """
        Extract from multiple pages
        
        Args:
            raw_text_dict (Dict[int, str]): Dictionary mapping page numbers to raw text
            
        Returns:
            Dict[int, ExtractionResult]: Extraction results for each page
        """
        results = {}
        for page_num, text in raw_text_dict.items():
            results[page_num] = self.extract(text, page_num)
        
        return results
    
    def to_json(self, extraction_result: ExtractionResult) -> str:
        """
        Convert extraction result to JSON
        
        Args:
            extraction_result (ExtractionResult): Extraction result object
            
        Returns:
            str: JSON string
        """
        return json.dumps(asdict(extraction_result), indent=2, default=str)
    
    def to_dict(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """
        Convert extraction result to dictionary
        
        Args:
            extraction_result (ExtractionResult): Extraction result object
            
        Returns:
            Dict: Dictionary representation
        """
        return asdict(extraction_result)


# Utility function
def get_extractor() -> DocumentExtractor:
    """Factory function to create extractor instance"""
    return DocumentExtractor()
