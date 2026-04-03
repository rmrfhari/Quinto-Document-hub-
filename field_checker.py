"""
Quinto — Field Checker Module
Cross-document inconsistency detection using Gemini.
Called by main_input.py after OCR + extraction is complete.
"""

import google.generativeai as genai
import json
import sys

# ==========================================================
# UTF-8 OUTPUT FIX
# ==========================================================
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ==========================================================
# 1. CONFIGURATION & API SETUP
# ==========================================================
API_KEY = "AIzaSyD_cdViTI-pHfIMF5Nd4EzL7d5CcoiNktQ"
genai.configure(api_key=API_KEY)

# FIX: gemini-1.5-flash works on all SDK versions.
# Switch to "gemini-2.5-flash" only after: pip install -U google-generativeai
GEMINI_MODEL = "gemini-1.5-flash"

# ==========================================================
# 2. MASTER MATRIX
# ==========================================================
MASTER_REGISTRY = {
    "Total Quantity": {
        "base_risk_weight": 8.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Packing List", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 8},
            "Packing List":       {"present": True, "mandatory": "optional",  "weight": 6},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 8},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "Shipper Name & Address": {
        "base_risk_weight": 10.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Packing List", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 10},
            "Packing List":       {"present": True, "mandatory": "optional",  "weight": 5},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 10},
            "Certificate of Origin": {"present": True, "mandatory": "optional", "weight": 5},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "Consignee Name & Address": {
        "base_risk_weight": 10.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Packing List", "Bill of Lading", "Certificate of Origin"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 10},
            "Packing List":       {"present": True, "mandatory": "optional",  "weight": 5},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 10},
            "Certificate of Origin": {"present": True, "mandatory": "optional", "weight": 7},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "Total Gross Weight": {
        "base_risk_weight": 7.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Packing List", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 7},
            "Packing List":       {"present": True, "mandatory": "mandatory", "weight": 7},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 7},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "Declared Value (Total)": {
        "base_risk_weight": 9.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Customs Declaration"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 9},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": False, "mandatory": None, "weight": 0},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": True, "mandatory": "mandatory", "weight": 9},
        }
    },
    "Incoterms": {
        "base_risk_weight": 6.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 6},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 6},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": True, "mandatory": "optional", "weight": 3},
        }
    },
    "Port of Loading": {
        "base_risk_weight": 6.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 6},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 6},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "Port of Discharge": {
        "base_risk_weight": 6.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Bill of Lading"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 6},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": True, "mandatory": "mandatory", "weight": 6},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": False, "mandatory": None, "weight": 0},
        }
    },
    "HS Code": {
        "base_risk_weight": 7.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Customs Declaration"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "optional",  "weight": 5},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": False, "mandatory": None, "weight": 0},
            "Certificate of Origin": {"present": False, "mandatory": None, "weight": 0},
            "Customs Declaration":   {"present": True, "mandatory": "mandatory", "weight": 7},
        }
    },
    "Country of Origin": {
        "base_risk_weight": 8.0,
        "llm_call": True,
        "appearing_in": ["Commercial Invoice", "Certificate of Origin"],
        "doc_types": {
            "Commercial Invoice": {"present": True, "mandatory": "mandatory", "weight": 8},
            "Packing List":       {"present": False, "mandatory": None, "weight": 0},
            "Bill of Lading":     {"present": False, "mandatory": None, "weight": 0},
            "Certificate of Origin": {"present": True, "mandatory": "mandatory", "weight": 8},
            "Customs Declaration":   {"present": True, "mandatory": "optional", "weight": 5},
        }
    },
}

# ==========================================================
# 3. FIELD LOOKUP
# ==========================================================
def get_field_matrix_info(field_name: str, registry: dict) -> dict | None:
    if field_name in registry:
        return registry[field_name]
    lower = field_name.lower()
    for key, val in registry.items():
        if key.lower() == lower:
            return val
    return None


# ==========================================================
# 4. BUILD CONTEXT FOR A SINGLE FIELD
# ==========================================================
def build_field_context(target_field: str, documents: list, registry: dict) -> dict:
    matrix_info = get_field_matrix_info(target_field, registry)
    if not matrix_info:
        return {
            "error": f"Field '{target_field}' not found in Master Matrix.",
            "target_field": target_field,
        }

    doc_values = []
    for doc in documents:
        doc_type = doc.get("doc_type", "Unknown")
        value = doc["fields"].get(target_field)

        matrix_presence = matrix_info["doc_types"].get(doc_type, {})
        doc_values.append({
            "doc_type":        doc_type,
            "value":           value,
            "value_present":   value is not None,
            "matrix_expected": matrix_presence.get("present", False),
            "mandatory_level": matrix_presence.get("mandatory"),
            "matrix_weight":   matrix_presence.get("weight", 0),
        })

    return {
        "target_field":     target_field,
        "base_risk_weight": matrix_info["base_risk_weight"],
        "llm_call_flag":    matrix_info["llm_call"],
        "appearing_in":     matrix_info["appearing_in"],
        "doc_values":       doc_values,
    }


# ==========================================================
# 5. JSON CLEANING HELPER
# ==========================================================
def _clean_json_response(raw: str) -> str:
    """Strip markdown code fences Gemini sometimes wraps around JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        if first_newline != -1:
            raw = raw[first_newline + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
    return raw.strip()


# ==========================================================
# 6. SINGLE-FIELD ANALYSIS (calls Gemini)
# ==========================================================
def analyze_inconsistencies(target_field: str, documents: list, registry: dict) -> dict:
    context = build_field_context(target_field, documents, registry)
    if "error" in context:
        return context

    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
You are a trade document compliance expert. Find VALUE INCONSISTENCIES for a specific
field across multiple shipping documents belonging to the same shipment.

=== FIELD UNDER ANALYSIS ===
Field Name      : {context['target_field']}
Base Risk Weight: {context['base_risk_weight']} / 10
Matrix says field appears in: {", ".join(context['appearing_in']) if context['appearing_in'] else "Not specified"}

=== FIELD VALUES EXTRACTED FROM EACH DOCUMENT ===
{json.dumps(context['doc_values'], indent=2)}

=== YOUR TASK ===
Identify: VALUE MISMATCHES, PRESENCE VIOLATIONS, SEMANTIC MISMATCHES, RISK ASSESSMENT.

=== MANDATORY OUTPUT FORMAT (JSON ONLY - no markdown, no extra text) ===
{{
  "field": "<field name>",
  "base_risk_weight": <number>,
  "overall_severity": "CRITICAL | HIGH | MEDIUM | LOW | NONE",
  "inconsistencies_found": true | false,
  "summary": "<one-sentence plain-English summary>",
  "issues": [
    {{
      "type": "VALUE_MISMATCH | PRESENCE_VIOLATION | SEMANTIC_MISMATCH",
      "severity": "CRITICAL | HIGH | MEDIUM | LOW",
      "documents_involved": ["<doc_type_1>", "<doc_type_2>"],
      "details": "<what exactly is inconsistent and why it matters>",
      "values": {{
        "<doc_type_1>": "<value or null>",
        "<doc_type_2>": "<value or null>"
      }},
      "recommendation": "<suggested corrective action>"
    }}
  ],
  "compliant_documents": ["<doc types where the field value looks correct>"],
  "matrix_coverage_note": "<brief note on how many doc types expect this field vs. provided>"
}}
"""

    raw_text = ""
    try:
        response = model.generate_content(prompt)
        raw_text = response.text
        cleaned = _clean_json_response(raw_text)
        result = json.loads(cleaned)
        result["_matrix_context"] = context
        return result

    except json.JSONDecodeError as e:
        return {
            "error": "Gemini returned non-JSON output",
            "raw_response": raw_text,
            "parse_error": str(e),
        }
    except Exception as e:
        return {"error": "API call failed", "details": str(e)}


# ==========================================================
# 7. BATCH MODE — checks ALL registry fields across documents
#    This is what main_input.py calls.
# ==========================================================
def analyze_all_cross_doc_fields(documents: list, registry: dict) -> dict:
    """
    Iterate over every field in MASTER_REGISTRY that:
      - has llm_call = True
      - is expected in >= 2 document types
      - appears in at least one of the submitted documents

    Returns a dict keyed by field name with Gemini analysis results.
    """
    present_fields: set = set()
    for doc in documents:
        present_fields.update(doc["fields"].keys())

    results = {}
    for field_name, meta in registry.items():
        if not meta.get("llm_call"):
            continue
        expected_in = [dt for dt, info in meta["doc_types"].items() if info["present"]]
        if len(expected_in) < 2:
            continue
        if field_name not in present_fields:
            continue

        print(f"  -> Checking: {field_name} (risk={meta['base_risk_weight']})")
        results[field_name] = analyze_inconsistencies(field_name, documents, registry)

    return results


# ==========================================================
# 8. REPORT PRINTER (for CLI use)
# ==========================================================
def print_report(result: dict, mode: str = "single"):
    SEVERITY_LABEL = {
        "CRITICAL": "[CRITICAL]",
        "HIGH":     "[HIGH]",
        "MEDIUM":   "[MEDIUM]",
        "LOW":      "[LOW]",
        "NONE":     "[OK]",
    }

    def print_single(res: dict):
        if "error" in res:
            print(f"\n[ERROR] {res['error']}")
            if "details" in res:
                print(f"    {res['details']}")
            if "raw_response" in res:
                print(f"    Raw response preview: {res['raw_response'][:300]}")
            return

        sev = res.get("overall_severity", "UNKNOWN")
        print(f"\n{'='*70}")
        print(f"  FIELD  : {res.get('field', 'N/A')}")
        print(f"  RISK   : {res.get('base_risk_weight', '?')}/10")
        print(f"  STATUS : {SEVERITY_LABEL.get(sev, '[?]')} {sev}")
        print(f"  SUMMARY: {res.get('summary', '')}")
        print(f"{'='*70}")

        issues = res.get("issues", [])
        if not issues:
            print("  [OK] No inconsistencies detected.")
        else:
            for i, issue in enumerate(issues, 1):
                sev_i = issue.get("severity", "?")
                print(f"\n  Issue #{i} [{issue.get('type','?')}]  Severity: {SEVERITY_LABEL.get(sev_i,'[?]')} {sev_i}")
                print(f"  Documents : {', '.join(issue.get('documents_involved', []))}")
                print(f"  Details   : {issue.get('details', '')}")
                vals = issue.get("values", {})
                if vals:
                    print("  Values    :")
                    for doc_t, val in vals.items():
                        print(f"              {doc_t:35s} -> {val}")
                print(f"  Fix       : {issue.get('recommendation', '')}")

        compliant = res.get("compliant_documents", [])
        if compliant:
            print(f"\n  [OK] Compliant docs : {', '.join(compliant)}")
        note = res.get("matrix_coverage_note", "")
        if note:
            print(f"  [i]  Matrix note    : {note}")

    if mode == "single":
        print_single(result)
    else:
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "NONE": 4, "UNKNOWN": 5}
        sorted_items = sorted(
            result.items(),
            key=lambda kv: severity_order.get(kv[1].get("overall_severity", "UNKNOWN"), 5)
        )
        print("\n" + "="*70)
        print("  BATCH CROSS-DOCUMENT INCONSISTENCY REPORT")
        print("="*70)
        issues_found = sum(1 for _, r in sorted_items if r.get("inconsistencies_found"))
        print(f"  Fields analysed   : {len(sorted_items)}")
        print(f"  Fields with issues: {issues_found}")
        print("="*70)
        for _, res in sorted_items:
            print_single(res)


# ==========================================================
# 9. CLI ENTRY POINT (standalone use)
# ==========================================================
SAMPLE_DOCUMENTS = [
    {
        "doc_type": "Commercial Invoice",
        "fields": {
            "Shipper Name & Address": "Acme Exports Pvt Ltd, Mumbai, India",
            "Consignee Name & Address": "Global Imports LLC, New York, USA",
            "Total Quantity": 500,
            "Product Description": "Cotton T-Shirts",
            "Country of Origin": "India",
            "Country of Destination": "USA",
            "Total Gross Weight": 320.5,
            "Total Net Weight": 300.0,
            "Incoterms": "CIF",
            "Invoice Number": "INV-2024-001",
            "Invoice Date": "2024-01-15",
            "Declared Value (Total)": 15000,
            "Currency of Transaction": "USD",
            "HS Code": "6109.10",
            "Port of Loading": "INNSA",
            "Port of Discharge": "USLAX",
            "Place of Issue": "Mumbai",
        }
    },
    {
        "doc_type": "Packing List",
        "fields": {
            "Shipper Name & Address": "Acme Exports Pvt Ltd, Mumbai",  # trimmed address
            "Consignee Name & Address": "Global Imports LLC, New York, USA",
            "Total Quantity": 480,                                       # mismatch
            "Product Description": "Cotton T-Shirts (Men)",
            "Total Gross Weight": 320.5,
            "Total Net Weight": 298.0,
            "Number of Packages": 20,
            "Packing List Number": "PL-2024-001",
        }
    },
    {
        "doc_type": "Bill of Lading",
        "fields": {
            "Shipper Name & Address": "Acme Exports Pvt Ltd, Mumbai, India",
            "Consignee Name & Address": "Global Imports LLC, New York, USA",
            "Total Quantity": 500,
            "Product Description": "Cotton T-Shirts",
            "Country of Destination": "United States",
            "Total Gross Weight": 320.5,
            "Port of Loading": "INNSA",
            "Port of Discharge": "USLAX",
            "Incoterms": "FOB",                                          # mismatch
            "Number of Packages": 20,
            "Bill of Lading Number": "BOL-2024-001",
            "Vessel Name": "MV Ocean Star",
            "Voyage Number": "VS-001",
            "On-board Date": "2024-01-20",
            "Place of Issue": "Mumbai",
            "Container Number": "TCKU3456789",
            "Seal Number": "SL12345",
        }
    }
]

if __name__ == "__main__":
    registry = MASTER_REGISTRY
    print(f"Loaded {len(registry)} fields.  Model: {GEMINI_MODEL}\n")

    batch_mode = "--all" in sys.argv

    if batch_mode:
        print("MODE: BATCH\n")
        results = analyze_all_cross_doc_fields(SAMPLE_DOCUMENTS, registry)
        print_report(results, mode="batch")
        out_path = "inconsistency_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[SAVED] {out_path}")
    else:
        TARGET = "Total Quantity"
        print(f"MODE: SINGLE FIELD — '{TARGET}'\n")
        result = analyze_inconsistencies(TARGET, SAMPLE_DOCUMENTS, registry)
        print_report(result, mode="single")
        out_path = "inconsistency_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n[SAVED] {out_path}")
