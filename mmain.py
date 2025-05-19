import fitz  # PyMuPDF
from pdf2image import convert_from_path
import PyPDF2
import spacy
import pytesseract
from PIL import Image
import pandas as pd
from fuzzywuzzy import fuzz
import re
import os
import datetime
from pathlib import Path
import glob
from typing import Dict, List, Tuple, Union
import logging
import traceback



# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")



# AJCC TNM dictionary with improved keywords and context
ajcc_tnm = {
    "TX": ["primary", "tumor", "cannot", "be", "assessed", "not", "visualized", "TX:", "pTX", "cTX", "ypTX", "cannot", "evaluate", "measured"],
    "T0": ["no", "evidence", "of", "primary", "tumor", "identified", "T0:", "pT0", "cT0", "ypT0", "residual", "identifiable"],
    "Tis (DCIS)": ["ductal", "carcinoma", "in", "situ", "DCIS", "intraductal", "Tis", "(DCIS)", "pTis", "(DCIS)", "non-invasive", "pure"],
    "Tis (Paget)": ["Paget", "disease", "Paget's", "nipple", "involvement", "only", "Tis", "(Paget)", "pTis", "(Paget)"],
    "T1mi": ["microinvasion", "microinvasive", "carcinoma", "T1mi", "pT1mi", "invasive", "focus", "≤", "0.1", "cm", "focal"],
    "T1a": ["tumor", ">", "0.1", "cm", "but", "≤", "0.5", "0.1-0.5", "T1a", "pT1a", "cT1a", "ypT1a", "invasive", "carcinoma", "0.2", "0.3", "0.4"],
    "T1b": ["tumor", ">", "0.5", "cm", "but", "≤", "1.0", "0.5-1.0", "T1b", "pT1b", "cT1b", "ypT1b", "invasive", "carcinoma", "0.6", "0.7", "0.8", "0.9"],
    "T1c": ["tumor", ">", "1.0", "cm", "but", "≤", "2.0", "1.0-2.0", "T1c", "pT1c", "cT1c", "ypT1c", "invasive", "carcinoma", "1.2", "1.5", "1.8"],
    "T2": ["tumor", ">", "2.0", "cm", "but", "≤", "5.0", "2.0-5.0", "T2", "pT2", "cT2", "ypT2", "invasive", "carcinoma", "2.5", "3.0", "4.0", "4.5"],
    "T3": ["tumor", ">", "5.0", "cm", "size", "greater", "than", "5", "T3", "pT3", "cT3", "ypT3", "invasive", "carcinoma", "5.5", "6.0", "7.0"],
    "T4a": ["chest", "wall", "invasion", "extension", "T4a", "pT4a", "cT4a", "ypT4a", "invasion", "of", "pectoralis", "muscle", "direct"],
    "T4b": ["skin", "ulceration", "satellite", "nodules", "edema", "T4b", "pT4b", "cT4b", "ypT4b", "peau", "d'orange", "invasion"],
    "T4c": ["both", "chest", "wall", "invasion", "and", "skin", "changes", "T4c", "pT4c", "cT4c", "ypT4c", "criteria", "of", "T4a", "T4b"],
    "T4d": ["inflammatory", "carcinoma", "breast", "cancer", "clinical", "T4d", "pT4d", "cT4d", "ypT4d", "dermal", "lymphatic", "invasion"],

    "N0": ["no", "regional", "lymph", "node", "metastasis", "negative", "0/16", "0", "of", "17", "N0", "pN0", "cN0", "ypN0"],
    "N1mi": ["micrometastases", "≤", "2.0", "mm", "<", "2.0", "0.2", "focus", "N1mi", "pN1mi", "small", "cluster"],
    "N1a": ["metastases", "in", "1-3", "axillary", "nodes", "1", "2", "3", "N1a", "pN1a", "cN1a", "ypN1a"],
    "N1b": ["metastases", "in", "internal", "mammary", "nodes", "positive", "sentinel", "N1b", "pN1b", "cN1b", "ypN1b"],
    "N1c": ["metastases", "in", "1-3", "axillary", "and", "internal", "mammary", "nodes", "N1c", "pN1c", "cN1c", "ypN1c"],
    "N2a": ["metastases", "in", "4-9", "axillary", "nodes", "4", "5", "6", "7", "8", "9", "N2a", "pN2a", "cN2a", "ypN2a"],
    "N3a": ["metastases", "in", "≥10", "axillary", "nodes", "10", "11", "12", "N3a", "pN3a", "cN3a", "ypN3a"],

    "M0": ["no", "distant", "metastasis", "evidence", "M0", "pM0", "cM0", "negative", "workup"],
    "M1": ["distant", "metastasis", "present", "evidence", "to", "bone", "liver", "lung", "brain", "M1", "pM1", "cM1"]
}




# Extract text from a PDF with improved error handling
def extract_text_from_pdf(pdf_path):
    try:
        pdf_doc = fitz.open(pdf_path)
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        pdf_doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""



def extract_age_or_dob(report_text):
    age = None
    report_date = None
    
    # Current date as fallback
    current_date = datetime.date.today()
    
    # Improved regex patterns with more variations
    age_pattern = r"\b(?:Age|AGE|age|Patient Age|PATIENT AGE):?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years\s*old|y\.o\.|yo|years|yrs|year old)\b"
    
    # More comprehensive report date pattern
    report_date_pattern = r"\b(?:Report Date|Date of Report|Reported on|REPORT DATE|SPECIMEN COLLECTION DATE/TIME|Date of Collection|Specimen Received Date|Date of Service|COLLECTION DATE|ACCESSION DATE|Laboratory Report Date|Issued on|FINAL REPORT DATE|RESULT DATE|Sample Collection Date|Date and Time of Report|Test Performed Date|Date of Procedure|Date of Specimen|Report Generated On):?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"

    # More comprehensive DOB pattern
    dob_pattern = r"\b(?:DOB|dob|D\.O\.B\.|D\.O\.B|Date of Birth|DATE OF BIRTH|Birth Date|BIRTH DATE|Born on):?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})"

    # Search for age
    age_match = re.search(age_pattern, report_text)
    if age_match:
        age = next(filter(None, age_match.groups()))  # Extract first non-None match
        return int(age)

    # Search for report date
    report_date_match = re.search(report_date_pattern, report_text)
    if report_date_match:
        report_date_str = report_date_match.group(1)
        try:
            # Try multiple date formats
            for date_format in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y"]:
                try:
                    report_date = datetime.datetime.strptime(report_date_str, date_format).date()
                    break
                except ValueError:
                    continue
        except Exception as e:
            print(f"Error parsing report date: {e}")

    # Use extracted report date, or fallback to current date
    report_date = report_date if report_date else current_date

    # Search for DOB
    dob_match = re.search(dob_pattern, report_text)
    if dob_match:
        dob_str = dob_match.group(1)
        try:
            # Try multiple date formats
            for date_format in ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%m-%d-%Y"]:
                try:
                    dob = datetime.datetime.strptime(dob_str, date_format).date()
                    # Calculate age based on report date
                    age = report_date.year - dob.year - ((report_date.month, report_date.day) < (dob.month, dob.day))
                    return age
                except ValueError:
                    continue
        except Exception as e:
            print(f"Error parsing DOB: {e}")

    return None





# Improved OCR for scanned PDFs
def ocr_from_image(pdf_path):
    try:
        pdf_doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(pdf_doc)):
            page = pdf_doc[page_num]
            # Higher resolution for better OCR results
            pix = page.get_pixmap(matrix=fitz.Matrix(4, 4))  # Increased resolution
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Save image for debugging (optional)
            img.save(f"page_{page_num}.png")
            # Perform OCR
            text += pytesseract.image_to_string(img, config='--psm 6 --oem 3')
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        traceback.print_exc()
        return ""



# Direct TNM pattern extraction - new function for more accurate stage detection
# Enhanced TNM pattern extraction with multiple approaches
def extract_direct_tnm_patterns(text):
    results = {"T": None, "N": None, "M": None}
    
    # More comprehensive patterns with variations
    t_patterns = [
        r"(?:T Classification|T Stage|T category|Tumor Stage|Primary Tumor|pT|cT|ypT|T)[\s:]+([T][0-4][a-d]?(?:\([^)]+\))?|Tis(?:\s*\([^)]+\))?|T[Xx])",
        r"(?:Tumor|Primary Tumor|Invasive Carcinoma)[^\n.]*?([T][0-4][a-d]?(?:\([^)]+\))?|Tis(?:\s*\([^)]+\))?|T[Xx])",
        r"Stage.*?([T][0-4][a-d]?(?:\([^)]+\))?|Tis(?:\s*\([^)]+\))?|T[Xx])N[0-3]",
        r"tumor size.*?(\d+(?:\.\d+)?)\s*(?:mm|cm|centimeter|millimeter)"
    ]
    
    n_patterns = [
        r"(?:N Classification|N Stage|N category|Nodal Stage|Regional Lymph Nodes|pN|cN|ypN|N)[\s:]+([N][0-3][a-c]?(?:\s?[mi]+)?|N[Xx])",
        r"(?:lymph node|node).*?([N][0-3][a-c]?(?:\s?[mi]+)?|N[Xx])",
        r"(\d+)\s*(?:of|\/)\s*(\d+)\s*(?:lymph )?nodes?(?:\s*positive|\s*involved)",
        r"Stage.*?T[0-4][a-d]([N][0-3][a-c]?)"
    ]
    
    m_patterns = [
        r"(?:M Classification|M Stage|M category|Metastasis Stage|Distant Metastasis|pM|cM|M)[\s:]+([M][0-1]|M[Xx])",
        r"(?:metastasis|metastases).*?([M][0-1]|M[Xx])",
        r"Stage.*?T[0-4][a-d]N[0-3][a-c]?([M][0-1])",
        r"(?:no evidence of|without|with)\s+(?:distant\s+)?metastatic disease"
    ]
    
    # Process T stage
    for pattern in t_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            # Handle tumor size conversion to T stage
            if pattern == t_patterns[3]:  # Size pattern
                size_mm = float(match.group(1))
                if 'mm' not in match.group():
                    size_mm *= 10  # Convert cm to mm
                
                if size_mm <= 1:
                    results["T"] = "T1mi"
                elif size_mm <= 5:
                    results["T"] = "T1a"
                elif size_mm <= 10:
                    results["T"] = "T1b"
                elif size_mm <= 20:
                    results["T"] = "T1c"
                elif size_mm <= 50:
                    results["T"] = "T2"
                else:
                    results["T"] = "T3"
            else:
                t_value = match.group(1)
                if t_value:
                    results["T"] = t_value.upper()
                    break
    
    # Process N stage with node count logic
    for pattern in n_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if pattern == n_patterns[2]:  # Node count pattern
                pos_nodes = int(match.group(1))
                if pos_nodes == 0:
                    results["N"] = "N0"
                elif pos_nodes <= 3:
                    results["N"] = "N1"
                elif pos_nodes <= 9:
                    results["N"] = "N2a"
                else:
                    results["N"] = "N3a"
            else:
                n_value = match.group(1)
                if n_value:
                    results["N"] = n_value.upper()
                    break
    
    # Process M stage with context analysis
    for pattern in m_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            if pattern == m_patterns[3]:  # No metastasis pattern
                if 'without' in match.group() or 'no evidence' in match.group().lower():
                    results["M"] = "M0"
            else:
                m_value = match.group(1) if len(match.groups()) > 0 else None
                if m_value:
                    results["M"] = m_value.upper()
                    break
    
    return results



# Improved fuzzy matching with better confidence scoring
def improved_fuzzy_match(text, tnm_dict, threshold=75):
    matched_stages = {"T": None, "N": None, "M": None, "Details": []}
    
    # First try direct pattern extraction (most accurate)
    direct_results = extract_direct_tnm_patterns(text)
    if direct_results["T"]:
        matched_stages["T"] = direct_results["T"]
    if direct_results["N"]:
        matched_stages["N"] = direct_results["N"]
    if direct_results["M"]:
        matched_stages["M"] = direct_results["M"]
    
    # Then use fuzzy matching as backup/additional validation
    for stage, phrases in tnm_dict.items():
        # Create a window-based approach for better context
        best_matches = []
        for phrase in phrases:
            if len(phrase) < 4:  # Skip very short phrases to reduce false positives
                continue
                
            # Look for exact matches first (higher confidence)
            if phrase.lower() in text.lower():
                best_matches.append({"Phrase": phrase, "Score": 100, "Context": get_context(text, phrase)})
                continue
                
            # Split text into paragraphs for better context matching
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                # Only process paragraph if it might be relevant
                if len(para) > 10:
                    score = fuzz.partial_ratio(phrase.lower(), para.lower())
                    if score > threshold:
                        best_matches.append({
                            "Phrase": phrase, 
                            "Score": score,
                            "Context": get_context(text, phrase)
                        })
        
        if best_matches:
            # Sort matches by score
            best_matches = sorted(best_matches, key=lambda x: x["Score"], reverse=True)
            matched_stages["Details"].append({
                "Stage": stage,
                "Matched Phrases": best_matches[:3]  # Keep top 3 matches
            })
            
            # Only update stage if not already set by direct pattern extraction
            if stage.startswith("T") and not matched_stages["T"]:
                matched_stages["T"] = stage
            elif stage.startswith("N") and not matched_stages["N"]:
                matched_stages["N"] = stage
            elif stage.startswith("M") and not matched_stages["M"]:
                matched_stages["M"] = stage
                
    return matched_stages



# Get context for better understanding matches
def get_context(text, phrase):
    try:
        # Find the approximate position of the phrase
        phrase_pos = text.lower().find(phrase.lower())
        if phrase_pos >= 0:
            # Get surrounding text (50 chars before and after)
            start = max(0, phrase_pos - 50)
            end = min(len(text), phrase_pos + len(phrase) + 50)
            context = text[start:end]
            # Highlight the phrase in the context
            return context.replace(phrase, f"**{phrase}**")
        return ""
    except:
        return ""



# Enhanced highlight matches in PDF with better coloring
def highlight_matches_in_pdf(input_path, output_path, tnm_dict, matched_stages):
    try:
        pdf_doc = fitz.open(input_path)
        
        # Color-coding by stage type
        t_color = (1, 0.6, 0.6)  # light red for T stages
        n_color = (0.6, 0.6, 1)  # light blue for N stages
        m_color = (0.6, 1, 0.6)  # light green for M stages
        
        # Highlight direct TNM mentions first
        for page in pdf_doc:
            page_text = page.get_text()
            
            # Highlight T stage
            if matched_stages["T"]:
                t_pattern = f"{matched_stages['T']}"
                t_instances = page.search_for(t_pattern)
                for inst in t_instances:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=t_color)
                    annot.update()
            
            # Highlight N stage
            if matched_stages["N"]:
                n_pattern = f"{matched_stages['N']}"
                n_instances = page.search_for(n_pattern)
                for inst in n_instances:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=n_color)
                    annot.update()
            
            # Highlight M stage
            if matched_stages["M"]:
                m_pattern = f"{matched_stages['M']}"
                m_instances = page.search_for(m_pattern)
                for inst in m_instances:
                    annot = page.add_highlight_annot(inst)
                    annot.set_colors(stroke=m_color)
                    annot.update()
        
        # Highlight supporting phrases from fuzzy matching
        for detail in matched_stages["Details"]:
            stage = detail["Stage"]
            
            # Choose color based on stage type
            if stage.startswith("T"):
                color = t_color
            elif stage.startswith("N"):
                color = n_color
            elif stage.startswith("M"):
                color = m_color
            else:
                color = (0.7, 0.7, 0.7)  # gray default
                
            for match in detail["Matched Phrases"]:
                phrase = match["Phrase"]
                # Only highlight phrases with high confidence
                if match["Score"] > 85:
                    for page in pdf_doc:
                        instances = page.search_for(phrase)
                        for inst in instances:
                            annot = page.add_highlight_annot(inst)
                            annot.set_colors(stroke=color)
                            annot.update()
        
        # Add a legend annotation to the first page
        first_page = pdf_doc[0]
        legend_text = "TNM Staging Legend:\n"
        legend_text += "- Red: T Stage (Tumor)\n"
        legend_text += "- Blue: N Stage (Nodes)\n"
        legend_text += "- Green: M Stage (Metastasis)"
        
        rect = fitz.Rect(20, 20, 200, 80)
        first_page.add_text_annot(rect, legend_text)
        
        # Save the highlighted PDF
        pdf_doc.save(output_path)
        pdf_doc.close()
        print(f"Highlighted PDF saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error highlighting matches in PDF: {e}")
        return False



# Improved stage determination function
def determine_overall_stage(matched_stages):
    t_stage = matched_stages.get("T")
    n_stage = matched_stages.get("N")
    m_stage = matched_stages.get("M")
    
    # Input validation and normalization
    def normalize_stage(stage):
        if not stage:
            return None
        stage = stage.upper().strip()
        # Remove prefixes if present
        if not any(stage.startswith(x) for x in ["T", "N", "M"]):
            for prefix in ["T", "N", "M"]:
                if prefix in stage:
                    stage = stage[stage.find(prefix):]
                    break
        return stage
    
    t_stage = normalize_stage(t_stage)
    n_stage = normalize_stage(n_stage)
    m_stage = normalize_stage(m_stage)
    
    # Metastatic disease is always Stage IV
    if m_stage == "M1":
        return "Stage IV"
    
    # Enhanced staging logic based on AJCC 8th edition
    if t_stage and n_stage:
        # Stage 0
        if t_stage in ["TIS", "TIS(DCIS)", "TIS(PAGET)"]:
            return "Stage 0"
        
        # Stage IA
        if t_stage in ["T1", "T1MI", "T1A", "T1B", "T1C"] and n_stage == "N0":
            return "Stage IA"
        
        # Stage IB
        if (t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C"] and 
            n_stage in ["N1MI", "N1MIC"]):
            return "Stage IB"
        
        # Stage IIA
        if ((t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C"] and 
             n_stage in ["N1", "N1A", "N1B", "N1C"]) or
            (t_stage == "T2" and n_stage == "N0")):
            return "Stage IIA"
        
        # Stage IIB
        if ((t_stage == "T2" and n_stage in ["N1", "N1A", "N1B", "N1C"]) or
            (t_stage == "T3" and n_stage == "N0")):
            return "Stage IIB"
        
        # Stage IIIA
        if ((t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C", "T2"] and 
             n_stage in ["N2", "N2A", "N2B"]) or
            (t_stage == "T3" and n_stage in ["N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"])):
            return "Stage IIIA"
        
        # Stage IIIB
        if (t_stage in ["T4", "T4A", "T4B", "T4C"] and 
            n_stage in ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"]):
            return "Stage IIIB"
        
        # Stage IIIC
        if n_stage in ["N3", "N3A", "N3B", "N3C"]:
            return "Stage IIIC"
        
        # Special case for inflammatory breast cancer
        if t_stage in ["T4D", "T4"]:
            if n_stage in ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"]:
                return "Stage IIIB"
            elif n_stage in ["N3", "N3A", "N3B", "N3C"]:
                return "Stage IIIC"
    
    return "Stage Not Determined"



def get_extended_context(text, target, window_size=200):
    """Get extended context around a target phrase"""
    if not target:
        return ""
    pos = text.lower().find(target.lower())
    if pos == -1:
        return ""
    start = max(0, pos - window_size)
    end = min(len(text), pos + len(target) + window_size)
    return text[start:end]



def calculate_confidence_score(stages):
    """Calculate confidence score for the extracted stages"""
    score = 0
    total_weight = 0
    
    # Weight for each stage type
    weights = {"T": 0.4, "N": 0.4, "M": 0.2}
    
    for stage_type, stage_value in stages.items():
        if stage_type in weights:
            weight = weights[stage_type]
            total_weight += weight
            
            if stage_value:
                # Add to score based on confidence in the stage
                if stage_type in ["T", "N", "M"]:
                    context_score = 1.0  # Base score for having a value
                    details = next((d for d in stages.get("Details", []) if d["Stage"] == stage_value), None)
                    
                    if details:
                        # Add bonus for high-scoring matches
                        matches = details.get("Matched Phrases", [])
                        if matches:
                            avg_score = sum(m["Score"] for m in matches) / len(matches)
                            context_score = min(1.0, avg_score / 100)
                    
                    score += weight * context_score
    
    # Normalize score to percentage
    final_score = (score / total_weight) * 100 if total_weight > 0 else 0
    return round(final_score, 2)



def create_summary_statistics(df):
    """Create summary statistics from the results"""
    stats = {
        'Total Files Processed': len(df),
        'Successfully Processed': len(df[df['Processing Status'] == 'Success']),
        'Partial Success': len(df[df['Processing Status'].str.contains('Partial', na=False)]),
        'Failed': len(df[df['Processing Status'].str.contains('Failed', na=False)]),
        'Average Confidence Score': df['Confidence Level'].mean(),
        'Files with Complete TNM Staging': len(df[
            (df['T Stage'] != 'Not Found') & 
            (df['N Stage'] != 'Not Found') & 
            (df['M Stage'] != 'Not Found')
        ])
    }
    return pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])



def create_processing_log(df):
    """Create a processing log with details about each file"""
    log_data = df[['File Name', 'Processing Status', 'Processing Date', 'Confidence Level']].copy()
    log_data['Status Details'] = df.apply(
        lambda row: f"TNM Stages Found: {sum(1 for stage in ['T Stage', 'N Stage', 'M Stage'] if row[stage] != 'Not Found')}/3",
        axis=1
    )
    return log_data



def validate_tnm_stages(matched_stages, report_text):
    """Additional validation of TNM stages based on context and relationships"""
    validated = matched_stages.copy()
    
    # Extract context windows around each stage
    t_context = get_extended_context(report_text, validated.get("T", ""))
    n_context = get_extended_context(report_text, validated.get("N", ""))
    m_context = get_extended_context(report_text, validated.get("M", ""))
    
    # Validate T stage
    if validated.get("T"):
        if "breast" not in report_text.lower() and "tumor" not in t_context.lower():
            validated["T"] = None
    
    # Validate N stage
    if validated.get("N"):
        if "node" not in n_context.lower() and "lymph" not in n_context.lower():
            validated["N"] = None
    
    # Validate M stage
    if validated.get("M"):
        if "metasta" not in report_text.lower() and "spread" not in m_context.lower():
            validated["M"] = None
    
    return validated



def calculate_accuracy_metrics(all_data):
    """Calculate accuracy metrics for the extraction process"""
    total_files = len(all_data)
    successful_extractions = 0
    stage_completeness = 0
    confidence_scores = []
    
    for data in all_data:
        # Check if all stages were found
        stages_found = sum(1 for stage in ['T Stage', 'N Stage', 'M Stage'] 
                         if data[stage] not in ['Not Found', 'Error'])
        stage_completeness += stages_found / 3
        
        # Check if overall stage was determined
        if data['Overall Cancer Stage'] not in ['Not Found', 'Error', 'Stage Not Determined']:
            successful_extractions += 1
        
        # Add confidence score
        if isinstance(data.get('Confidence Level'), (int, float)):
            confidence_scores.append(data['Confidence Level'])
    
    metrics = {
        'Total Files Processed': total_files,
        'Successful Stage Determinations': successful_extractions,
        'Overall Success Rate': (successful_extractions / total_files) * 100 if total_files > 0 else 0,
        'Stage Completeness': (stage_completeness / total_files) * 100 if total_files > 0 else 0,
        'Average Confidence Score': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    }
    
    # Print accuracy metrics
    print("\nAccuracy Metrics:")
    print(f"Total Files Processed: {metrics['Total Files Processed']}")
    print(f"Successful Stage Determinations: {metrics['Successful Stage Determinations']}")
    print(f"Overall Success Rate: {metrics['Overall Success Rate']:.2f}%")
    print(f"Stage Completeness: {metrics['Stage Completeness']:.2f}%")
    print(f"Average Confidence Score: {metrics['Average Confidence Score']:.2f}%")
    
    return metrics



# Process multiple PDFs from a folder
def process_folder_with_pdfs(input_folder, output_folder):
    # Get all PDF files in the input folder and subfolders
    pdf_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    if not pdf_files:
        print(f"No PDF files found in folder: {input_folder}")
        return False
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # List to collect data from all PDFs
    all_data = []
    
    # Process each PDF file
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\nProcessing file {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        
        try:
            # Extract text with improved error handling
            report_text = extract_text_from_pdf(pdf_path)
            if not report_text.strip():
                print("Primary text extraction failed. Attempting OCR...")
                report_text = ocr_from_image(pdf_path)
            
            if not report_text.strip():
                raise Exception("No text could be extracted")
            
            # Extract patient information
            age = extract_age_or_dob(report_text)
            print(f"Patient Age: {age if age else 'Not Found'}")
            
            # Extract TNM stages with improved accuracy
            matched_stages = improved_fuzzy_match(report_text, ajcc_tnm, threshold=70)  # Lower threshold for better recall
            
            # Additional validation step
            validated_stages = validate_tnm_stages(matched_stages, report_text)
            
            # Determine overall cancer stage
            overall_stage = determine_overall_stage(validated_stages)
            
            # Print results
            print("\nExtracted TNM Stages:")
            for key, value in validated_stages.items():
                if key != "Details":
                    print(f"  {key} Stage: {value if value else 'Not Found'}")
            print(f"Overall Cancer Stage: {overall_stage}")
            
            # Create highlighted PDF
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            highlight_path = os.path.join(output_folder, f"{base_name}_highlighted.pdf")
            highlight_success = highlight_matches_in_pdf(pdf_path, highlight_path, ajcc_tnm, validated_stages)
            
            # Store results
            result_data = {
                "File Name": os.path.basename(pdf_path),
                "Age": age if age else "Not Found",
                "T Stage": validated_stages["T"] if validated_stages["T"] else "Not Found",
                "N Stage": validated_stages["N"] if validated_stages["N"] else "Not Found",
                "M Stage": validated_stages["M"] if validated_stages["M"] else "Not Found",
                "Overall Cancer Stage": overall_stage,
                "Confidence Level": calculate_confidence_score(validated_stages),
                "Processing Status": "Success" if highlight_success else "Partial Success",
                "Processing Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            all_data.append(result_data)
            
        except Exception as e:
            print(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
            all_data.append({
                "File Name": os.path.basename(pdf_path),
                "Age": "Error",
                "T Stage": "Error",
                "N Stage": "Error",
                "M Stage": "Error",
                "Overall Cancer Stage": "Error",
                "Confidence Level": 0,
                "Processing Status": f"Failed - {str(e)}",
                "Processing Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Save results to Excel with multiple sheets
    try:
        excel_path = os.path.join(output_folder, "TNM_Analysis_Summary.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results sheet
            df_main = pd.DataFrame(all_data)
            df_main.to_excel(writer, sheet_name='TNM Results', index=False)
            
            # Summary statistics sheet
            create_summary_statistics(df_main).to_excel(writer, sheet_name='Summary Statistics')
            
            # Processing log sheet
            create_processing_log(df_main).to_excel(writer, sheet_name='Processing Log')
        
        print(f"\nDetailed results saved to: {excel_path}")
        return True
        
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
        return False



def process_single_pdf(pdf_path, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"\nProcessing file: {os.path.basename(pdf_path)}")
    
    # Extract text from PDF or fallback to OCR
    report_text = extract_text_from_pdf(pdf_path)
    if not report_text.strip():
        print("No text extracted from PDF. Attempting OCR...")
        report_text = ocr_from_image(pdf_path)
    
    if not report_text.strip():
        print(f"WARNING: Could not extract any text from {os.path.basename(pdf_path)}, skipping.")
        return {
            "File Name": os.path.basename(pdf_path),
            "Age": "Not Found",
            "T Stage": "Not Found",
            "N Stage": "Not Found", 
            "M Stage": "Not Found",
            "Overall Cancer Stage": "Not Found",
            "Status": "Failed - No text extracted"
        }
    
    # Extract age or calculate from DOB
    age = extract_age_or_dob(report_text)
    print(f"Patient Age: {age if age else 'Not Found'}")

    # Extract TNM stages
    matched_stages = improved_fuzzy_match(report_text, ajcc_tnm)
    
    print("Matched TNM Stages:")
    for key, value in matched_stages.items():
        if key != "Details":
            print(f"  {key} Stage: {value if value else 'Not Found'}")

    # Determine overall cancer stage
    overall_stage = determine_overall_stage(matched_stages)
    print(f"Overall Cancer Stage: {overall_stage}")

    # Prepare output file path for highlighted PDF
    pdf_name = os.path.basename(pdf_path)
    base_name, _ = os.path.splitext(pdf_name)
    highlight_path = os.path.join(output_folder, f"{base_name}_highlighted.pdf")
    
    # Highlight matches in the PDF
    print(f"Highlighting matches and saving to: {highlight_path}")
    highlight_success = highlight_matches_in_pdf(pdf_path, highlight_path, ajcc_tnm, matched_stages)
    
    # Prepare data for Excel
    data = {
        "File Name": pdf_name,
        "Age": age if age else "Not Found",
        "T Stage": matched_stages["T"] if matched_stages["T"] else "Not Found",
        "N Stage": matched_stages["N"] if matched_stages["N"] else "Not Found", 
        "M Stage": matched_stages["M"] if matched_stages["M"] else "Not Found",
        "Overall Cancer Stage": overall_stage,
        "Status": "Successfully Processed" if highlight_success else "Partial Processing - Highlighting Failed"
    }
    
    # Save data to Excel
    excel_path = os.path.join(output_folder, f"{base_name}_TNM_data.xlsx")
    try:
        df = pd.DataFrame([data])
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"Summary data saved to: {excel_path}")
        return True
    except Exception as e:
        print(f"Error saving summary data to Excel: {e}")
        return False



def main():
    # Replace this with your folder path containing PDFs
    input_folder = r"C:\Users\Md Shoaib\c tutorials course\Downloads\Module VII\Project\Path Report"
    
    # Create output folder in the same directory
    output_folder = os.path.join(input_folder, "TNM_Results")
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing PDFs from: {input_folder}")
    print(f"Results will be saved to: {output_folder}")
    
    # Process all PDFs in the folder
    success = process_folder_with_pdfs(input_folder, output_folder)
    
    if success:
        print("\nSUCCESS: All PDFs processed successfully.")
        print(f"Results saved to: {output_folder}")
    else:
        print("\nERROR: Some files may have failed processing. Check the logs above for details.")



if __name__ == "__main__":
    main()