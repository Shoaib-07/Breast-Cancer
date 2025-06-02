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
    # Tumor (T)
    "TX": ["cannot", "assessed", "inconclusive", "results", "unknown", "tumor", "size", "measurable", "unavailable"],
    "T0": ["no", "tumor", "found", "absence", "evidence", "disease", "detectable", "lesions", "malignancy"],
    "Tis (DCIS)": ["ductal", "carcinoma", "in-situ", "non-invasive", "localized", "microcalcifications", "early", "stage"],
    "Tis (Paget)": ["Paget", "disease", "breast", "nipple", "skin", "lesions", "infiltration", "in-situ"],
    "T1mi": ["microinvasion", "small", "0.1", "cm", "smaller", "microinvasive", "minimal", "invasion"],
    "T1a": ["tumor", "0.1", "0.2", "0.3", "0.4", "0.5", "cm", "localized", "early", "stage"],
    "T1b": ["tumor", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "cm", "small", "localized", "early", "invasive"],
    "T1c": ["tumor", "1", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2", "cm", "localized", "confined"],
    "T2": ["tumor", "2", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7", "2.8", "2.9", "3", "3.0", "3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "4", "4.0", "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "5", "5.0", "cm", "invasive", "localized", "moderate", "size", "extension"],
    "T3": ["tumor", "larger", "5", "6", "7", "8", "9", "10", "5.0", "6.0", "7.0", "8.0", "9.0", "10.0", "cm", "advanced", "disease", "invasive", "large", "tumor", "extensive", "infiltration"],
    "T4a": ["tumor", "spread", "chest", "wall", "locally", "advanced", "direct", "invasion", "thoracic"],
    "T4b": ["tumor", "spread", "skin", "localized", "invasion", "inflammatory", "cancer", "edema", "ulceration"],
    "T4c": ["tumor", "spread", "chest", "wall", "skin", "extensive", "disease", "local", "diffuse", "infiltration"],
    "T4d": ["inflammatory", "breast", "cancer", "aggressive", "redness", "swelling", "rapid", "growth", "painful"],

    # Nodes (N)
    "N0": ["no", "regional", "lymph", "node", "involvement", "metastasis", "not", "involved", "negative", "clear"],
    "N1mi": ["micrometastasis", "≤", "0.2", "cm", "micrometastatic", "micro", "spread", "subclinical"],
    "N1a": ["1", "2", "3", "axillary", "lymph", "nodes", "involved", "positive", "metastasis", "detection", "palpable"],
    "N1b": ["internal", "mammary", "lymph", "nodes", "involved", "positive", "micrometastasis", "regional", "spread"],
    "N1c": ["both", "axillary", "internal", "mammary", "lymph", "nodes", "involved", "metastatic", "positive"],
    "N2a": ["4", "5", "6", "7", "8", "9", "axillary", "lymph", "nodes", "involved", "positive", "metastatic", "enlarged", "palpable"],
    "N2b": ["10", "or", "more", "11", "12", "13", "14", "15", "axillary", "lymph", "nodes", "involved", "positive", "significant", "enlarged", "metastatic"],
    "N3a": ["lymph", "nodes", "above", "clavicle", "involved", "positive", "supraclavicular", "metastatic", "node", "enlarged"],
    "N3b": ["infraclavicular", "lymph", "nodes", "involved", "positive", "enlarged", "metastasis", "axillary", "invasive"],
    "N3c": ["supraclavicular", "lymph", "nodes", "involved", "positive", "regional", "extended", "enlarged"],

    # Metastasis (M)
    "M0": ["no", "distant", "metastasis", "identified", "absent", "spread", "absence", "metastatic"],
    "M1": ["extracapsular", "extension", "deposit", "distant", "metastasis", "lymphovascular", "invasion", "microcalcification", "spread", "systemic", "nodal", "involvement"]
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
def extract_direct_tnm_patterns(report_text):
    """
    Enhanced TNM extraction with precise pattern matching
    Returns dict with keys 'T', 'N', 'M' and their values
    """
    # Normalize text for case-insensitive matching
    text = report_text.upper().replace(" ", "").replace("-", "")
    
    # Initialize result dict
    result = {'T': None, 'N': None, 'M': None}
    
    # Enhanced regex patterns for each category
    patterns = {
        'T': [
            r'\bT([0-4X])([A-DI]*(?:\(.*?\))*)\b',  # Standard T patterns
            r'\bT(IS|ISDCIS|ISPAGET)\b',            # In situ variants
            r'\bT1(MI|MICROINVASIVE)\b',            # Microinvasion
            r'(?<![A-Z])T([0-4])(?![A-Z])'          # Standalone T numbers
        ],
        'N': [
            r'\bN([0-3X])([A-CI]*)\b',              # Standard N patterns
            r'\bN([0-3])(MI|MICRO)\b',              # Micrometastasis
            r'(?<![A-Z])N([0-3])(?![A-Z])'          # Standalone N numbers
        ],
        'M': [
            r'\bM([01X])\b',                         # Standard M patterns
            r'(?<![A-Z])M([01])(?![A-Z])',          # Standalone M numbers
            r'METASTASIS.*?(YES|NO|PRESENT|ABSENT)'  # Metastasis mentions
        ]
    }
    
    # Search for each category
    for category in ['T', 'N', 'M']:
        for pattern in patterns[category]:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Clean and normalize the matched value
                value = ''.join([g for g in match.groups() if g])
                value = value.replace("(", "").replace(")", "")
                
                # Special handling for each category
                if category == 'T':
                    if 'ISDCIS' in value:
                        value = 'is (DCIS)'
                    elif 'ISPAGET' in value:
                        value = 'is (PAGET)'
                    elif 'MI' in value or 'MICROINVASIVE' in value:
                        value = '1mi'
                
                if category == 'N' and ('MI' in value or 'MICRO' in value):
                    value = '1mi'
                
                if category == 'M':
                    if value in ['PRESENT', 'YES']:
                        value = '1'
                    elif value in ['ABSENT', 'NO']:
                        value = '0'
                
                # Only update if we found a better match (longer/more specific)
                if not result[category] or len(value) > len(result[category]):
                    result[category] = value.upper()
    
    # Post-processing validation
    for category in result:
        if result[category]:
            # Remove invalid characters
            result[category] = re.sub(r'[^0-9XISMIA-Z]', '', result[category])
            # Convert numbers to consistent format
            result[category] = re.sub(r'(\d)([A-Z])', r'\1\2', result[category])
    
    # Handle special cases
    if result['M'] and 'METASTASIS' in report_text.upper() and 'PRESENT' in report_text.upper():
        result['M'] = '1'
    if result['M'] and 'METASTASIS' in report_text.upper() and 'ABSENT' in report_text.upper():
        result['M'] = '0'
    
    return result



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
    """Determine overall cancer stage from TNM using AJCC 8th edition guidelines"""
    t_stage = matched_stages.get("T")
    n_stage = matched_stages.get("N")
    m_stage = matched_stages.get("M")

    # Convert to uppercase for case-insensitive comparison
    t_stage = t_stage.upper() if t_stage else None
    n_stage = n_stage.upper() if n_stage else None
    m_stage = m_stage.upper() if m_stage else None

    # Handle metastatic cases first (M1 is always Stage IV)
    if m_stage == "M1":
        return "Stage IV (Metastatic)"
    
    # Handle cases where we can't determine stage
    if not t_stage or not n_stage:
        return "Stage Not Determined (Insufficient TNM data)"

    # STAGE 0 (Tis N0 M0)
    if t_stage in ["TIS", "TIS (DCIS)", "TIS (PAGET)"] and n_stage == "N0":
        return "Stage 0 (In Situ)"
    
    # STAGE IA (T1 N0 M0)
    if t_stage in ["T1", "T1MI", "T1A", "T1B", "T1C"] and n_stage == "N0":
        return "Stage IA"
    
    # STAGE IB (T0/T1 N1mi M0)
    if (t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C"] and 
        n_stage in ["N1MI", "N1MIC"]):
        return "Stage IB (Microscopic nodal involvement)"
    
    # STAGE IIA (T0/T1 N1 M0 OR T2 N0 M0)
    if ((t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C"] and 
         n_stage in ["N1", "N1A", "N1B", "N1C"]) or
        (t_stage == "T2" and n_stage == "N0")):
        return "Stage IIA"
    
    # STAGE IIB (T2 N1 M0 OR T3 N0 M0)
    if ((t_stage == "T2" and n_stage in ["N1", "N1A", "N1B", "N1C"]) or
        (t_stage == "T3" and n_stage == "N0")):
        return "Stage IIB"
    
    # STAGE IIIA (T0-T2 N2 M0 OR T3 N1-N2 M0)
    if ((t_stage in ["T0", "T1", "T1MI", "T1A", "T1B", "T1C", "T2"] and 
         n_stage in ["N2", "N2A", "N2B"]) or
        (t_stage == "T3" and n_stage in ["N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"])):
        return "Stage IIIA"
    
    # STAGE IIIB (T4 N0-N2 M0)
    if (t_stage in ["T4", "T4A", "T4B", "T4C"] and 
        n_stage in ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"]):
        return "Stage IIIB (Locally Advanced)"
    
    # STAGE IIIC (Any T N3 M0)
    if n_stage in ["N3", "N3A", "N3B", "N3C"]:
        return "Stage IIIC (Extensive Nodal Involvement)"
    
    # Inflammatory breast cancer (T4d)
    if t_stage in ["T4D", "T4"]:
        if n_stage in ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B"]:
            return "Stage IIIB (Inflammatory)"
        elif n_stage in ["N3", "N3A", "N3B", "N3C"]:
            return "Stage IIIC (Inflammatory with N3)"
    
    # If we get here, the stage couldn't be determined
    return "Stage Not Determined (Unmatched TNM Combination)"



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
                "Overall Cancer Stage": overall_stage
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
                "Confidence Level": 0
            })
    
    # Save results to Excel with multiple sheets
    try:
        excel_path = os.path.join(output_folder, "TNM_Analysis_Summary.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main results sheet
            df_main = pd.DataFrame(all_data)
            df_main.to_excel(writer, sheet_name='TNM Results', index=False)
        
        print(f"\nDetailed results saved to: {excel_path}")
        return True
        
    except Exception as e:
        print(f"Error saving results to Excel: {e}")
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
