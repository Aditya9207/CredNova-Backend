import re

PAN_REGEX = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
# Robust DOB Regex matching DD/MM/YYYY with potential spaces and OCR substitutions (O/o/I/l/S)
DOB_REGEX = re.compile(r'\b(\d{1,2}\s*[/\-]\s*\d{1,2}\s*[/\-]\s*[0-9OolIS]{4})\b')


def extract_pan_fields(full_text: str, lines: list[str]) -> dict:
    result = {"pan_number": None, "name": None, "father_name": None, "dob": None}

    # PAN number — remove spaces since OCR sometimes splits characters
    pan_match = PAN_REGEX.search(full_text.replace(" ", ""))
    if pan_match:
        result["pan_number"] = pan_match.group()

    dob_match = DOB_REGEX.search(full_text)
    if dob_match:
        dob_str = dob_match.group()
        # Normalize common OCR character confusions
        dob_str = dob_str.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('S', '5')
        # Split by slash or dash
        parts = re.split(r'[/\-]', dob_str)
        if len(parts) == 3:
            day = re.sub(r'\D', '', parts[0])
            month = re.sub(r'\D', '', parts[1])
            year = re.sub(r'\D', '', parts[2])
            if len(day) == 1:
                day = "0" + day
            if len(month) == 1:
                month = "0" + month
            if len(day) == 2 and len(month) == 2 and len(year) == 4:
                result["dob"] = f"{year}-{month}-{day}"

    # Name/Father's Name are typically the line right after their label
    for i, line in enumerate(lines):
        low = line.lower().strip()
        if "father" in low and i + 1 < len(lines):
            result["father_name"] = lines[i + 1].strip()
        elif "name" in low and i + 1 < len(lines):
            # Check if this contains name label but not father
            candidate = lines[i + 1].strip()
            if candidate and not any(c.isdigit() for c in candidate):
                result["name"] = candidate

    # Fallback for Name: Look for first uppercase line that looks like a name and isn't a header
    if not result["name"]:
        noise_words = {"income", "tax", "department", "govt", "government", "india", "permanent", "account", "number", "card", "signature"}
        for line in lines:
            line_strip = line.strip()
            if (line_strip.isupper() and 
                re.match(r'^[A-Z\s\.]+$', line_strip) and 
                len(line_strip.split()) >= 2 and
                not any(nw in line_strip.lower() for nw in noise_words)):
                result["name"] = line_strip
                break

    return result
