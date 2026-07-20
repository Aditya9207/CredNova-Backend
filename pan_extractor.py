import re

PAN_EXACT_REGEX = re.compile(r'[A-Z]{5}[0-9]{4}[A-Z]')
# Broad pattern to find candidate PAN strings even with OCR misreads
PAN_CANDIDATE_REGEX = re.compile(r'[A-Z0-9]{10}')
# Robust DOB Regex matching DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY even with OCR letter confusion (O/o/I/l/S) in day/month/year
DOB_REGEX = re.compile(r'\b([0-9OolIS]{1,2}\s*[/\-\.\s]\s*[0-9OolIS]{1,2}\s*[/\-\.\s]\s*[0-9OolIS]{4})\b')


def fix_pan_ocr_misreads(raw: str) -> str | None:
    """Correct OCR letter/number confusion in a candidate 10-character PAN string."""
    if len(raw) != 10:
        return None

    num_to_let = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '2': 'Z'}
    let_to_num = {'O': '0', 'o': '0', 'Q': '0', 'I': '1', 'l': '1', 'S': '5', 's': '5', 'B': '8', 'Z': '2'}

    res = []
    # First 5 characters must be UPPERCASE LETTERS (0-4)
    for c in raw[:5]:
        if c.isalpha():
            res.append(c.upper())
        elif c in num_to_let:
            res.append(num_to_let[c])
        else:
            return None

    # Middle 4 characters must be DIGITS (5-8)
    for c in raw[5:9]:
        if c.isdigit():
            res.append(c)
        elif c in let_to_num:
            res.append(let_to_num[c])
        else:
            return None

    # Last character must be UPPERCASE LETTER (9)
    c = raw[9]
    if c.isalpha():
        res.append(c.upper())
    elif c in num_to_let:
        res.append(num_to_let[c])
    else:
        return None

    candidate = "".join(res)
    if PAN_EXACT_REGEX.fullmatch(candidate):
        return candidate
    return None


def extract_pan_fields(full_text: str, lines: list[str]) -> dict:
    result = {"pan_number": None, "name": None, "father_name": None, "dob": None}

    # 1. PAN Number extraction
    # First search for exact PAN regex on word tokens
    tokens = [re.sub(r'[^A-Z0-9]', '', word.upper()) for word in full_text.split()]
    for token in tokens:
        if PAN_EXACT_REGEX.fullmatch(token):
            result["pan_number"] = token
            break

    # If no token match, check concatenated lines or line-by-line candidate tokens
    if not result["pan_number"]:
        for line in lines:
            clean_line = re.sub(r'[^A-Z0-9]', '', line.upper())
            for candidate in PAN_CANDIDATE_REGEX.findall(clean_line):
                fixed = fix_pan_ocr_misreads(candidate)
                if fixed:
                    result["pan_number"] = fixed
                    break
            if result["pan_number"]:
                break

    # Fallback: full text stripped of spaces
    if not result["pan_number"]:
        clean_full = re.sub(r'[^A-Z0-9]', '', full_text.upper())
        exact_match = PAN_EXACT_REGEX.search(clean_full)
        if exact_match:
            result["pan_number"] = exact_match.group()

    # 2. Date of Birth extraction
    dob_match = DOB_REGEX.search(full_text)
    if dob_match:
        dob_str = dob_match.group()
        # Normalize common OCR character confusions in day/month/year
        dob_str = dob_str.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1').replace('S', '5').replace('s', '5')
        parts = re.split(r'[/\-\.\s]+', dob_str.strip())
        if len(parts) >= 3:
            day = re.sub(r'\D', '', parts[0])
            month = re.sub(r'\D', '', parts[1])
            year = re.sub(r'\D', '', parts[2])
            if len(day) == 1:
                day = "0" + day
            if len(month) == 1:
                month = "0" + month
            if len(day) == 2 and len(month) == 2 and len(year) == 4:
                y_num = int(year)
                m_num = int(month)
                d_num = int(day)
                if 1940 <= y_num <= 2012 and 1 <= m_num <= 12 and 1 <= d_num <= 31:
                    result["dob"] = f"{year}-{month}-{day}"

    # Fallback DOB search
    if not result["dob"]:
        alt_match = re.search(r'\b(\d{2})[/\-](\d{2})[/\-](\d{4})\b', full_text)
        if alt_match:
            d, m, y = alt_match.groups()
            if 1940 <= int(y) <= 2012:
                result["dob"] = f"{y}-{m}-{d}"

    # 3. Name / Father's Name extraction
    noise_words = {
        "income", "tax", "department", "govt", "government", "india", "permanent",
        "account", "number", "card", "signature", "father", "name", "date", "birth",
        "dob", "pan", "holder"
    }

    for i, line in enumerate(lines):
        low = line.lower().strip()
        if "father" in low and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate and not any(c.isdigit() for c in candidate):
                result["father_name"] = candidate
        elif "name" in low and "father" not in low and i + 1 < len(lines):
            candidate = lines[i + 1].strip()
            if candidate and not any(c.isdigit() for c in candidate) and len(candidate) > 2:
                result["name"] = candidate

    # Fallback for Name: Look for uppercase lines looking like a name
    if not result["name"]:
        for line in lines:
            line_strip = line.strip()
            if (line_strip.isupper() and 
                re.match(r'^[A-Z\s\.]+$', line_strip) and 
                len(line_strip.split()) >= 2 and
                not any(nw in line_strip.lower() for nw in noise_words)):
                result["name"] = line_strip
                break

    return result

