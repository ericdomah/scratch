import os
import glob
import re
import pandas as pd
import numpy as np
from pypdf import PdfReader

# Standardized regions
REGIONS = ["Lefkoşa", "Gazimağusa", "Girne", "Güzelyurt", "İskele"]

def parse_consumer_counts(pdf_path):
    """Parses Bölgesel Tüketici Adetleri monthly pages."""
    reader = PdfReader(pdf_path)
    records = []
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    for idx, page in enumerate(reader.pages):
        month = months[idx] if idx < 12 else f"Month_{idx+1}"
        text = page.extract_text()
        if not text:
            continue
        lines = text.split('\n')
        
        for line in lines:
            line_str = line.strip()
            tokens = line_str.split()
            if not tokens:
                continue
                
            # Filter tokens to find numeric values
            numbers = []
            desc_tokens = []
            
            # Check if first token is a tariff code (e.g., "02.", "204.", "102")
            first_is_code = False
            if tokens:
                clean_first = tokens[0].replace('.', '').replace(',', '')
                if clean_first.isdigit():
                    first_is_code = True
            
            for i, t in enumerate(tokens):
                if i == 0 and first_is_code:
                    desc_tokens.append(t)
                    continue
                
                clean_t = t.replace('.', '').replace(',', '')
                if clean_t.isdigit():
                    numbers.append(clean_t)
                else:
                    desc_tokens.append(t)
                    
            # In this sheet, there are 5 regional columns + 1 total column
            if len(numbers) >= 5 and any(tar in "".join(desc_tokens).upper() for tar in ["KONUT", "TİCARİ", "ENDÜSTRİ", "TURİZM", "SAVUNMA", "DEVLET", "SU MOTORLARI"]):
                desc = " ".join(desc_tokens)
                records.append({
                    "Month": month,
                    "Tariff_Group": desc,
                    "Lefkosa": int(numbers[0]) if len(numbers) > 0 else 0,
                    "Magusa": int(numbers[1]) if len(numbers) > 1 else 0,
                    "Girne": int(numbers[2]) if len(numbers) > 2 else 0,
                    "Guzelyurt": int(numbers[3]) if len(numbers) > 3 else 0,
                    "Iskele": int(numbers[4]) if len(numbers) > 4 else 0,
                })
    return pd.DataFrame(records)

def parse_daily_load(pdf_path):
    """Parses daily peak and trough consumption MW from System Control reports."""
    reader = PdfReader(pdf_path)
    records = []
    
    current_date = None
    min_mw = None
    max_mw = None
    
    filename = os.path.basename(pdf_path)
    
    # Process text from all pages
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        lines = text.split('\n')
        for line in lines:
            line_str = line.strip()
            
            # Match TARIH 01.01.2023
            date_match = re.search(r'TARIH\s+(\d{2}\.\d{2}\.\d{4})', line_str, re.IGNORECASE)
            if date_match:
                if current_date and (min_mw is not None or max_mw is not None):
                    records.append({
                        "Date": current_date,
                        "Min_MW": min_mw,
                        "Max_MW": max_mw,
                        "Source": filename
                    })
                current_date = date_match.group(1)
                min_mw = None
                max_mw = None
                continue
                
            # Match EN DUŞUK TUKETİM MW 144,39
            min_match = re.search(r'EN\s+DU[ŞS]UK\s+T[ÜU]KET[İI]M\s+MW\s+([\d,.]+)', line_str, re.IGNORECASE)
            if min_match:
                min_mw = float(min_match.group(1).replace(',', '.'))
                continue
                
            # Match EN YÜKSEK TÜKETİM MW 279,74
            max_match = re.search(r'EN\s+Y[ÜU]KSEK\s+T[ÜU]KET[İI]M\s+MW\s+([\d,.]+)', line_str, re.IGNORECASE)
            if max_match:
                max_mw = float(max_match.group(1).replace(',', '.'))
                continue
                
    if current_date and (min_mw is not None or max_mw is not None):
        records.append({
            "Date": current_date,
            "Min_MW": min_mw,
            "Max_MW": max_mw,
            "Source": filename
        })
        
    return pd.DataFrame(records)

def parse_annual_statistic(pdf_path):
    """Parses annual/monthly statistic pages."""
    reader = PdfReader(pdf_path)
    records = []
    filename = os.path.basename(pdf_path)
    year_match = re.search(r'(\d{4})', filename)
    year = int(year_match.group(1)) if year_match else 2024
    
    text = reader.pages[0].extract_text()
    if not text:
        return pd.DataFrame()
        
    lines = text.split('\n')
    for line in lines:
        line_str = line.strip()
        tokens = line_str.split()
        if len(tokens) < 3:
            continue
            
        numbers = []
        desc_tokens = []
        for t in tokens:
            clean_t = t.replace('.', '').replace(',', '').replace('%', '')
            if re.match(r'^\d+([.,]\d+)*$', t) or t.startswith('%'):
                numbers.append(t)
            else:
                desc_tokens.append(t)
                
        desc = " ".join(desc_tokens)
        if len(numbers) >= 5 and any(tar in desc.upper() for tar in ["KONUT", "TİCARİ", "ENDÜSTRİ", "TURİZM", "SAVUNMA", "DEVLET", "TOP", "ÜRETİM", "SANTRAL"]):
            records.append({
                "Year": year,
                "Category": desc,
                "Values": ", ".join(numbers),
                "Source": filename
            })
            
    return pd.DataFrame(records)

def parse_outages(pdf_path):
    """Parses outage reports."""
    reader = PdfReader(pdf_path)
    records = []
    filename = os.path.basename(pdf_path)
    
    # Process up to 10 pages for representative samples of outages
    max_pages = min(10, len(reader.pages))
    
    for page_num in range(max_pages):
        text = reader.pages[page_num].extract_text()
        if not text:
            continue
        lines = text.split('\n')
        for line in lines:
            line_str = line.strip()
            datetime_pattern = r'(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2})\s+(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2})'
            match = re.search(datetime_pattern, line_str)
            if match:
                start_dt = match.group(1)
                end_dt = match.group(2)
                rest = line_str[match.end():].strip()
                
                cause = "Other Interruption"
                for potential_cause in ["Havai Hat Arızası", "Santral Arızası", "Planlı Bakım / Onarım Çalışması", "Planlı Bakım", "Bakım/Onarım Çalışması", "Arıza"]:
                    if potential_cause in rest:
                        cause = potential_cause
                        rest = rest.replace(potential_cause, "").strip()
                        break
                        
                records.append({
                    "Start_Time": start_dt,
                    "End_Time": end_dt,
                    "Cause": cause,
                    "Description": rest[:100],
                    "Source": filename
                })
    return pd.DataFrame(records)

def execute_pipeline():
    download_dir = "c:/Users/User/Downloads"
    pdf_files = glob.glob(os.path.join(download_dir, "*.pdf"))
    
    print("=" * 70)
    print("      GRIDGUARD AI - KIB-TEK EMPIRICAL INGESTION & DATA CLEANING")
    print("=" * 70)
    print(f"Scanning downloads directory: {download_dir}")
    print(f"Total PDF files detected: {len(pdf_files)}")
    
    # Lists for compiled datasets
    consumer_stats_dfs = []
    daily_load_dfs = []
    annual_stat_dfs = []
    outage_dfs = []
    
    for f in pdf_files:
        basename = os.path.basename(f)
        
        # 1. Regional Consumer Counts
        if "bölgesel tüketici adetleri" in basename.lower() or "bolgesel" in basename.lower():
            print(f"  [+] Processing Consumer Counts: {basename}")
            df = parse_consumer_counts(f)
            if not df.empty:
                consumer_stats_dfs.append(df)
                
        # 2. Daily Grid Loads
        elif "tüketim raporu" in basename.lower() or "tuketim raporu" in basename.lower():
            print(f"  [+] Processing Daily Consumption Log: {basename}")
            df = parse_daily_load(f)
            if not df.empty:
                daily_load_dfs.append(df)
                
        # 3. Outage interruptions
        elif "kesinti raporu" in basename.lower():
            print(f"  [+] Processing Outage Records: {basename}")
            df = parse_outages(f)
            if not df.empty:
                outage_dfs.append(df)
                
        # 4. Statistical Sheets
        elif "statistic" in basename.lower() or "istatistik" in basename.lower() or basename.lower().endswith("rapordu.pdf") or "mali_raporu" in basename.lower():
            # Filter standard statistics
            if "kesinti" not in basename.lower() and "tüketim" not in basename.lower():
                print(f"  [+] Processing Statistical Report: {basename}")
                df = parse_annual_statistic(f)
                if not df.empty:
                    annual_stat_dfs.append(df)
                    
    # Save datasets
    output_dir = "c:/Users/User/Downloads/scratch-main/data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 50)
    print("   SAVING CLEANED DATASETS FOR TRAINING")
    print("=" * 50)
    
    # 1. Regional Consumers
    if consumer_stats_dfs:
        consumer_stats = pd.concat(consumer_stats_dfs, ignore_index=True)
        path = os.path.join(output_dir, "kibtek_consumer_stats.csv")
        consumer_stats.to_csv(path, index=False)
        print(f"[SUCCESS] Regional Consumer stats compiled: {path} ({len(consumer_stats)} rows)")
    else:
        print("[WARNING] No regional consumer counts parsed.")
        
    # 2. Daily Grid Loads
    if daily_load_dfs:
        daily_loads = pd.concat(daily_load_dfs, ignore_index=True)
        path = os.path.join(output_dir, "kibtek_daily_load_stats.csv")
        # Ensure dates are unique and sorted
        daily_loads.drop_duplicates(subset=["Date"], keep="first", inplace=True)
        daily_loads.to_csv(path, index=False)
        print(f"[SUCCESS] Daily Load telemetry compiled: {path} ({len(daily_loads)} rows)")
    else:
        print("[WARNING] No daily load curves parsed.")
        
    # 3. Interruption Logs
    if outage_dfs:
        outages = pd.concat(outage_dfs, ignore_index=True)
        path = os.path.join(output_dir, "kibtek_outages_cleaned.csv")
        outages.to_csv(path, index=False)
        print(f"[SUCCESS] Cleaned Outages log compiled: {path} ({len(outages)} rows)")
    else:
        print("[WARNING] No outage reports parsed.")
        
    # 4. Annual / Monthly Statistical Summaries
    if annual_stat_dfs:
        annual_stats = pd.concat(annual_stat_dfs, ignore_index=True)
        path = os.path.join(output_dir, "kibtek_annual_statistics.csv")
        annual_stats.to_csv(path, index=False)
        print(f"[SUCCESS] Annual statistical summaries compiled: {path} ({len(annual_stats)} rows)")
    else:
        print("[WARNING] No statistical sheets parsed.")
        
    print("\n" + "=" * 50)
    print("   CONSOLIDATION PIPELINE COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    execute_pipeline()
