import os
import pdfplumber
import pandas as pd
from flask import Flask, render_template, request, jsonify
import re
from collections import defaultdict
import json
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class PDFAnalyzer:
    """
    Advanced PDF analyzer using pdfplumber best practices
    """
    
    def __init__(self):
        self.table_settings = {
            'vertical_strategy': 'text',  # Back to text strategy with better settings
            'horizontal_strategy': 'text',
            'min_words_vertical': 1,
            'min_words_horizontal': 1,
            'text_tolerance': 8,  # Increased tolerance for better grouping
            'text_x_tolerance': 8,
            'text_y_tolerance': 8,
            'join_tolerance': 8,
            'join_x_tolerance': 8,
            'join_y_tolerance': 8,
            'snap_tolerance': 8,
            'snap_x_tolerance': 8,
            'snap_y_tolerance': 8
        }
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main analysis function using pdfplumber best practices
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document_info = {
                    'filename': os.path.basename(pdf_path),
                    'total_pages': len(pdf.pages),
                    'pages': [],
                    'summary': {}
                }
                
                for page_num, page in enumerate(pdf.pages, 1):
                    logger.info(f"Processing page {page_num}")
                    page_analysis = self.analyze_page(page, page_num)
                    document_info['pages'].append(page_analysis)
                
                # Create document summary
                document_info['summary'] = self.create_document_summary(document_info['pages'])
                return document_info
                
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            return {'error': str(e), 'filename': os.path.basename(pdf_path)}
    
    def analyze_page(self, page, page_num: int) -> Dict[str, Any]:
        """
        Analyze a single page using pdfplumber best practices with advanced segmentation
        """
        logger.info(f"Processing page {page_num}")
        
        # Step 1: Crop page margins for better content extraction
        cropped_page = self.crop_page_margins(page)
        
        # Step 2: Segment the page into metadata, body, and footer regions
        meta_region, body_region, footer_region = self.segment_page(cropped_page)
        
        # Step 3: Extract metadata (key-value pairs)
        metadata = self.extract_metadata(meta_region)
        
        # Step 4: Extract the main line-items table with precise column boundaries
        line_items_table = self.extract_line_items_table(body_region)
        
        # Step 5: Extract totals and summary information
        totals = self.extract_totals(footer_region)
        
        # Step 6: Extract additional tables and content (legacy method for compatibility)
        text_objects = self.extract_text_content(cropped_page)
        additional_tables = self.extract_tables(cropped_page)
        sections = self.identify_sections(text_objects, cropped_page)
        parsed_sections = self.parse_sections(sections, cropped_page)
        
        # Step 7: Validate and clean the extracted data
        validated_data = self.validate_and_clean_extracted_data(metadata, line_items_table, totals)
        
        # Step 8: Create debug information
        debug_info = {
            'original_chars': len(page.chars) if page.chars else 0,
            'cropped_chars': len(cropped_page.chars) if cropped_page.chars else 0,
            'sections_found': len(sections),
            'tables_found': len(additional_tables) + (1 if line_items_table else 0),
            'metadata_fields': len(metadata),
            'line_items_count': len(line_items_table) if line_items_table else 0,
            'totals_fields': len(totals)
        }
        
        return {
            'page_num': page_num,
            'dimensions': {'width': page.width, 'height': page.height},
            'metadata': metadata,
            'line_items_table': line_items_table,
            'totals': totals,
            'text_content': text_objects,
            'tables': additional_tables,
            'sections': parsed_sections,
            'debug_info': debug_info,
            'validated_data': validated_data
        }
    
    def crop_page_margins(self, page) -> Any:
        """
        Crop page margins using pdfplumber best practices
        """
        # Get page dimensions
        width, height = page.width, page.height
        
        # Define margin percentages (adjust based on your PDFs)
        margin_percent = 0.05  # 5% margin
        
        # Calculate crop box
        margin_x = width * margin_percent
        margin_y = height * margin_percent
        
        crop_box = (
            margin_x,      # x0 (left)
            margin_y,      # y0 (top)
            width - margin_x,  # x1 (right)
            height - margin_y  # y1 (bottom)
        )
        
        return page.crop(crop_box)
    
    def _extract_text_content_old(self, page) -> List[Dict[str, Any]]:
        """
        Extract text content using pdfplumber's advanced features
        """
        text_objects = []
        
        # Method 1: Use extract_words() with better settings to avoid character splitting
        try:
            words = page.extract_words(
                x_tolerance=5,  # Increased tolerance to group characters better
                y_tolerance=5,
                keep_blank_chars=False,
                use_text_flow=True
            )
            
            # Group words into lines
            lines = self.group_words_into_lines(words)
            
            for line in lines:
                # Clean up text by removing excessive spaces and fixing character spacing
                line_text = ' '.join([word['text'] for word in line])
                line_text = self.clean_text(line_text)
                
                if line_text.strip():
                    text_objects.append({
                        'text': line_text,
                        'x0': min(word['x0'] for word in line),
                        'x1': max(word['x1'] for word in line),
                        'top': min(word['top'] for word in line),
                        'bottom': max(word['bottom'] for word in line),
                        'font_size': line[0].get('size', 12) if line else 12,
                        'words': line
                    })
                
        except Exception as e:
            logger.warning(f"Word extraction failed: {e}")
            
            # Fallback: Use extract_text() with layout preservation
            try:
                raw_text = page.extract_text(layout=True)
                if raw_text:
                    lines = raw_text.split('\n')
                    for i, line in enumerate(lines):
                        line_text = self.clean_text(line.strip())
                        if line_text:
                            text_objects.append({
                                'text': line_text,
                                'x0': 0,
                                'x1': page.width,
                                'top': i * 20,
                                'bottom': (i + 1) * 20,
                                'font_size': 12,
                                'words': []
                            })
            except Exception as e2:
                logger.error(f"Text extraction fallback failed: {e2}")
        
        return text_objects
    
    def _group_words_into_lines_old(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Group words into lines based on vertical position
        """
        if not words:
            return []
        
        # Sort words by vertical position, then horizontal
        sorted_words = sorted(words, key=lambda w: (w['top'], w['x0']))
        
        lines = []
        current_line = []
        last_y = None
        tolerance = 5  # pixels
        
        for word in sorted_words:
            if last_y is None or abs(word['top'] - last_y) <= tolerance:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
            last_y = word['top']
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def extract_tables(self, page) -> List[Dict[str, Any]]:
        """
        Extract tables using pdfplumber's advanced table detection with intelligent filtering
        """
        tables = []
        
        try:
            # Method 1: Try pdfplumber's built-in table extraction
            extracted_tables = page.extract_tables(table_settings=self.table_settings)
            logger.info(f"Raw extracted tables count: {len(extracted_tables)}")
            
            for i, table in enumerate(extracted_tables):
                if table and len(table) > 1:  # At least header + 1 row
                    # Clean table data
                    cleaned_table = []
                    for row in table:
                        cleaned_row = [self.clean_text(str(cell)) if cell else "" for cell in row]
                        if any(cell for cell in cleaned_row):  # Skip completely empty rows
                            cleaned_table.append(cleaned_row)
                    
                    if cleaned_table and len(cleaned_table) > 1:
                        logger.info(f"Table {i} headers: {cleaned_table[0]}")
                        logger.info(f"Table {i} rows: {len(cleaned_table)}")
                        
                        # Intelligent table validation
                        if self.is_valid_table(cleaned_table):
                            logger.info(f"Table {i} VALIDATED as table")
                            # Create DataFrame and convert to serializable format
                            # Ensure unique column names
                            headers = cleaned_table[0]
                            unique_headers = []
                            seen_headers = set()
                            
                            for header in headers:
                                original_header = header
                                counter = 1
                                while header in seen_headers:
                                    header = f"{original_header}_{counter}"
                                    counter += 1
                                seen_headers.add(header)
                                unique_headers.append(header)
                            
                            df = pd.DataFrame(cleaned_table[1:], columns=unique_headers)
                            tables.append({
                                'table_index': i,
                                'dataframe_dict': df.to_dict('records'),  # Convert to dict instead of DataFrame
                                'columns': df.columns.tolist(),  # Add column names
                                'html': df.to_html(classes='table table-bordered table-striped table-sm', index=False),
                                'raw_data': cleaned_table,
                                'rows': len(cleaned_table),
                                'columns_count': len(cleaned_table[0]) if cleaned_table else 0
                            })
                        else:
                            logger.info(f"Table {i} REJECTED as non-table")
            
            # Method 2: If no tables found, try custom table detection from text content
            if not tables:
                logger.info("No tables found with pdfplumber, trying custom detection...")
                custom_tables = self.detect_tables_from_text(page)
                tables.extend(custom_tables)
                        
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        
        logger.info(f"Final tables count: {len(tables)}")
        return tables
    
    def detect_tables_from_text(self, page) -> List[Dict[str, Any]]:
        """
        Custom table detection based on text content patterns
        """
        tables = []
        
        try:
            # Extract all text objects
            text_objects = self.extract_text_content(page)
            
            # Look for automotive repair table patterns - more specific to what we see in the image
            automotive_table_patterns = [
                r'LINE\s*#.*DESCRIPTION.*OPERATION.*UNITS.*TOTAL',
                r'LABOR.*PART.*LINE.*DESCRIPTION.*OPERATION',
                r'LINE.*DESCRIPTION.*OPERATION.*TYPE.*UNITS',
                r'SUPPLEMENT.*LINE.*DESCRIPTION.*OPERATION',
                r'LINE.*DESCRIPTION.*R\s*&\s*[A-Z].*UNITS',
                r'S\d+\s+\d+',  # Supplement line numbers like "S1 7"
                r'^\d+\s+R\s*&\s*[A-Z]',  # Line numbers with R&I operations
                r'FRT\s+BUMPER',  # Common repair descriptions
                r'REMOVE\s*/\s*REPLACE',  # Common operations
                r'FRT\s+COMBINATION\s+LAMP'  # More specific descriptions
            ]
            
            # Group text objects by vertical position to find potential table rows
            sorted_text = sorted(text_objects, key=lambda x: x['top'])
            
            # Look for patterns that indicate table headers or repair line items
            for i, text_obj in enumerate(sorted_text):
                text = text_obj['text'].upper()
                
                # Check if this looks like a table header or repair line item
                if any(re.search(pattern, text, re.IGNORECASE) for pattern in automotive_table_patterns):
                    logger.info(f"Found potential table content: {text}")
                    
                    # Try to extract table starting from this position
                    table_data = self.extract_table_from_position(sorted_text, i)
                    if table_data and len(table_data) > 1:
                        # Validate the extracted table
                        if self.is_valid_table(table_data):
                            logger.info(f"Custom table detected with {len(table_data)} rows")
                            
                            # Create DataFrame
                            headers = table_data[0]
                            unique_headers = []
                            seen_headers = set()
                            
                            for header in headers:
                                original_header = header
                                counter = 1
                                while header in seen_headers:
                                    header = f"{original_header}_{counter}"
                                    counter += 1
                                seen_headers.add(header)
                                unique_headers.append(header)
                            
                            df = pd.DataFrame(table_data[1:], columns=unique_headers)
                            tables.append({
                                'table_index': len(tables),
                                'dataframe_dict': df.to_dict('records'),
                                'columns': df.columns.tolist(),
                                'html': df.to_html(classes='table table-bordered table-striped table-sm', index=False),
                                'raw_data': table_data,
                                'rows': len(table_data),
                                'columns_count': len(table_data[0]) if table_data else 0,
                                'detection_method': 'custom_text_analysis'
                            })
                            break  # Found one table, stop looking
                        
        except Exception as e:
            logger.warning(f"Custom table detection failed: {e}")
        
        return tables
    
    def extract_table_from_position(self, text_objects: List[Dict], start_index: int) -> List[List[str]]:
        """
        Extract table data starting from a specific text object position
        """
        table_data = []
        max_rows = 50  # Limit to prevent infinite loops
        row_count = 0
        
        # Start from the potential header
        current_index = start_index
        
        # First, try to find a proper header row
        header_found = False
        header_row = None
        
        # Look for a header row in the next few lines
        for i in range(start_index, min(start_index + 5, len(text_objects))):
            text = text_objects[i]['text'].strip()
            if self.looks_like_table_header(text):
                header_row = self.create_header_row(text)
                header_found = True
                current_index = i + 1
                break
        
        # If no header found, create a default header based on the first data row
        if not header_found:
            # Look ahead to find the first data row to determine column structure
            for i in range(start_index, min(start_index + 10, len(text_objects))):
                text = text_objects[i]['text'].strip()
                if self.looks_like_repair_line(text):
                    columns = self.split_text_into_columns(text)
                    if len(columns) > 1:
                        header_row = self.create_default_header(len(columns))
                        current_index = i
                        break
        
        if header_row:
            table_data.append(header_row)
        
        # Now extract data rows
        while current_index < len(text_objects) and row_count < max_rows:
            text_obj = text_objects[current_index]
            text = text_obj['text'].strip()
            
            if not text:
                current_index += 1
                continue
            
            # Check if this looks like a repair line item
            if self.looks_like_repair_line(text):
                # Split text into potential columns
                columns = self.split_text_into_columns(text)
                
                if columns and len(columns) > 1:
                    table_data.append(columns)
                    row_count += 1
            
            current_index += 1
            
            # Stop if we hit a section that doesn't look like table data
            if self.is_end_of_table(text):
                break
        
        return table_data
    
    def looks_like_table_header(self, text: str) -> bool:
        """
        Check if text looks like a table header
        """
        header_patterns = [
            r'LINE\s*#.*DESCRIPTION.*OPERATION',
            r'LABOR.*PART.*LINE.*DESCRIPTION',
            r'LINE.*DESCRIPTION.*OPERATION.*TYPE',
            r'SUPPLEMENT.*LINE.*DESCRIPTION',
            r'LINE.*DESCRIPTION.*R\s*&\s*[A-Z].*UNITS'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in header_patterns)
    
    def looks_like_repair_line(self, text: str) -> bool:
        """
        Check if text looks like a repair line item
        """
        repair_patterns = [
            r'^\d+\s+R\s*&\s*[A-Z]',  # Line numbers with R&I operations
            r'^S\d+\s+\d+',  # Supplement line numbers
            r'FRT\s+BUMPER',  # Common repair descriptions
            r'REMOVE\s*/\s*REPLACE',  # Common operations
            r'FRT\s+COMBINATION\s+LAMP',  # More specific descriptions
            r'\$\d+\.?\d*',  # Contains price
            r'\d+\.?\d*\s*(hrs?|hours?)',  # Contains hours
            r'NEW|EXISTING|REMANUFACTURED'  # Contains part type
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in repair_patterns)
    
    def create_header_row(self, text: str) -> List[str]:
        """
        Create a header row from text
        """
        # Try to extract meaningful headers from the text
        if 'LINE' in text.upper() and 'DESCRIPTION' in text.upper():
            return ['Line #', 'Description', 'Operation', 'Type', 'Units', 'Rate', 'Type', 'Number', 'Qty', 'Total Price', 'Tax']
        elif 'LABOR' in text.upper() and 'PART' in text.upper():
            return ['Labor/Part', 'Line #', 'Description', 'Operation', 'Type', 'Units', 'Rate', 'Type', 'Number', 'Qty', 'Total Price', 'Tax']
        else:
            # Default headers based on common automotive repair table structure
            return ['Line #', 'Description', 'Operation', 'Type', 'Units', 'Rate', 'Type', 'Number', 'Qty', 'Total Price', 'Tax']
    
    def create_default_header(self, column_count: int) -> List[str]:
        """
        Create a default header based on column count
        """
        default_headers = ['Line #', 'Description', 'Operation', 'Type', 'Units', 'Rate', 'Type', 'Number', 'Qty', 'Total Price', 'Tax']
        
        if column_count <= len(default_headers):
            return default_headers[:column_count]
        else:
            # Extend with generic column names
            extended_headers = default_headers.copy()
            for i in range(len(default_headers), column_count):
                extended_headers.append(f'Column {i+1}')
            return extended_headers
    
    def split_text_into_columns(self, text: str) -> List[str]:
        """
        Split text into columns based on automotive repair table patterns
        """
        # Clean the text first
        text = self.clean_text(text)
        
        # Pattern 1: Complex supplement line with all details
        # Example: "S1 7 Frt Bumper Cover Remove / Replace Body INC# 4.0 New 51 11 7 389 903 1 $1,087.47 Yes"
        pattern1 = r'^(S\d+)\s+(\d+)\s+(.+?)\s+(Remove\s*/\s*Replace|R\s*&\s*[A-Z])\s+(Body|Refinish|Mechanical)\s+([A-Z#]+)\s+(\d+\.?\d*)\s+(New|Existing|Remanufactured)\s+([\d\s]+)\s+(\d+)\s+(\$[\d,]+\.?\d*)\s+(Yes|No)$'
        match1 = re.match(pattern1, text, re.IGNORECASE)
        if match1:
            return [
                match1.group(1),  # Supplement
                match1.group(2),  # Line number
                match1.group(3),  # Description
                match1.group(4),  # Operation
                match1.group(5),  # Type
                match1.group(6),  # Units
                match1.group(7),  # Rate
                match1.group(8),  # New/Existing
                match1.group(9),  # Part number
                match1.group(10), # Qty
                match1.group(11), # Price
                match1.group(12)  # Tax
            ]
        
        # Pattern 2: Simple line with R&I operation
        # Example: "1 R&I Description 3.8 3.8 Existing"
        pattern2 = r'^(\d+)\s+(R\s*&\s*[A-Z])\s+(.+?)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(Existing|New)$'
        match2 = re.match(pattern2, text, re.IGNORECASE)
        if match2:
            return [match2.group(1), match2.group(2), match2.group(3), match2.group(4), match2.group(5), match2.group(6)]
        
        # Pattern 3: Simple line number + description
        # Example: "4 R&I Bumper Cover 1.2"
        pattern3 = r'^(\d+)\s+(R\s*&\s*[A-Z])\s+(.+?)\s+(\d+\.?\d*)$'
        match3 = re.match(pattern3, text, re.IGNORECASE)
        if match3:
            return [match3.group(1), match3.group(2), match3.group(3), match3.group(4)]
        
        # Pattern 4: Very simple line number + description
        # Example: "10 R&I Wheel Opening Mldg Japan Built 0.3"
        pattern4 = r'^(\d+)\s+(.+?)\s+(\d+\.?\d*)$'
        match4 = re.match(pattern4, text)
        if match4:
            return [match4.group(1), match4.group(2), match4.group(3)]
        
        # Pattern 5: Simple line number + description
        pattern5 = r'^(\d+)\s+(.+)$'
        match5 = re.match(pattern5, text)
        if match5:
            return [match5.group(1), match5.group(2)]
        
        # If no pattern matches, try to split by common delimiters
        # Look for multiple spaces or tabs that might indicate columns
        if re.search(r'\s{3,}', text):  # 3 or more spaces
            return [col.strip() for col in re.split(r'\s{3,}', text) if col.strip()]
        
        # Single column
        return [text]
    
    def is_end_of_table(self, text: str) -> bool:
        """
        Determine if text indicates the end of a table
        """
        end_indicators = [
            'TOTAL', 'SUBTOTAL', 'GRAND TOTAL', 'ESTIMATE TOTAL',
            'PLEASE', 'CONTACT', 'REQUIRED', 'AUTHORIZATION',
            'COMMITTED', 'VERSION', 'PRINTED', 'COPYRIGHT',
            'PAGE', 'OF'
        ]
        
        text_upper = text.upper()
        return any(indicator in text_upper for indicator in end_indicators)
    
    def is_valid_table(self, table_data: List[List[str]]) -> bool:
        """
        Determine if the extracted data is actually a table vs. structured text
        """
        if not table_data or len(table_data) < 2:
            logger.info("Table validation failed: insufficient data")
            return False
        
        headers = table_data[0]
        data_rows = table_data[1:]
        
        # Check 1: Headers should be meaningful and not too long
        if len(headers) < 2 or len(headers) > 15:  # Reduced max columns to be more strict
            logger.info(f"Table validation failed: header count {len(headers)} not in range [2, 15]")
            return False
        
        # Check 2: Headers should not be too long (likely text blocks)
        for header in headers:
            if len(header) > 100:  # Increased limit for longer headers like "Operation Type"
                logger.info(f"Table validation failed: header too long: {header}")
                return False
        
        # Check 3: Data rows should have consistent structure
        if not data_rows:
            logger.info("Table validation failed: no data rows")
            return False
        
        expected_cols = len(headers)
        consistent_structure = True
        data_content_count = 0
        
        for row in data_rows:
            if len(row) != expected_cols:
                consistent_structure = False
                break
            
            # Count rows with actual data (not just empty or single words)
            non_empty_cells = sum(1 for cell in row if cell.strip() and len(cell.strip()) > 1)
            if non_empty_cells >= 2:  # At least 2 cells with meaningful content
                data_content_count += 1
        
        if not consistent_structure or data_content_count < 1:
            logger.info(f"Table validation failed: structure inconsistent or no content. Consistent: {consistent_structure}, content_count: {data_content_count}")
            return False
        
        # Check 4: Look for automotive repair table patterns - this is the key check
        header_text = ' '.join(headers).upper()
        automotive_patterns = [
            'LINE', 'DESCRIPTION', 'OPERATION', 'UNITS', 'RATE', 'TOTAL',
            'PARTS', 'LABOR', 'PRICE', 'QUANTITY', 'AMOUNT', 'TAX',
            'ESTIMATE', 'SUPPLEMENT', 'TYPE', 'NUMBER', 'CEG', 'QTY',
            'S1', 'S2', 'R&I', 'REMOVE', 'REPLACE', 'BUMPER', 'LAMP'
        ]
        
        has_automotive_patterns = any(pattern in header_text for pattern in automotive_patterns)
        logger.info(f"Header text: {header_text}")
        logger.info(f"Has automotive patterns: {has_automotive_patterns}")
        
        # If we have automotive patterns, it's likely a table - be more permissive
        if has_automotive_patterns:
            logger.info("Table validation passed: automotive patterns found")
            return True
        
        # Also check the data rows for automotive patterns
        for row in data_rows:
            row_text = ' '.join(row).upper()
            if any(pattern in row_text for pattern in automotive_patterns):
                logger.info(f"Found automotive patterns in data row: {row_text}")
                logger.info("Table validation passed: automotive patterns found in data")
                return True
        
        # Check 5: Avoid treating key-value pairs as tables
        kv_pattern_count = 0
        for row in data_rows:
            if len(row) == 2 and ':' in row[0]:
                kv_pattern_count += 1
        
        # If more than 70% of rows look like key-value pairs, it's probably not a table
        if kv_pattern_count > len(data_rows) * 0.7:
            logger.info(f"Table validation failed: too many key-value pairs ({kv_pattern_count}/{len(data_rows)})")
            return False
        
        # Check 6: Look for actual tabular data patterns
        has_numeric_data = False
        has_structured_data = False
        
        for row in data_rows:
            # Check for numeric values (prices, quantities, etc.)
            for cell in row:
                if re.search(r'\$\d+\.?\d*|\d+\.?\d*%|\d+\.?\d*\s*(units?|hrs?|hours?)', cell, re.IGNORECASE):
                    has_numeric_data = True
                    break
            
            # Check for structured data (line numbers, part numbers, etc.)
            if len(row) >= 3:
                # Look for patterns like "1", "R&I", "Description"
                if (re.match(r'^\d+$', row[0].strip()) and 
                    re.search(r'R\s*&\s*[A-Z]|REMOVE|REPLACE|REPAIR', row[1], re.IGNORECASE)):
                    has_structured_data = True
                    break
        
        logger.info(f"Has numeric data: {has_numeric_data}, has structured data: {has_structured_data}")
        
        # Final decision: Must have automotive patterns OR structured data with numeric content
        result = (has_automotive_patterns or (has_structured_data and has_numeric_data))
        logger.info(f"Table validation final result: {result}")
        return result
    
    def identify_sections(self, text_objects: List[Dict], page) -> List[Dict[str, Any]]:
        """
        Identify different sections of the page with intelligent grouping
        """
        if not text_objects:
            return []
        
        # First, identify logical content blocks based on spacing and content
        content_blocks = self.group_content_blocks(text_objects)
        
        # Then categorize each block
        result_sections = []
        for i, block in enumerate(content_blocks):
            if block['content']:
                # Sort content by position
                sorted_content = sorted(block['content'], 
                                      key=lambda x: (x['top'], x['x0']))
                
                # Determine section type based on content and position
                section_type = self.determine_section_type(sorted_content, block['name'])
                
                result_sections.append({
                    'name': block['name'],
                    'type': section_type,
                    'content': sorted_content,
                    'y_range': block['y_range'],
                    'block_index': i
                })
        
        return result_sections
    
    def group_content_blocks(self, text_objects: List[Dict]) -> List[Dict[str, Any]]:
        """
        Group text objects into logical content blocks based on spacing and content patterns
        """
        if not text_objects:
            return []
        
        # Sort by vertical position
        sorted_objects = sorted(text_objects, key=lambda x: x['top'])
        
        blocks = []
        current_block = []
        last_y = None
        spacing_threshold = 30  # pixels - significant gap indicates new block
        
        for obj in sorted_objects:
            if last_y is None:
                current_block.append(obj)
            elif obj['top'] - last_y <= spacing_threshold:
                # Small gap, same block
                current_block.append(obj)
            else:
                # Large gap, start new block
                if current_block:
                    blocks.append(self.create_content_block(current_block))
                current_block = [obj]
            
            last_y = obj['bottom']  # Use bottom for better spacing detection
        
        # Add the last block
        if current_block:
            blocks.append(self.create_content_block(current_block))
        
        return blocks
    
    def create_content_block(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Create a content block with metadata
        """
        if not content:
            return {'name': 'empty', 'content': [], 'y_range': (0, 0)}
        
        # Calculate block boundaries
        min_y = min(obj['top'] for obj in content)
        max_y = max(obj['bottom'] for obj in content)
        
        # Determine block type based on content
        all_text = ' '.join([obj['text'] for obj in content])
        all_text_upper = all_text.upper()
        
        # Identify block type based on content patterns
        if any(keyword in all_text_upper for keyword in ['ESTIMATE', 'CLAIM', 'POLICY', 'VIN']):
            block_name = 'document_header'
        elif any(keyword in all_text_upper for keyword in ['VEHICLE', 'YEAR', 'MAKE', 'MODEL']):
            block_name = 'vehicle_info'
        elif any(keyword in all_text_upper for keyword in ['LABOR', 'PARTS', 'TOTAL', 'COST']):
            block_name = 'cost_summary'
        elif any(keyword in all_text_upper for keyword in ['LINE', 'DESCRIPTION', 'OPERATION', 'R&I']):
            block_name = 'repair_items'
        elif any(keyword in all_text_upper for keyword in ['CUSTOMER', 'INSURED', 'CLAIMANT']):
            block_name = 'customer_info'
        elif any(keyword in all_text_upper for keyword in ['OPTIONS', 'FEATURES']):
            block_name = 'vehicle_options'
        elif any(keyword in all_text_upper for keyword in ['REMARKS', 'AGREED', 'SUPPLEMENT']):
            block_name = 'remarks_section'
        elif any(keyword in all_text_upper for keyword in ['COMMITTED', 'VERSION', 'PRINTED', 'COPYRIGHT']):
            block_name = 'footer_metadata'
        elif any(keyword in all_text_upper for keyword in ['PLEASE', 'CONTACT', 'REQUIRED', 'AUTHORIZATION']):
            block_name = 'disclaimer'
        else:
            block_name = 'general_content'
        
        return {
            'name': block_name,
            'content': content,
            'y_range': (min_y, max_y)
        }
    
    def determine_section_type(self, content: List[Dict], section_name: str) -> str:
        """
        Determine the type of content in a section
        """
        if not content:
            return 'empty'
        
        # Combine all text
        all_text = ' '.join([obj['text'] for obj in content])
        all_text_upper = all_text.upper()
        
        # Analyze content patterns
        if section_name == 'header':
            if any(keyword in all_text_upper for keyword in ['ESTIMATE', 'CLAIM', 'POLICY', 'VIN', 'INSURANCE']):
                return 'document_header'
            elif any(keyword in all_text_upper for keyword in ['COMPANY', 'ADDRESS', 'PHONE', 'FAX']):
                return 'company_header'
            else:
                return 'general_header'
        
        elif section_name == 'body':
            # Check for automotive repair patterns
            if re.search(r'\b(R\s*&\s*[A-Z]|S\d+)\b', all_text, re.IGNORECASE):
                return 'repair_items'
            elif re.search(r'\b(LABOR|PARTS|SUBLET|TAX|TOTAL)\b', all_text, re.IGNORECASE):
                return 'cost_summary'
            elif re.search(r'\b(SUPPLEMENT|ESTIMATE|AMOUNT)\b', all_text, re.IGNORECASE):
                return 'estimate_summary'
            elif any(keyword in all_text_upper for keyword in ['VEHICLE', 'YEAR', 'MAKE', 'MODEL']):
                return 'vehicle_info'
            elif any(keyword in all_text_upper for keyword in ['CUSTOMER', 'CLAIMANT', 'INSURED']):
                return 'customer_info'
            else:
                return 'general_body'
        
        elif section_name == 'footer':
            if any(keyword in all_text_upper for keyword in ['PLEASE', 'CONTACT', 'REQUIRED', 'AUTHORIZATION']):
                return 'disclaimer'
            elif any(keyword in all_text_upper for keyword in ['PAGE', 'TOTAL', 'SIGNATURE']):
                return 'page_footer'
            else:
                return 'general_footer'
        
        return 'unknown'
    
    def parse_sections(self, sections: List[Dict], page) -> List[Dict[str, Any]]:
        """
        Parse each section based on its content type
        """
        parsed_sections = []
        
        for section in sections:
            section_type = section['type']
            content = section['content']
            
            parsed_data = self.parse_section_content(content, section_type)
            
            parsed_sections.append({
                'name': section['name'],
                'type': section_type,
                'content': content,
                'parsed_data': parsed_data,
                'y_range': section['y_range']
            })
        
        return parsed_sections
    
    def parse_section_content(self, content: List[Dict], section_type: str) -> Dict[str, Any]:
        """
        Parse section content based on its type
        """
        if section_type == 'document_header':
            return self.parse_document_header(content)
        elif section_type == 'vehicle_info':
            return self.parse_vehicle_info(content)
        elif section_type == 'customer_info':
            return self.parse_customer_info(content)
        elif section_type == 'repair_items':
            return self.parse_repair_items(content)
        elif section_type == 'cost_summary':
            return self.parse_cost_summary(content)
        elif section_type == 'estimate_summary':
            return self.parse_estimate_summary(content)
        elif section_type == 'vehicle_options':
            return self.parse_vehicle_options(content)
        elif section_type == 'remarks_section':
            return self.parse_remarks_section(content)
        elif section_type == 'footer_metadata':
            return self.parse_footer_metadata(content)
        elif section_type == 'disclaimer':
            return self.parse_disclaimer(content)
        else:
            return self.parse_general_content(content)
    
    def parse_document_header(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse document header information
        """
        header_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract VIN (17 character alphanumeric)
            vin_match = re.search(r'\b[A-Z0-9]{17}\b', text.upper())
            if vin_match and 'vin' not in header_data:
                header_data['vin'] = vin_match.group()
            
            # Extract claim number
            if 'claim_number' not in header_data:
                claim_patterns = [
                    r'CLAIM\s*#?\s*:?\s*([A-Z0-9\-]+)',
                    r'CLAIM\s+NUMBER\s*:?\s*([A-Z0-9\-]+)',
                    r'#([A-Z0-9\-]{8,})'
                ]
                
                for pattern in claim_patterns:
                    claim_match = re.search(pattern, text, re.IGNORECASE)
                    if claim_match:
                        header_data['claim_number'] = claim_match.group(1)
                        break
            
            # Extract estimate number
            if 'estimate_number' not in header_data:
                estimate_patterns = [
                    r'ESTIMATE\s*#?\s*:?\s*([A-Z0-9\-]+)',
                    r'EST\s*#?\s*:?\s*([A-Z0-9\-]+)',
                    r'WORKFILE\s+ID\s*:?\s*([A-Z0-9\-]+)'
                ]
                
                for pattern in estimate_patterns:
                    est_match = re.search(pattern, text, re.IGNORECASE)
                    if est_match:
                        header_data['estimate_number'] = est_match.group(1)
                        break
        
        # If no structured data found, return raw text
        if not header_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                header_data['raw_text'] = raw_texts[:3]
        
        return header_data
    
    def parse_vehicle_info(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse vehicle information
        """
        vehicle_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract year
            if 'year' not in vehicle_data:
                year_match = re.search(r'\b(19|20)\d{2}\b', text)
                if year_match:
                    vehicle_data['year'] = year_match.group()
            
            # Extract make and model
            if 'make' not in vehicle_data:
                make_model_patterns = [
                    r'(\d{4})\s+([A-Z]+)\s+([A-Z\s]+)',
                    r'YEAR\s*:?\s*(\d{4})\s+MAKE\s*:?\s*([A-Z]+)\s+MODEL\s*:?\s*([A-Z\s]+)'
                ]
                
                for pattern in make_model_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        vehicle_data['year'] = match.group(1)
                        vehicle_data['make'] = match.group(2).strip()
                        vehicle_data['model'] = match.group(3).strip()
                        break
        
        # If no structured data found, return raw text
        if not vehicle_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                vehicle_data['raw_text'] = raw_texts[:3]
        
        return vehicle_data
    
    def parse_customer_info(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse customer information
        """
        customer_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract customer name
            name_patterns = [
                r'CUSTOMER\s*:?\s*([A-Z\s]+)',
                r'CLAIMANT\s*:?\s*([A-Z\s]+)',
                r'INSURED\s*:?\s*([A-Z\s]+)'
            ]
            
            for pattern in name_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    customer_data['name'] = match.group(1).strip()
                    break
        
        # If no structured data found, return raw text
        if not customer_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                customer_data['raw_text'] = raw_texts[:3]
        
        return customer_data
    
    def parse_repair_items(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse repair line items
        """
        repair_items = []
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Parse automotive repair line item
            item = self.parse_repair_line_item(text)
            if item:
                repair_items.append(item)
        
        # If no items found, return raw text
        if not repair_items:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                return {'raw_text': raw_texts[:5]}
        
        return {'items': repair_items}
    
    def parse_repair_line_item(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse a single repair line item
        """
        # Pattern 1: S1/S2 prefix + line number + operation + description
        prefix_op_match = re.match(r'^(S\d+)\s+(\d+)\s+(R\s*&\s*[A-Z])\s+(.+)$', text, re.IGNORECASE)
        if prefix_op_match:
            return {
                'supplement': prefix_op_match.group(1),
                'line_number': prefix_op_match.group(2),
                'operation': prefix_op_match.group(3).strip(),
                'description': prefix_op_match.group(4).strip()
            }
        
        # Pattern 2: Line number + operation + description
        op_match = re.match(r'^(\d+)\s+(R\s*&\s*[A-Z])\s+(.+)$', text, re.IGNORECASE)
        if op_match:
            return {
                'line_number': op_match.group(1),
                'operation': op_match.group(2).strip(),
                'description': op_match.group(3).strip()
            }
        
        # Pattern 3: S1/S2 prefix + line number + item ID + description
        prefix_id_match = re.match(r'^(S\d+)\s+(\d+)\s+(\d+)\s+(.+)$', text)
        if prefix_id_match:
            return {
                'supplement': prefix_id_match.group(1),
                'line_number': prefix_id_match.group(2),
                'item_id': prefix_id_match.group(3),
                'description': prefix_id_match.group(4).strip()
            }
        
        # Pattern 4: Line number + item ID + description
        id_match = re.match(r'^(\d+)\s+(\d+)\s+(.+)$', text)
        if id_match:
            return {
                'line_number': id_match.group(1),
                'item_id': id_match.group(2),
                'description': id_match.group(3).strip()
            }
        
        # Pattern 5: Simple line number + description
        simple_match = re.match(r'^(\d+)\s+(.+)$', text)
        if simple_match:
            return {
                'line_number': simple_match.group(1),
                'description': simple_match.group(2).strip()
            }
        
        return None
    
    def parse_cost_summary(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse cost summary information
        """
        cost_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract labor rates
            labor_rate_match = re.search(r'LABOR\s+RATES\s+([^$]+)\s+\$([\d,]+)/\$([\d,]+)', text, re.IGNORECASE)
            if labor_rate_match:
                cost_data['labor_rates'] = {
                    'carrier': f"${labor_rate_match.group(2)}/${labor_rate_match.group(3)}",
                    'description': labor_rate_match.group(1).strip()
                }
            
            # Extract labor costs
            labor_patterns = [
                r'BODY\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)',
                r'REFINISH\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)',
                r'MECHANICAL\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)'
            ]
            
            for pattern in labor_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'labor' not in cost_data:
                        cost_data['labor'] = {}
                    
                    labor_type = 'body' if 'BODY' in pattern else 'refinish' if 'REFINISH' in pattern else 'mechanical'
                    cost_data['labor'][labor_type] = {
                        'units': match.group(1),
                        'rate': match.group(2),
                        'sublet': match.group(3) if len(match.groups()) > 2 else '0.00',
                        'total': match.group(4) if len(match.groups()) > 3 else match.group(3)
                    }
            
            # Extract parts costs
            parts_match = re.search(r'TAXABLE\s+PARTS\s+\$([\d,]+\.\d+)', text, re.IGNORECASE)
            if parts_match:
                cost_data['parts'] = parts_match.group(1)
            
            # Extract tax information
            tax_match = re.search(r'TAXABLE\s+\$([\d,]+\.\d+)\s+TAX\s+([\d\.]+)%\s+\$([\d,]+\.\d+)', text, re.IGNORECASE)
            if tax_match:
                cost_data['tax'] = {
                    'taxable_amount': tax_match.group(1),
                    'tax_rate': tax_match.group(2),
                    'tax_amount': tax_match.group(3)
                }
            
            # Extract total
            total_match = re.search(r'LABOR\s+TOTAL\s+\$([\d,]+\.\d+)', text, re.IGNORECASE)
            if total_match:
                cost_data['labor_total'] = total_match.group(1)
        
        # If no structured data found, return raw text
        if not cost_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                cost_data['raw_text'] = raw_texts[:5]
        
        return cost_data
    
    def parse_estimate_summary(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse estimate summary information
        """
        summary_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract supplement estimates
            supplement_patterns = [
                r'SUPPLEMENT\s*(\d+)\s*\$([\d,]+\.\d+)',
                r'ESTIMATE\s*(\d+)\s*\$([\d,]+\.\d+)'
            ]
            
            for pattern in supplement_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if 'supplements' not in summary_data:
                        summary_data['supplements'] = []
                    summary_data['supplements'].append({
                        'number': match.group(1),
                        'amount': match.group(2)
                    })
            
            # Extract labor totals
            labor_patterns = [
                r'BODY\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)',
                r'REFINISH\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)',
                r'MECHANICAL\s+LABOR\s+(\d+\.?\d*)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)\s+\$([\d,]+\.\d+)'
            ]
            
            for pattern in labor_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'labor' not in summary_data:
                        summary_data['labor'] = {}
                    
                    labor_type = 'body' if 'BODY' in pattern else 'refinish' if 'REFINISH' in pattern else 'mechanical'
                    summary_data['labor'][labor_type] = {
                        'units': match.group(1),
                        'rate': match.group(2),
                        'sublet': match.group(3) if len(match.groups()) > 2 else '0.00',
                        'total': match.group(4) if len(match.groups()) > 3 else match.group(3)
                    }
            
            # Extract parts amount
            parts_match = re.search(r'TAXABLE\s+PARTS\s+\$([\d,]+\.\d+)', text, re.IGNORECASE)
            if parts_match:
                summary_data['parts_amount'] = parts_match.group(1)
            
            # Extract net totals
            net_patterns = [
                r'NET\s+SUPPLEMENT\s+\$([\d,]+\.\d+)',
                r'NET\s+ESTIMATE\s+TOTAL\s+\$([\d,]+\.\d+)',
                r'ORIGINAL\s+ESTIMATE\s+\$([\d,]+\.\d+)'
            ]
            
            for pattern in net_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'net_totals' not in summary_data:
                        summary_data['net_totals'] = {}
                    
                    if 'NET SUPPLEMENT' in pattern:
                        summary_data['net_totals']['net_supplement'] = match.group(1)
                    elif 'NET ESTIMATE TOTAL' in pattern:
                        summary_data['net_totals']['net_estimate_total'] = match.group(1)
                    elif 'ORIGINAL ESTIMATE' in pattern:
                        summary_data['net_totals']['original_estimate'] = match.group(1)
        
        # If no structured data found, return raw text
        if not summary_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                summary_data['raw_text'] = raw_texts[:5]
        
        return summary_data
    
    def parse_vehicle_options(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse vehicle options/features list
        """
        options = []
        for obj in content:
            text = obj['text'].strip()
            if text and len(text) > 2:
                # Clean up option text
                option = self.clean_text(text)
                if option and not option.startswith('Options'):
                    options.append(option)
        
        return {'options': options, 'count': len(options)}
    
    def parse_remarks_section(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse remarks and agreement sections
        """
        remarks = []
        agreements = []
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Look for agreement patterns
            if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}\s+AGREED', text, re.IGNORECASE):
                agreements.append(self.clean_text(text))
            elif text.startswith('Remarks') or 'AGREED' in text.upper():
                remarks.append(self.clean_text(text))
            else:
                remarks.append(self.clean_text(text))
        
        return {
            'remarks': remarks,
            'agreements': agreements,
            'total_items': len(remarks) + len(agreements)
        }
    
    def parse_footer_metadata(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse footer metadata (dates, versions, etc.)
        """
        metadata = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Extract key-value pairs
            kv_patterns = [
                r'(Committed On):\s*([^\n]+)',
                r'(Printed On):\s*([^\n]+)',
                r'(Version):\s*([^\n]+)',
                r'(Profile):\s*([^\n]+)',
                r'(Page):\s*([^\n]+)'
            ]
            
            for pattern in kv_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    key = match.group(1).lower().replace(' ', '_')
                    value = self.clean_text(match.group(2))
                    metadata[key] = value
                    break
            
            # Extract copyright info
            if 'Copyright' in text:
                metadata['copyright'] = self.clean_text(text)
        
        return metadata
    
    def parse_disclaimer(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse disclaimer text
        """
        disclaimer_text = ' '.join([obj['text'] for obj in content])
        return {'text': self.clean_text(disclaimer_text)}
    
    def parse_general_content(self, content: List[Dict]) -> Dict[str, Any]:
        """
        Parse general content as key-value pairs
        """
        general_data = {}
        
        for obj in content:
            text = obj['text'].strip()
            if not text:
                continue
            
            # Look for key-value patterns
            kv_patterns = [
                r'([A-Za-z\s]+):\s*([^\n]+)',
                r'([A-Za-z\s]+)\s*[-=]\s*([^\n]+)'
            ]
            
            for pattern in kv_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    key = self.clean_text(match.group(1))
                    value = self.clean_text(match.group(2))
                    if key and value and len(key.strip()) > 1:
                        general_data[key] = value
        
        # If no structured data found, return raw text
        if not general_data:
            raw_texts = [obj['text'] for obj in content if obj['text'].strip()]
            if raw_texts:
                general_data['raw_text'] = raw_texts[:5]
        
        return general_data
    
    def create_document_summary(self, pages: List[Dict]) -> Dict[str, Any]:
        """
        Create a summary of the entire document with advanced extraction data
        """
        summary = {
            'document_info': {},
            'vehicle_info': {},
            'customer_info': {},
            'repair_items': [],
            'cost_summary': {},
            'estimate_summary': {},
            'vehicle_options': [],
            'remarks_sections': [],
            'footer_metadata': {},
            'disclaimers': [],
            'tables': [],
            # New advanced extraction fields
            'advanced_metadata': {},
            'advanced_line_items': [],
            'advanced_totals': {},
            'validation_results': {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'statistics': {}
            }
        }
        
        # Collect advanced extraction data
        all_metadata = {}
        all_line_items = []
        all_totals = {}
        all_warnings = []
        all_errors = []
        total_line_items = 0
        total_calculated = 0
        
        for page in pages:
            # Collect legacy tables
            summary['tables'].extend(page.get('tables', []))
            
            # Collect advanced extraction data
            if 'metadata' in page:
                all_metadata.update(page['metadata'])
            
            if 'line_items_table' in page and page['line_items_table']:
                all_line_items.extend(page['line_items_table'])
                total_line_items += len(page['line_items_table'])
            
            if 'totals' in page:
                all_totals.update(page['totals'])
            
            # Collect validation data
            if 'validated_data' in page:
                validation = page['validated_data']
                if 'warnings' in validation:
                    all_warnings.extend(validation['warnings'])
                if 'errors' in validation:
                    all_errors.extend(validation['errors'])
                if 'statistics' in validation:
                    if 'calculated_total' in validation['statistics']:
                        total_calculated += validation['statistics']['calculated_total']
            
            # Collect section data (legacy method)
            for section in page.get('sections', []):
                section_type = section['type']
                parsed_data = section.get('parsed_data', {})
                
                if section_type == 'document_header':
                    summary['document_info'].update(parsed_data)
                elif section_type == 'vehicle_info':
                    summary['vehicle_info'].update(parsed_data)
                elif section_type == 'customer_info':
                    summary['customer_info'].update(parsed_data)
                elif section_type == 'repair_items':
                    if 'items' in parsed_data:
                        summary['repair_items'].extend(parsed_data['items'])
                elif section_type == 'cost_summary':
                    summary['cost_summary'].update(parsed_data)
                elif section_type == 'estimate_summary':
                    summary['estimate_summary'].update(parsed_data)
                elif section_type == 'vehicle_options':
                    if 'options' in parsed_data:
                        summary['vehicle_options'].extend(parsed_data['options'])
                elif section_type == 'remarks_section':
                    summary['remarks_sections'].append(parsed_data)
                elif section_type == 'footer_metadata':
                    summary['footer_metadata'].update(parsed_data)
                elif section_type == 'disclaimer':
                    summary['disclaimers'].append(parsed_data)
        
        # Update summary with advanced data
        summary['advanced_metadata'] = all_metadata
        summary['advanced_line_items'] = all_line_items
        summary['advanced_totals'] = all_totals
        
        # Update validation results
        summary['validation_results']['warnings'] = all_warnings
        summary['validation_results']['errors'] = all_errors
        summary['validation_results']['statistics'] = {
            'total_line_items': total_line_items,
            'total_calculated': total_calculated,
            'total_pages': len(pages)
        }
        
        # Overall validation
        if all_warnings or all_errors:
            summary['validation_results']['is_valid'] = False
        
        return summary
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        if not text:
            return ""
        
        # Remove control characters first
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Fix common character spacing issues (like "A A m m e e r r i i c c a a n n")
        # This pattern matches repeated characters with spaces
        text = re.sub(r'(\w)\s+\1', r'\1', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR artifacts
        text = re.sub(r'([A-Z])\s+([A-Z])', r'\1\2', text)  # Fix spaced letters
        text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)  # Fix spaced numbers
        
        return text
    
    # ============================================================================
    # ADVANCED SEGMENTATION AND EXTRACTION METHODS (ChatGPT Guidelines)
    # ============================================================================
    
    def segment_page(self, page) -> Tuple[Any, Any, Any]:
        """
        Segment the page into metadata, body, and footer regions
        Based on ChatGPT guidelines for optimal PDF processing
        """
        try:
            # Extract all words with their positions
            words = page.extract_words(x_tolerance=5, y_tolerance=5)
            
            # Find the start of the main table (look for "Line #" or similar)
            start_y = None
            for word in words:
                if re.search(r'Line\s*#|LABOR|PART|Description|Operation', word['text'], re.IGNORECASE):
                    start_y = word['top']
                    break
            
            # Find the end of the main table (look for "Estimate Totals" or similar)
            end_y = None
            for word in words:
                if re.search(r'Estimate\s+Totals|Total\s+Estimate|Grand\s+Total|Net\s+Estimate', word['text'], re.IGNORECASE):
                    end_y = word['top']
                    break
            
            # If we can't find clear boundaries, use page proportions
            if start_y is None:
                start_y = page.height * 0.3  # Top 30% for metadata
            
            if end_y is None:
                end_y = page.height * 0.8  # Bottom 20% for footer
            
            # Ensure boundaries are within page limits
            start_y = max(0, min(start_y, page.height))
            end_y = max(start_y, min(end_y, page.height))
            
            # Create the three regions with proper bounds checking
            try:
                meta_region = page.within_bbox((0, 0, page.width, start_y))
            except:
                meta_region = page
            
            try:
                body_region = page.within_bbox((0, start_y, page.width, end_y))
            except:
                body_region = page
            
            try:
                footer_region = page.within_bbox((0, end_y, page.width, page.height))
            except:
                footer_region = page
            
            logger.info(f"Page segmented: metadata (0-{start_y:.1f}), body ({start_y:.1f}-{end_y:.1f}), footer ({end_y:.1f}-{page.height:.1f})")
            return meta_region, body_region, footer_region
            
        except Exception as e:
            logger.warning(f"Page segmentation failed: {e}")
            # Fallback: return the same page for all regions
            return page, page, page
    
    def extract_metadata(self, meta_region) -> Dict[str, str]:
        """
        Extract metadata (key-value pairs) from the metadata region
        Based on ChatGPT guidelines for robust metadata extraction
        """
        metadata = {}
        
        try:
            # Extract text and split into lines
            text = meta_region.extract_text()
            if not text:
                return metadata
            
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Pattern 1: "Key: Value" format
                match = re.match(r'^([\w\s/]+?):\s*(.+)$', line)
                if match:
                    key, value = match.groups()
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        metadata[key] = value
                        continue
                
                # Pattern 2: "Key Value" format (no colon)
                # Look for common automotive metadata patterns
                patterns = [
                    r'^VIN[:\s]*([A-Z0-9]{17})',
                    r'^Estimate\s+ID[:\s]*([A-Z0-9-]+)',
                    r'^Claim\s+#[:\s]*([A-Z0-9-]+)',
                    r'^Policy\s+#[:\s]*([A-Z0-9-]+)',
                    r'^Year[:\s]*(\d{4})',
                    r'^Make[:\s]*([A-Za-z\s]+)',
                    r'^Model[:\s]*([A-Za-z0-9\s]+)',
                    r'^Mileage[:\s]*([0-9,]+)',
                    r'^Date[:\s]*([0-9/]+)',
                    r'^Time[:\s]*([0-9:]+)',
                    r'^Federal\s+ID[:\s]*([0-9-]+)',
                    r'^Customer[:\s]*([A-Za-z\s,]+)',
                    r'^Job\s+Number[:\s]*([A-Z0-9-]+)'
                ]
                
                for pattern in patterns:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        key = re.match(r'^([A-Za-z\s]+)', line).group(1).strip()
                        value = match.group(1).strip()
                        metadata[key] = value
                        break
        
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def extract_line_items_table(self, body_region) -> List[Dict[str, Any]]:
        """
        Extract the main line-items table with precise column boundaries
        Based on ChatGPT guidelines for robust table extraction
        """
        try:
            # Step 1: Detect column boundaries by analyzing header positions
            col_edges = self.infer_column_boundaries(body_region)
            
            if not col_edges:
                logger.warning("Could not infer column boundaries")
                return []
            
            # Step 2: Extract table data using column boundaries
            raw_rows = self.extract_table_by_columns(body_region, col_edges)
            
            # Step 3: Clean and normalize the data
            clean_rows = self.clean_and_normalize_table_rows(raw_rows)
            
            # Step 4: Convert to structured format
            col_names = ["Line#", "PartCode", "Description", "Operation", "Type", "Units", "Rate", "PartType", "Number", "Qty", "TotalPrice", "Tax"]
            
            line_items = []
            for row in clean_rows:
                if len(row) >= 3:  # At least line number, description, and one other field
                    item = dict(zip(col_names[:len(row)], row))
                    line_items.append(item)
            
            return line_items
            
        except Exception as e:
            logger.warning(f"Line items table extraction failed: {e}")
            return []
    
    def _infer_column_boundaries_old(self, body_region) -> List[Tuple[float, float]]:
        """
        Infer column boundaries by analyzing header word positions
        Based on ChatGPT guidelines for precise column detection
        """
        try:
            # Extract all words from the body region with better settings
            words = body_region.extract_words(x_tolerance=5, y_tolerance=5)
            
            # First, try to find the actual repair table headers
            repair_table_headers = ["LINE", "DESCRIPTION", "OPERATION", "TYPE", "UNITS", "CEG", "NUMBER", "QTY", "PRICE", "TAX"]
            header_words = []
            
            # Look for header words in the top 30% of the body region
            top_threshold = body_region.height * 0.3
            top_words = [w for w in words if w['top'] < top_threshold]
            
            # Look for exact repair table headers first
            for word in top_words:
                if any(header.lower() in word['text'].lower() for header in repair_table_headers):
                    header_words.append(word)
            
            # If we found repair table headers, use them to create precise boundaries
            if len(header_words) >= 3:  # Need at least 3 headers to be confident
                x_positions = sorted(set(round(w['x0']) for w in header_words))
                
                # Create column boundaries for repair table
                col_edges = []
                for i in range(len(x_positions) - 1):
                    col_edges.append((x_positions[i], x_positions[i + 1]))
                
                # Add final column boundary
                col_edges.append((x_positions[-1], body_region.width))
                
                logger.info(f"Found repair table headers: {[w['text'] for w in header_words]}")
                logger.info(f"Created {len(col_edges)} column boundaries for repair table")
                return col_edges
            
            # Fallback: look for any table-like headers
            header_keywords = ["Line", "Description", "Operation", "Type", "Units", "Rate", "Total", "Tax", "Qty", "Number", "Part", "CEG"]
            
            for word in top_words:
                if any(keyword.lower() in word['text'].lower() for keyword in header_keywords):
                    header_words.append(word)
            
            if not header_words:
                # If no headers found, look for words that might be headers
                for word in top_words:
                    if len(word['text']) > 2 and word['text'].isupper():
                        header_words.append(word)
            
            if not header_words:
                # Last resort: use the first few words as potential headers
                header_words = top_words[:10]
            
            # Get unique x-coordinates and sort them
            x_positions = sorted(set(round(w['x0']) for w in header_words))
            
            if len(x_positions) < 2:
                return []
            
            # Filter out positions that are too close together (likely same column)
            filtered_positions = [x_positions[0]]
            for pos in x_positions[1:]:
                if pos - filtered_positions[-1] > 30:  # Minimum 30 points between columns
                    filtered_positions.append(pos)
            
            if len(filtered_positions) < 2:
                return []
            
            # Create column boundaries (limit to reasonable number of columns)
            col_edges = []
            max_columns = 15  # Limit to prevent over-segmentation
            
            for i in range(min(len(filtered_positions) - 1, max_columns - 1)):
                col_edges.append((filtered_positions[i], filtered_positions[i + 1]))
            
            # Add final column boundary
            if len(col_edges) < max_columns:
                col_edges.append((filtered_positions[-1], body_region.width))
            
            logger.info(f"Inferred {len(col_edges)} column boundaries from {len(header_words)} header words")
            return col_edges
            
        except Exception as e:
            logger.warning(f"Column boundary inference failed: {e}")
            return []
    
    def _extract_table_by_columns_old(self, body_region, col_edges: List[Tuple[float, float]]) -> List[List[str]]:
        """
        Extract table data by mapping words into rows and columns
        Based on ChatGPT guidelines for precise table extraction
        """
        try:
            from collections import defaultdict
            
            # Extract all words with better settings
            words = body_region.extract_words(x_tolerance=5, y_tolerance=5)
            
            # Group words by row (using y-coordinate with tolerance)
            rows = defaultdict(lambda: [""] * len(col_edges))
            
            for word in words:
                # Find row by rounding top coordinate with tolerance
                row_key = round(word['top'] / 10) * 10  # Group within 10 points
                
                # Find which column it belongs to
                col_idx = None
                for i, (x0, x1) in enumerate(col_edges):
                    if x0 <= word['x0'] < x1:
                        col_idx = i
                        break
                
                if col_idx is not None:
                    # Append text to the appropriate column
                    current_text = rows[row_key][col_idx]
                    if current_text:
                        rows[row_key][col_idx] = current_text + " " + word['text']
                    else:
                        rows[row_key][col_idx] = word['text']
            
            # Convert to list and sort by row key
            sorted_rows = []
            for row_key in sorted(rows.keys()):
                row_data = rows[row_key]
                # Only include rows that have meaningful content
                if any(cell.strip() for cell in row_data):
                    # Clean up the row data
                    cleaned_row = []
                    for cell in row_data:
                        cell = cell.strip()
                        # Remove excessive spaces
                        cell = re.sub(r'\s+', ' ', cell)
                        cleaned_row.append(cell)
                    sorted_rows.append(cleaned_row)
            
            logger.info(f"Extracted {len(sorted_rows)} rows with {len(col_edges)} columns")
            return sorted_rows
            
        except Exception as e:
            logger.warning(f"Table extraction by columns failed: {e}")
            return []
    
    def _clean_and_normalize_table_rows_old(self, raw_rows: List[List[str]]) -> List[List[str]]:
        """
        Clean and normalize table rows
        Based on ChatGPT guidelines for data cleaning
        """
        clean_rows = []
        
        for row in raw_rows:
            if not row or not any(cell.strip() for cell in row):
                continue
            
            # Clean each cell
            clean_row = []
            for cell in row:
                cell = cell.strip()
                
                # Normalize operations
                if re.search(r'(Remove|R&R|R&I)', cell, re.IGNORECASE):
                    cell = "Remove/Replace"
                
                # Clean numeric values
                if re.search(r'\$[\d,]+\.?\d*', cell):
                    cell = re.sub(r'[^\d.]', '', cell)  # Keep only digits and decimal
                
                # Clean up common issues
                cell = re.sub(r'\s+', ' ', cell)  # Multiple spaces to single
                cell = re.sub(r'\*$', '', cell)  # Remove trailing asterisk
                
                clean_row.append(cell)
            
            # Check if this looks like a repair line item
            is_repair_line = False
            
            # Look for line number at start
            if clean_row and re.match(r'^\d+', clean_row[0]):
                is_repair_line = True
            
            # Look for repair operations
            row_text = ' '.join(clean_row).upper()
            if re.search(r'R&[IR]|REMOVE|REPLACE|REPAIR|REFINISH|BUMPER|LAMP|FENDER', row_text):
                is_repair_line = True
            
            # Look for automotive part numbers or descriptions
            if re.search(r'\d{3,6}|BUMPER|COVER|LAMP|FENDER|MOLDING', row_text):
                is_repair_line = True
            
            # Only include rows that look like repair items or have meaningful content
            if is_repair_line or len([c for c in clean_row if c]) >= 3:
                clean_rows.append(clean_row)
        
        logger.info(f"Cleaned {len(clean_rows)} rows from {len(raw_rows)} raw rows")
        return clean_rows
    
    def extract_totals(self, footer_region) -> Dict[str, str]:
        """
        Extract totals and summary information from the footer region
        Based on ChatGPT guidelines for totals extraction
        """
        totals = {}
        
        try:
            text = footer_region.extract_text()
            if not text:
                return totals
            
            lines = text.split('\n')
            
            # Look for total patterns
            total_patterns = [
                r'Total\s+Labor[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Paint[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Parts[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Sublet[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Other[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Tax[:\s]*\$?([\d,]+\.?\d*)',
                r'Total\s+Estimate[:\s]*\$?([\d,]+\.?\d*)',
                r'Grand\s+Total[:\s]*\$?([\d,]+\.?\d*)',
                r'Estimate\s+Total[:\s]*\$?([\d,]+\.?\d*)'
            ]
            
            for line in lines:
                line = line.strip()
                for pattern in total_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        key = re.match(r'([A-Za-z\s]+)', line).group(1).strip()
                        value = match.group(1).strip()
                        totals[key] = value
                        break
        
        except Exception as e:
            logger.warning(f"Totals extraction failed: {e}")
        
        return totals
    
    def validate_and_clean_extracted_data(self, metadata: Dict[str, str], line_items: List[Dict], totals: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate and clean the extracted data with sanity checks
        Based on ChatGPT guidelines for data validation
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Validate line items
            if line_items:
                # Check that every row has a valid line number
                invalid_lines = []
                for i, item in enumerate(line_items):
                    if 'Line#' in item and not re.match(r'^\d+', str(item['Line#'])):
                        invalid_lines.append(i)
                
                if invalid_lines:
                    validation_results['warnings'].append(f"Invalid line numbers found in rows: {invalid_lines}")
                
                # Calculate total from line items
                calculated_total = 0
                for item in line_items:
                    if 'TotalPrice' in item and item['TotalPrice']:
                        try:
                            price = float(str(item['TotalPrice']).replace(',', ''))
                            calculated_total += price
                        except ValueError:
                            pass
                
                validation_results['statistics']['calculated_total'] = calculated_total
                validation_results['statistics']['line_items_count'] = len(line_items)
            
            # Validate metadata
            required_fields = ['VIN', 'Estimate ID', 'Year', 'Make', 'Model']
            missing_fields = [field for field in required_fields if field not in metadata]
            
            if missing_fields:
                validation_results['warnings'].append(f"Missing recommended fields: {missing_fields}")
            
            # Validate totals
            if totals and 'calculated_total' in validation_results['statistics']:
                # Try to find a total in the totals section
                total_found = False
                for key, value in totals.items():
                    if 'total' in key.lower() and value:
                        try:
                            extracted_total = float(str(value).replace(',', ''))
                            calculated_total = validation_results['statistics']['calculated_total']
                            
                            # Allow for small differences (tax, rounding, etc.)
                            if abs(extracted_total - calculated_total) < 100:
                                total_found = True
                                break
                        except ValueError:
                            pass
                
                if not total_found:
                    validation_results['warnings'].append("Total from line items doesn't match extracted totals")
            
            # Overall validation
            if validation_results['warnings']:
                validation_results['is_valid'] = False
        
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
            validation_results['is_valid'] = False
        
        return validation_results

# Initialize the analyzer
analyzer = PDFAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.pdf'):
            # Save uploaded file
            upload_path = os.path.join('uploads', file.filename)
            file.save(upload_path)
            
            # Analyze the PDF
            result = analyzer.analyze_pdf(upload_path)
            
            return jsonify(result)
    
    return render_template('upload.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_pdf_api():
    """
    API endpoint for PDF analysis
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and file.filename.endswith('.pdf'):
        # Save uploaded file
        upload_path = os.path.join('uploads', file.filename)
        file.save(upload_path)
        
        # Analyze the PDF
        result = analyzer.analyze_pdf(upload_path)
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    app.run(debug=True, port=5001)
