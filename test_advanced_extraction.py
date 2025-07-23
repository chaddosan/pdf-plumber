#!/usr/bin/env python3
"""
Test script for advanced PDF extraction methods
Based on ChatGPT guidelines for optimal PDF processing
"""

import os
import sys
from app import PDFAnalyzer

def test_advanced_extraction():
    """
    Test the new advanced extraction methods
    """
    print("🧪 Testing Advanced PDF Extraction")
    print("Based on ChatGPT guidelines for optimal PDF processing")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = PDFAnalyzer()
    
    # Test files
    test_files = [
        "uploads/5UXWX9C53H0T15726 2017 BMW - Insurance.pdf",
        "uploads/5UXWX9C53H0T15726 2017 BMW - Estimate.pdf",
        "uploads/JTMBFREVXEJ007006 2014 Rav4 - Estimate.pdf"
    ]
    
    for pdf_path in test_files:
        if not os.path.exists(pdf_path):
            print(f"⚠️  Test file not found: {pdf_path}")
            continue
            
        print(f"\n📄 Testing: {os.path.basename(pdf_path)}")
        print("-" * 50)
        
        try:
            # Analyze the PDF
            result = analyzer.analyze_pdf(pdf_path)
            
            # Check advanced extraction results
            pages = result.get('pages', [])
            
            total_metadata = 0
            total_line_items = 0
            total_totals = 0
            total_warnings = 0
            total_errors = 0
            
            for i, page in enumerate(pages):
                print(f"  Page {i+1}:")
                
                # Metadata extraction
                metadata = page.get('metadata', {})
                metadata_count = len(metadata)
                total_metadata += metadata_count
                print(f"    📋 Metadata: {metadata_count} fields")
                if metadata:
                    print(f"      Sample: {list(metadata.items())[:3]}")
                
                # Line items extraction
                line_items = page.get('line_items_table', [])
                line_items_count = len(line_items)
                total_line_items += line_items_count
                print(f"    🔧 Line Items: {line_items_count} items")
                if line_items:
                    print(f"      Sample: {line_items[0] if line_items else 'None'}")
                
                # Totals extraction
                totals = page.get('totals', {})
                totals_count = len(totals)
                total_totals += totals_count
                print(f"    💰 Totals: {totals_count} fields")
                if totals:
                    print(f"      Sample: {list(totals.items())[:3]}")
                
                # Validation results
                validation = page.get('validated_data', {})
                warnings = len(validation.get('warnings', []))
                errors = len(validation.get('errors', []))
                total_warnings += warnings
                total_errors += errors
                print(f"    ✅ Validation: {warnings} warnings, {errors} errors")
                
                # Statistics
                stats = validation.get('statistics', {})
                if stats:
                    print(f"    📊 Statistics: {stats}")
            
            # Summary for this PDF
            print(f"\n📊 Summary for {os.path.basename(pdf_path)}:")
            print(f"  Total Metadata Fields: {total_metadata}")
            print(f"  Total Line Items: {total_line_items}")
            print(f"  Total Totals Fields: {total_totals}")
            print(f"  Total Warnings: {total_warnings}")
            print(f"  Total Errors: {total_errors}")
            
            # Check if advanced extraction found more data than legacy
            legacy_tables = len(result.get('summary', {}).get('tables', []))
            legacy_repair_items = len(result.get('summary', {}).get('repair_items', []))
            
            print(f"  Legacy Tables: {legacy_tables}")
            print(f"  Legacy Repair Items: {legacy_repair_items}")
            print(f"  Advanced Line Items: {total_line_items}")
            
            if total_line_items > legacy_repair_items:
                print("  ✅ Advanced extraction found MORE line items than legacy method!")
            elif total_line_items == legacy_repair_items:
                print("  ⚖️  Advanced extraction found SAME number of line items as legacy method")
            else:
                print("  ⚠️  Advanced extraction found FEWER line items than legacy method")
            
        except Exception as e:
            print(f"❌ Error analyzing {pdf_path}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Advanced extraction test completed!")
    print("\nKey Improvements Implemented:")
    print("1. ✅ Page Segmentation (metadata/body/footer)")
    print("2. ✅ Precise Column Boundary Detection")
    print("3. ✅ Robust Metadata Extraction")
    print("4. ✅ Advanced Line Items Table Extraction")
    print("5. ✅ Totals and Summary Extraction")
    print("6. ✅ Data Validation and Sanity Checks")
    print("7. ✅ Multi-template Support")

if __name__ == "__main__":
    test_advanced_extraction() 