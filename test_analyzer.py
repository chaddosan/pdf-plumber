#!/usr/bin/env python3
"""
Test script for the PDF Analyzer
"""

import os
import sys
from app import analyzer

def test_pdf_analysis():
    """Test the PDF analyzer with sample files"""
    
    # Test files
    test_files = [
        'uploads/5UXWX9C53H0T15726 2017 BMW - Insurance.pdf',
        'uploads/5UXWX9C53H0T15726 2017 BMW - Estimate.pdf',
        'uploads/JTMBFREVXEJ007006 2014 Rav4 - Estimate.pdf'
    ]
    
    for pdf_path in test_files:
        if os.path.exists(pdf_path):
            print(f"\n{'='*60}")
            print(f"Testing: {os.path.basename(pdf_path)}")
            print(f"{'='*60}")
            
            try:
                # Analyze the PDF
                result = analyzer.analyze_pdf(pdf_path)
                
                if 'error' in result:
                    print(f"❌ Error: {result['error']}")
                    continue
                
                print(f"✅ Successfully analyzed {result['total_pages']} pages")
                
                # Print summary stats
                summary = result.get('summary', {})
                print(f"📊 Document Info: {len(summary.get('document_info', {}))} fields")
                print(f"🚗 Vehicle Info: {len(summary.get('vehicle_info', {}))} fields")
                print(f"🔧 Repair Items: {len(summary.get('repair_items', []))} items")
                print(f"📋 Tables Found: {len(summary.get('tables', []))} tables")
                
                # Print some extracted data
                if summary.get('document_info'):
                    print("\n📄 Document Information:")
                    for key, value in summary['document_info'].items():
                        if key != 'raw_text':
                            print(f"  {key}: {value}")
                
                if summary.get('vehicle_info'):
                    print("\n🚗 Vehicle Information:")
                    for key, value in summary['vehicle_info'].items():
                        if key != 'raw_text':
                            print(f"  {key}: {value}")
                
                if summary.get('repair_items'):
                    print(f"\n🔧 Repair Items ({len(summary['repair_items'])} found):")
                    for i, item in enumerate(summary['repair_items'][:3]):  # Show first 3
                        print(f"  {i+1}. {item}")
                
                # Page analysis
                pages = result.get('pages', [])
                print(f"\n📄 Page Analysis:")
                for page in pages:
                    debug = page.get('debug_info', {})
                    print(f"  Page {page['page_num']}: {debug.get('original_chars', 0)} chars, "
                          f"{debug.get('tables_found', 0)} tables, {debug.get('sections_found', 0)} sections")
                
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
        
        else:
            print(f"❌ File not found: {pdf_path}")

if __name__ == "__main__":
    print("🧪 Testing PDF Analyzer")
    print("Using pdfplumber best practices with proper cropping and section analysis")
    test_pdf_analysis()
    print("\n✅ Test completed!") 