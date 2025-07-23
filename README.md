# PDF Analyzer Pro

A modern, elegant web application for advanced PDF document processing using pdfplumber best practices. This application is specifically designed to handle automotive repair estimates and insurance documents, but can be adapted for other document types.

## 🚀 Features

### Core Functionality
- **Advanced PDF Processing**: Uses pdfplumber's latest features and best practices
- **Page Cropping**: Automatically crops page margins for better content extraction
- **Section Identification**: Intelligently identifies document sections (header, body, footer)
- **Table Detection**: Advanced table extraction without requiring vertical/horizontal lines
- **Content Classification**: Automatically categorizes content as key-value pairs, text, or tables
- **Multi-format Support**: Handles different PDF formats and layouts

### Web Interface
- **Modern UI**: Beautiful, responsive design with drag-and-drop functionality
- **Real-time Analysis**: Live processing with progress indicators
- **Interactive Results**: Comprehensive results display with expandable sections
- **Statistics Dashboard**: Overview of extracted data and document metrics

### Document Processing
- **Automotive Repair Estimates**: Specialized parsing for repair line items
- **Insurance Documents**: Extraction of policy numbers, claim information, VINs
- **Vehicle Information**: Year, make, model extraction
- **Cost Summaries**: Labor rates, parts costs, tax calculations
- **Customer Information**: Name and contact details

## 🛠️ Technology Stack

- **Backend**: Python 3.8+, Flask
- **PDF Processing**: pdfplumber (latest version)
- **Data Analysis**: pandas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5.3
- **Icons**: Font Awesome 6.4

## 📋 Requirements

- Python 3.8 or higher
- pip (Python package installer)
- Modern web browser

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd pdf_plumber
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install flask pdfplumber pandas
   ```

4. **Create required directories**:
   ```bash
   mkdir -p uploads output
   ```

## 🎯 Usage

### Starting the Application

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Run the application**:
   ```bash
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5001
   ```

### Using the Web Interface

1. **Upload a PDF**: Drag and drop a PDF file onto the upload area or click to browse
2. **Wait for Analysis**: The application will process the PDF and show progress
3. **Review Results**: Explore the extracted data in the organized sections:
   - Document Summary
   - Detected Tables
   - Page-by-Page Analysis

### API Usage

The application also provides a REST API endpoint:

```bash
curl -X POST -F "file=@your_document.pdf" http://localhost:5001/api/analyze
```

## 📊 How It Works

### 1. Page Cropping (Best Practice)
- Automatically removes 5% margins from each page
- Improves content extraction accuracy
- Reduces noise from page borders

### 2. Text Extraction
- Uses pdfplumber's `extract_words()` with optimal settings
- Groups words into logical lines
- Applies advanced text cleaning algorithms
- Handles character spacing issues

### 3. Table Detection
- Uses pdfplumber's built-in table extraction
- Text-based strategy (no lines required)
- Configurable tolerance settings
- Automatic DataFrame creation

### 4. Section Identification
- Divides pages into header (25%), body (50%), footer (25%)
- Analyzes content patterns to determine section types
- Supports multiple document formats

### 5. Content Parsing
- **Document Headers**: VIN, claim numbers, estimate numbers
- **Vehicle Info**: Year, make, model extraction
- **Repair Items**: Line items with operations and descriptions
- **Cost Summaries**: Labor rates, parts, taxes, totals
- **Customer Info**: Names and contact details

## 🔧 Configuration

### Table Extraction Settings
The application uses optimized table extraction settings:

```python
table_settings = {
    'vertical_strategy': 'text',
    'horizontal_strategy': 'text',
    'min_words_vertical': 1,
    'min_words_horizontal': 1,
    'text_tolerance': 3,
    'text_x_tolerance': 3,
    'text_y_tolerance': 3,
    'join_tolerance': 3,
    'join_x_tolerance': 3,
    'join_y_tolerance': 3,
    'snap_tolerance': 3,
    'snap_x_tolerance': 3,
    'snap_y_tolerance': 3
}
```

### Text Cleaning
Advanced text cleaning handles common PDF issues:
- Character spacing problems
- OCR artifacts
- Control characters
- Excessive whitespace

## 📁 Project Structure

```
pdf_plumber/
├── app.py                 # Main Flask application
├── test_analyzer.py       # Test script for verification
├── templates/
│   └── upload.html        # Modern web interface
├── uploads/               # Uploaded PDF files
├── output/                # Generated output files
├── venv/                  # Python virtual environment
└── README.md             # This file
```

## 🧪 Testing

Run the test script to verify the analyzer works with sample PDFs:

```bash
python test_analyzer.py
```

This will test the analyzer with the provided sample PDFs and show:
- Document information extraction
- Vehicle information parsing
- Repair items detection
- Table extraction results
- Page-by-page analysis

## 🔍 Sample Output

The application provides comprehensive analysis results:

### Document Summary
- Document information (VIN, claim numbers, etc.)
- Vehicle details (year, make, model)
- Cost breakdowns (labor, parts, taxes)
- Customer information

### Tables
- Automatically detected tables
- Cleaned and formatted data
- HTML output for web display

### Page Analysis
- Section-by-section breakdown
- Content type classification
- Raw text for debugging
- Statistics and metrics

## 🎨 UI Features

### Modern Design
- Gradient backgrounds
- Glassmorphism effects
- Smooth animations
- Responsive layout

### Interactive Elements
- Drag-and-drop file upload
- Real-time progress indicators
- Expandable sections
- Search and filter capabilities

### Data Visualization
- Statistics cards
- Progress bars
- Color-coded badges
- Clean table layouts

## 🔧 Customization

### Adding New Document Types
1. Extend the `determine_section_type()` method
2. Add new parsing functions in `parse_section_content()`
3. Update the summary creation logic

### Modifying Table Settings
Adjust the `table_settings` in the `PDFAnalyzer` class to optimize for your specific PDFs.

### Styling Changes
Modify the CSS in `templates/upload.html` to customize the appearance.

## 🐛 Troubleshooting

### Common Issues

1. **Table extraction warnings**: These are normal and don't affect functionality
2. **Character spacing issues**: The text cleaning should handle most cases
3. **Empty sections**: Some PDFs may not have content in all sections

### Debug Information
The application provides detailed debug information:
- Character counts per page
- Table detection results
- Section identification details
- Raw text extraction for troubleshooting

## 📈 Performance

- **Processing Speed**: Typically 1-3 seconds per page
- **Memory Usage**: Efficient processing with minimal memory footprint
- **Scalability**: Can handle multi-page documents efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [pdfplumber](https://github.com/jsvine/pdfplumber) - Excellent PDF processing library
- [Bootstrap](https://getbootstrap.com/) - Modern UI framework
- [Font Awesome](https://fontawesome.com/) - Beautiful icons

---

**PDF Analyzer Pro** - Advanced document processing made simple and elegant. 