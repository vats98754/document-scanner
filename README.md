# Document Scanner

A Python-based document scanner using OpenCV that detects document corners and applies perspective correction with hyperparameter tuning capabilities.

## Features

- **Advanced Document Detection**: Detects paper corners within images using multiple edge detection methods
- **Perspective Correction**: Transforms quadrilateral documents into rectangular scans
- **Hyperparameter Tuning**: Comprehensive hyperparameter optimization with 1,024 combinations
- **Quick Testing**: Fast hyperparameter testing with 48 combinations
- **Visualization**: Detailed analysis and visualization of results
- **Modular Design**: Clean separation of concerns with dedicated modules

## Project Structure

```
document-scanner/
├── src/
│   ├── document_scanner.py     # Core document scanning functions
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   └── analysis.py             # Result analysis and visualization
├── test_scanner.py             # Test suite and examples
├── computer-vision.ipynb       # Jupyter notebook with experiments
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Document Scanning

```python
from src.document_scanner import test_scanner

# Test document scanner on an image
image_path = "path/to/your/document.jpg"
original, corners_viz, scanned = test_scanner(image_path)
```

### Hyperparameter Tuning

```python
from src.hyperparameter_tuning import hyperparameter_tuning, quick_hyperparameter_test

# Quick test (48 combinations)
results, best = quick_hyperparameter_test("path/to/document.jpg")

# Full hyperparameter tuning (1,024 combinations)
results, best = hyperparameter_tuning("path/to/document.jpg")
```

### Analysis and Visualization

```python
from src.analysis import analyze_results, visualize_top_results

# Analyze results
sorted_results = analyze_results("hyperparameter_results")

# Visualize top performing combinations
visualize_top_results("hyperparameter_results", top_n=6)
```

### Running the Test Suite

```bash
python test_scanner.py
```

## Hyperparameter Tuning

The system tests the following parameters:

- **Blur Kernel**: [3, 5, 7, 9] - Gaussian blur kernel sizes
- **Canny Low**: [30, 50, 70, 100] - Lower Canny threshold
- **Canny High**: [100, 150, 200, 250] - Upper Canny threshold  
- **Epsilon Factor**: [0.01, 0.02, 0.03, 0.05] - Contour approximation factor
- **Min Area**: [500, 1000, 2000, 5000] - Minimum area threshold

### Results Organization

Results are saved in organized directory structures:

```
hyperparameter_results/
├── blur5_canny50-150_eps0.02_area1000/
│   ├── original.jpg
│   ├── edges.jpg
│   ├── contours.jpg
│   ├── blurred.jpg
│   └── results.json
├── hyperparameter_summary.json
├── parameter_effects.png
└── top_results_visualization.png
```

## Key Functions

### Core Document Scanner
- `document_scanner()`: Main scanning function with perspective correction
- `find_edges()`: Simple edge detection
- `order_corners()`: Orders corner points correctly
- `test_scanner()`: Test function with visualization

### Hyperparameter Optimization
- `hyperparameter_tuning()`: Full hyperparameter optimization
- `quick_hyperparameter_test()`: Fast testing with subset of parameters
- `document_scanner_with_hyperparams()`: Configurable scanner function

### Analysis and Visualization
- `analyze_results()`: Comprehensive result analysis
- `visualize_top_results()`: Visualization of best performing combinations
- `visualize_quick_results()`: Quick test result visualization
- `compare_hyperparameter_effects()`: Detailed parameter effect analysis

## Git Ignore Configuration

The `.gitignore` file is configured to:
- Ignore all hyperparameter result directories
- Keep only summary files: `hyperparameter_summary.json`, `parameter_effects.png`, `*_visualization.png`
- Standard Python, Jupyter, and IDE ignore patterns

## Dependencies

- OpenCV (cv2) - Computer vision operations
- NumPy - Numerical computations
- Matplotlib - Plotting and visualization
- itertools - Parameter combination generation
- json - Result serialization
- os - File system operations

## License

This project is open source and available under the MIT License.

7. **Download your scanned documents** from the results section

## Requirements

- Node.js (v14 or higher)
- A modern web browser with webcam support
- Camera permissions enabled

## Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Computer Vision**: OpenCV.js for document detection
- **Backend**: Node.js with Express.js
- **Camera API**: WebRTC getUserMedia API

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+

## Tips for Best Results

- Ensure good lighting
- Use a contrasting background (dark document on light surface or vice versa)
- Keep the document flat and unfolded
- Maintain steady hands during capture
- Position the entire document within the camera view

## Development

To run in development mode:

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## License

MIT License - feel free to use and modify as needed!
