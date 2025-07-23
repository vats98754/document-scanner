# Document Scanner

A comprehensive document scanner implementation featuring **from-scratch computer vision algorithms** and real-time camera-based corner detection with colored overlays.

## 🌟 Features

### Real-Time Camera Detection
- **Live Corner Detection**: Real-time Harris corner detection with colored circular overlays
- **Custom Computer Vision**: 100% from-scratch implementations without external CV libraries
- **Interactive Web Interface**: Browser-based camera integration with live processing
- **Document Capture**: One-click document capture with processed results

### Python Analysis Engine
- **Advanced Document Detection**: Detects paper corners within images using multiple edge detection methods
- **Perspective Correction**: Transforms quadrilateral documents into rectangular scans
- **Hyperparameter Tuning**: Comprehensive hyperparameter optimization with 1,024 combinations
- **Quick Testing**: Fast hyperparameter testing with 48 combinations
- **Visualization**: Detailed analysis and visualization of results
- **Modular Design**: Clean separation of concerns with dedicated modules

### From-Scratch Computer Vision Implementations
- **2D Convolution**: Custom convolution operations with kernel support
- **Sobel Edge Detection**: Manual implementation of Sobel operators (Gx, Gy)
- **Gaussian Blur**: Custom Gaussian kernel generation and application
- **Harris Corner Detection**: Complete Harris corner detector with non-maximum suppression
- **Real-Time Processing**: Optimized algorithms for live video processing

## 🔬 From-Scratch Computer Vision Algorithms

This project implements all computer vision algorithms from scratch without relying on external libraries like OpenCV. Here's how our custom implementations work:

### 2D Convolution Engine

```typescript
// Custom 2D convolution with kernel support
static convolve2D(imageData: ImageData, kernel: number[][], stride: number = 1): ImageData {
    // Applies convolution operation using nested loops
    // Supports arbitrary kernel sizes and stride values
    // Handles border conditions with zero-padding
}
```

**Key Features:**
- Pure JavaScript implementation for web compatibility
- Support for arbitrary kernel sizes (3x3, 5x5, etc.)
- Optimized memory access patterns
- Real-time performance for live video processing

### Sobel Edge Detection

```typescript
// Sobel kernels for edge detection
static getSobelKernels(): { x: number[][], y: number[][] } {
    return {
        x: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],  // Horizontal edges
        y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]   // Vertical edges
    };
}
```

**Implementation Details:**
- Separate X and Y gradient computation
- Edge magnitude calculation: `sqrt(Gx² + Gy²)`
- Gradient direction for advanced edge analysis
- Real-time edge visualization as green overlay dots

### Gaussian Blur & Kernel Generation

```typescript
// Dynamic Gaussian kernel generation
static generateGaussianKernel(size: number, sigma: number): number[][] {
    // Mathematical kernel generation: G(x,y) = (1/2πσ²) * e^(-(x²+y²)/2σ²)
    // Automatic normalization for proper convolution
    // Configurable sigma for blur strength control
}
```

**Features:**
- Mathematical precision in kernel computation
- Configurable blur strength via sigma parameter
- Automatic kernel normalization
- Support for various kernel sizes (3x3, 5x5, 7x7)

### Harris Corner Detection

```typescript
// Complete Harris corner detector implementation
static harrisCornerDetection(imageData: ImageData, threshold: number = 0.01): Corner[] {
    // 1. Compute image gradients using Sobel operators
    // 2. Calculate structure tensor components (Ixx, Iyy, Ixy)
    // 3. Apply Gaussian weighting to structure tensor
    // 4. Compute Harris response: R = det(M) - k*trace(M)²
    // 5. Apply threshold and non-maximum suppression
}
```

**Algorithm Steps:**
1. **Gradient Computation**: Custom Sobel operators for Ix, Iy
2. **Structure Tensor**: Second-moment matrix calculation
3. **Gaussian Weighting**: Spatial weighting of gradients
4. **Harris Response**: Mathematical corner strength measure
5. **Non-Maximum Suppression**: Remove redundant corner detections
6. **Color Coding**: Visual representation with colored overlays

### Real-Time Processing Pipeline

```typescript
private detectCorners(): void {
    // 1. Capture video frame to hidden canvas
    const imageData = this.hiddenCtx.getImageData(0, 0, width, height);
    
    // 2. Convert to grayscale (custom implementation)
    const grayImageData = CVUtils.toGrayscale(imageData);
    
    // 3. Apply Gaussian blur (noise reduction)
    const blurredImageData = CVUtils.gaussianBlur(grayImageData, 5, 1.0);
    
    // 4. Detect corners using Harris detector
    const corners = CVUtils.harrisCornerDetection(blurredImageData, 0.01);
    
    // 5. Draw colored overlays on live video
    this.drawColoredCorners(corners);
}
```

### Performance Optimizations

- **Memory Management**: Efficient ImageData manipulation
- **Kernel Caching**: Pre-computed Gaussian kernels for common sizes
- **Spatial Optimization**: Smart pixel sampling for real-time performance
- **Frame Rate Control**: Adaptive processing based on device capabilities

## 🏗️ Project Structure

```
document-scanner/
├── src/
│   ├── document_scanner.py      # Core document scanning functions
│   ├── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── analysis.py              # Result analysis and visualization
│   ├── sobel_kernels.py         # Custom Sobel kernel implementations
│   ├── script.ts                # Real-time web-based corner detection
│   └── server.ts                # Development server
├── test_scanner.py              # Test suite and examples
├── computer-vision.ipynb        # Jupyter notebook with experiments
├── index.html                   # Web interface for camera detection
├── styles.css                   # Web styling
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
├── tsconfig.json               # TypeScript configuration
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🚀 Installation & Setup

### Python Environment
1. Clone the repository:
```bash
git clone <repository-url>
cd document-scanner
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Web Interface Setup
1. Install Node.js dependencies:
```bash
npm install
```

2. Compile TypeScript:
```bash
npx tsc
```

3. Start local server:
```bash
python3 -m http.server 8000
```

4. Open browser and navigate to:
```
http://localhost:8000
```

## 📱 Usage

### Real-Time Camera Detection

1. **Start the Web Interface**:
   - Open `index.html` in a web browser (or use the local server)
   - Click "📷 Start Camera" to enable webcam access

2. **Live Corner Detection**:
   - Position a document or object in front of the camera
   - Observe real-time colored corner detection overlays:
     - 🔴 Red circles: Primary corners
     - 🟢 Green circles: Secondary corners  
     - 🔵 Blue circles: Additional feature points
     - 🟡 Yellow circles: Edge intersections
   - Corner response strength shown as circle radius
   - Live edge detection shown as green dots

3. **Capture Documents**:
   - Click "📸 Capture Document" to save current frame
   - Images saved with detected features highlighted
   - Download captured documents for further processing

### Python Document Analysis

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
results, best = quick_hyperparameter_tuning("path/to/document.jpg")

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
