"use strict";
// From-scratch computer vision utilities
class CVUtils {
    // Custom 2D convolution implementation
    static convolve2D(imageData, kernel, stride = 1) {
        const { width, height, data } = imageData;
        const kernelSize = kernel.length;
        const halfKernel = Math.floor(kernelSize / 2);
        const result = new ImageData(width, height);
        for (let y = halfKernel; y < height - halfKernel; y += stride) {
            for (let x = halfKernel; x < width - halfKernel; x += stride) {
                let sum = 0;
                // Apply kernel
                for (let ky = 0; ky < kernelSize; ky++) {
                    for (let kx = 0; kx < kernelSize; kx++) {
                        const px = x + kx - halfKernel;
                        const py = y + ky - halfKernel;
                        const pixelIndex = (py * width + px) * 4;
                        // Use grayscale value (assuming already converted)
                        const pixelValue = data[pixelIndex];
                        sum += pixelValue * kernel[ky][kx];
                    }
                }
                const index = (y * width + x) * 4;
                const clampedValue = Math.max(0, Math.min(255, Math.abs(sum)));
                result.data[index] = clampedValue; // R
                result.data[index + 1] = clampedValue; // G
                result.data[index + 2] = clampedValue; // B
                result.data[index + 3] = 255; // A
            }
        }
        return result;
    }
    // Convert RGB to grayscale
    static toGrayscale(imageData) {
        const { width, height, data } = imageData;
        const result = new ImageData(width, height);
        for (let i = 0; i < data.length; i += 4) {
            const gray = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
            result.data[i] = gray; // R
            result.data[i + 1] = gray; // G
            result.data[i + 2] = gray; // B
            result.data[i + 3] = 255; // A
        }
        return result;
    }
    // Gaussian kernel generation
    static generateGaussianKernel(size, sigma) {
        const kernel = [];
        const halfSize = Math.floor(size / 2);
        let sum = 0;
        for (let y = -halfSize; y <= halfSize; y++) {
            const row = [];
            for (let x = -halfSize; x <= halfSize; x++) {
                const value = Math.exp(-(x * x + y * y) / (2 * sigma * sigma));
                row.push(value);
                sum += value;
            }
            kernel.push(row);
        }
        // Normalize
        for (let y = 0; y < kernel.length; y++) {
            for (let x = 0; x < kernel[y].length; x++) {
                kernel[y][x] /= sum;
            }
        }
        return kernel;
    }
    // Sobel edge detection kernels
    static getSobelKernels() {
        return {
            x: [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            y: [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]
        };
    }
    // Sobel edge magnitude
    static sobelEdgeDetection(imageData) {
        const { x: sobelX, y: sobelY } = this.getSobelKernels();
        const gradX = this.convolve2D(imageData, sobelX);
        const gradY = this.convolve2D(imageData, sobelY);
        const { width, height } = imageData;
        const result = new ImageData(width, height);
        for (let i = 0; i < gradX.data.length; i += 4) {
            const gx = gradX.data[i];
            const gy = gradY.data[i];
            const magnitude = Math.sqrt(gx * gx + gy * gy);
            const clampedMag = Math.min(255, magnitude);
            result.data[i] = clampedMag; // R
            result.data[i + 1] = clampedMag; // G
            result.data[i + 2] = clampedMag; // B
            result.data[i + 3] = 255; // A
        }
        return result;
    }
    // Harris corner detection with configurable parameters
    static harrisCornerDetection(imageData, threshold = 0.01, k = 0.04, nmsRadius = 15) {
        const { width, height } = imageData;
        const gray = this.toGrayscale(imageData);
        const { x: sobelX, y: sobelY } = this.getSobelKernels();
        // Compute gradients
        const gradX = this.convolve2D(gray, sobelX);
        const gradY = this.convolve2D(gray, sobelY);
        // Gaussian kernel for smoothing (larger kernel for better results)
        const gaussianKernel = this.generateGaussianKernel(7, 1.5);
        // Compute Harris response
        const corners = [];
        const windowSize = 3; // Increased window size
        for (let y = windowSize; y < height - windowSize; y++) {
            for (let x = windowSize; x < width - windowSize; x++) {
                // Compute structure tensor components
                let Ixx = 0, Iyy = 0, Ixy = 0;
                for (let wy = -windowSize; wy <= windowSize; wy++) {
                    for (let wx = -windowSize; wx <= windowSize; wx++) {
                        const px = x + wx;
                        const py = y + wy;
                        const idx = (py * width + px) * 4;
                        const ix = gradX.data[idx] / 255;
                        const iy = gradY.data[idx] / 255;
                        const weight = gaussianKernel[wy + windowSize][wx + windowSize];
                        Ixx += weight * ix * ix;
                        Iyy += weight * iy * iy;
                        Ixy += weight * ix * iy;
                    }
                }
                // Harris response
                const det = Ixx * Iyy - Ixy * Ixy;
                const trace = Ixx + Iyy;
                const response = det - k * trace * trace;
                if (response > threshold) {
                    corners.push({ x, y, response });
                }
            }
        }
        // Non-maximum suppression with reduced radius for more corners
        return this.nonMaximumSuppression(corners, 15);
    }
    // Non-maximum suppression for corners
    static nonMaximumSuppression(corners, radius) {
        const filtered = [];
        const sortedCorners = [...corners].sort((a, b) => b.response - a.response);
        for (const corner of sortedCorners) {
            let isMaximum = true;
            for (const existing of filtered) {
                const dist = Math.sqrt(Math.pow(corner.x - existing.x, 2) +
                    Math.pow(corner.y - existing.y, 2));
                if (dist < radius) {
                    isMaximum = false;
                    break;
                }
            }
            if (isMaximum) {
                filtered.push(corner);
            }
        }
        return filtered;
    }
    // Gaussian blur using custom convolution
    static gaussianBlur(imageData, kernelSize = 5, sigma = 1.0) {
        const kernel = this.generateGaussianKernel(kernelSize, sigma);
        return this.convolve2D(imageData, kernel);
    }
}
class DocumentScanner {
    constructor() {
        this.stream = null;
        this.detectionEnabled = true;
        this.isProcessing = false;
        this.documentCount = 0;
        // CV Parameters
        this.cvParams = {
            harris: { threshold: 0.005, k: 0.04, nmsRadius: 15 },
            blur: { kernelSize: 5, sigma: 1.0 },
            edge: { threshold: 80, sampleRate: 3 },
            display: { showEdges: true, showCorners: true, cornerSize: 12 }
        };
        // Performance tracking
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.fps = 0;
        this.video = document.getElementById('video');
        this.overlayCanvas = document.getElementById('overlay-canvas');
        this.captureCanvas = document.getElementById('capture-canvas');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.captureCtx = this.captureCanvas.getContext('2d');
        // Create hidden canvas for image processing
        this.hiddenCanvas = document.createElement('canvas');
        this.hiddenCtx = this.hiddenCanvas.getContext('2d');
        this.startCameraBtn = document.getElementById('start-camera');
        this.captureBtn = document.getElementById('capture');
        this.toggleDetectionBtn = document.getElementById('toggle-detection');
        this.resultsContainer = document.getElementById('results-container');
        this.initializeEventListeners();
        this.initializeParameterControls();
        this.showMessage('Document scanner ready! Custom computer vision algorithms loaded.', 'success');
    }
    initializeEventListeners() {
        this.startCameraBtn.addEventListener('click', () => this.startCamera());
        this.captureBtn.addEventListener('click', () => this.captureDocument());
        this.toggleDetectionBtn.addEventListener('click', () => this.toggleDetection());
        // Handle video metadata loaded
        this.video.addEventListener('loadedmetadata', () => {
            this.setupCanvases();
            this.startDetectionLoop();
        });
    }
    async startCamera() {
        try {
            this.showMessage('Requesting camera access...', 'info');
            const constraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Prefer back camera on mobile
                }
            };
            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;
            this.startCameraBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.toggleDetectionBtn.disabled = false;
            // Show parameter controls
            const parametersDiv = document.getElementById('cv-parameters');
            if (parametersDiv) {
                parametersDiv.style.display = 'block';
            }
            this.showMessage('Camera started successfully!', 'success');
        }
        catch (error) {
            console.error('Error accessing camera:', error);
            this.showMessage('Error accessing camera. Please check permissions.', 'error');
        }
    }
    setupCanvases() {
        // Set overlay canvas size to match video display
        this.overlayCanvas.width = this.video.videoWidth;
        this.overlayCanvas.height = this.video.videoHeight;
        // Set capture canvas size to match video resolution
        this.captureCanvas.width = this.video.videoWidth;
        this.captureCanvas.height = this.video.videoHeight;
        // Set hidden canvas for processing
        this.hiddenCanvas.width = this.video.videoWidth;
        this.hiddenCanvas.height = this.video.videoHeight;
    }
    startDetectionLoop() {
        const detect = () => {
            if (this.video.readyState === 4 && this.detectionEnabled && !this.isProcessing) {
                this.detectCorners();
            }
            requestAnimationFrame(detect);
        };
        detect();
    }
    detectCorners() {
        try {
            // Performance tracking
            const startTime = performance.now();
            this.frameCount++;
            // Clear overlay
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
            // Draw video frame to hidden canvas for processing
            this.hiddenCtx.drawImage(this.video, 0, 0, this.hiddenCanvas.width, this.hiddenCanvas.height);
            // Get image data
            const imageData = this.hiddenCtx.getImageData(0, 0, this.hiddenCanvas.width, this.hiddenCanvas.height);
            // Convert to grayscale
            const grayImageData = CVUtils.toGrayscale(imageData);
            // Apply Gaussian blur to reduce noise (using configurable parameters)
            const blurredImageData = CVUtils.gaussianBlur(grayImageData, this.cvParams.blur.kernelSize, this.cvParams.blur.sigma);
            // Detect corners using Harris corner detection with configurable parameters
            const corners = CVUtils.harrisCornerDetection(blurredImageData, this.cvParams.harris.threshold, this.cvParams.harris.k, this.cvParams.harris.nmsRadius);
            // Draw colored corners on overlay (if enabled)
            if (this.cvParams.display.showCorners) {
                this.drawColoredCorners(corners);
            }
            // Detect edges for document outline (if enabled)
            if (this.cvParams.display.showEdges) {
                const edges = CVUtils.sobelEdgeDetection(blurredImageData);
                this.drawDocumentOutline(edges);
            }
            // Update performance info
            const endTime = performance.now();
            const processingTime = endTime - startTime;
            this.updatePerformanceInfo(processingTime, corners.length);
        }
        catch (error) {
            console.error('Corner detection error:', error);
        }
    }
    drawColoredCorners(corners) {
        // Brighter, more contrasting color palette for corners
        const colors = ['#FF0000', '#00FF00', '#0080FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFA500', '#FF69B4'];
        const cornerSize = this.cvParams.display.cornerSize;
        corners.forEach((corner, index) => {
            const color = colors[index % colors.length];
            // Draw larger corner circle with black outline for better visibility
            this.overlayCtx.fillStyle = color;
            this.overlayCtx.strokeStyle = '#000000';
            this.overlayCtx.lineWidth = 3;
            this.overlayCtx.beginPath();
            this.overlayCtx.arc(corner.x, corner.y, cornerSize, 0, 2 * Math.PI);
            this.overlayCtx.fill();
            this.overlayCtx.stroke();
            // Draw inner white circle for contrast
            this.overlayCtx.fillStyle = '#FFFFFF';
            this.overlayCtx.beginPath();
            this.overlayCtx.arc(corner.x, corner.y, cornerSize * 0.33, 0, 2 * Math.PI);
            this.overlayCtx.fill();
            // Draw corner response indicator (pulsing outer ring)
            const responseRadius = Math.max(cornerSize + 3, Math.min(cornerSize + 13, corner.response * 2000));
            this.overlayCtx.beginPath();
            this.overlayCtx.arc(corner.x, corner.y, responseRadius, 0, 2 * Math.PI);
            this.overlayCtx.strokeStyle = color;
            this.overlayCtx.lineWidth = 2;
            this.overlayCtx.setLineDash([5, 5]);
            this.overlayCtx.stroke();
            this.overlayCtx.setLineDash([]);
            // Label corner with index (larger, bold text with outline)
            this.overlayCtx.fillStyle = '#000000';
            this.overlayCtx.strokeStyle = '#FFFFFF';
            this.overlayCtx.font = `bold ${Math.max(12, cornerSize)}px Arial`;
            this.overlayCtx.textAlign = 'center';
            this.overlayCtx.lineWidth = 3;
            this.overlayCtx.strokeText((index + 1).toString(), corner.x, corner.y - cornerSize - 8);
            this.overlayCtx.fillText((index + 1).toString(), corner.x, corner.y - cornerSize - 8);
        });
        // Display corner count with better styling
        this.overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.overlayCtx.fillRect(10, 10, 220, 40);
        this.overlayCtx.strokeStyle = '#FFFFFF';
        this.overlayCtx.lineWidth = 2;
        this.overlayCtx.strokeRect(10, 10, 220, 40);
        this.overlayCtx.fillStyle = '#FFFFFF';
        this.overlayCtx.font = 'bold 18px Arial';
        this.overlayCtx.textAlign = 'left';
        this.overlayCtx.fillText(`Corners detected: ${corners.length}`, 20, 35);
    }
    drawDocumentOutline(edgeImageData) {
        const { width, height, data } = edgeImageData;
        // Use configurable parameters
        const threshold = this.cvParams.edge.threshold;
        const sampleRate = this.cvParams.edge.sampleRate;
        const outlinePoints = [];
        // Sample edge points with configurable density
        for (let y = 0; y < height; y += sampleRate) {
            for (let x = 0; x < width; x += sampleRate) {
                const index = (y * width + x) * 4;
                const edgeStrength = data[index];
                if (edgeStrength > threshold) {
                    outlinePoints.push({ x, y });
                }
            }
        }
        // Draw edge points with better visibility
        if (outlinePoints.length > 0) {
            // Draw larger, more visible edge points
            this.overlayCtx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            this.overlayCtx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
            this.overlayCtx.lineWidth = 1;
            outlinePoints.forEach(point => {
                this.overlayCtx.beginPath();
                this.overlayCtx.arc(point.x, point.y, 1.5, 0, 2 * Math.PI);
                this.overlayCtx.fill();
                this.overlayCtx.stroke();
            });
            // Add edge count info with current parameters
            this.overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.overlayCtx.fillRect(10, 60, 250, 50);
            this.overlayCtx.strokeStyle = '#00FF00';
            this.overlayCtx.lineWidth = 2;
            this.overlayCtx.strokeRect(10, 60, 250, 50);
            this.overlayCtx.fillStyle = '#00FF00';
            this.overlayCtx.font = 'bold 14px Arial';
            this.overlayCtx.textAlign = 'left';
            this.overlayCtx.fillText(`Edges: ${outlinePoints.length}`, 20, 80);
            this.overlayCtx.fillText(`Threshold: ${threshold} | Rate: ${sampleRate}`, 20, 100);
        }
    }
    captureDocument() {
        if (!this.video.videoWidth || !this.video.videoHeight)
            return;
        this.isProcessing = true;
        this.captureBtn.disabled = true;
        this.showMessage('Capturing document...', 'info');
        // Draw current video frame to capture canvas
        this.captureCtx.drawImage(this.video, 0, 0);
        // Get image data
        const imageData = this.captureCanvas.toDataURL('image/jpeg', 0.9);
        // Create document item
        this.createDocumentItem(imageData);
        this.isProcessing = false;
        this.captureBtn.disabled = false;
        this.showMessage('Document captured successfully!', 'success');
    }
    createDocumentItem(imageData) {
        this.documentCount++;
        // Remove "no documents" message if it exists
        const noDocsMsg = this.resultsContainer.querySelector('.no-documents');
        if (noDocsMsg) {
            noDocsMsg.remove();
        }
        const documentItem = document.createElement('div');
        documentItem.className = 'document-item';
        documentItem.innerHTML = `
            <h4>Document #${this.documentCount}</h4>
            <img src="${imageData}" alt="Captured Document">
            <div class="document-actions">
                <button class="btn btn-primary btn-small" onclick="scanner.downloadDocument('${imageData}', ${this.documentCount})">
                    💾 Download
                </button>
                <button class="btn btn-secondary btn-small" onclick="scanner.deleteDocument(this.parentElement.parentElement)">
                    🗑️ Delete
                </button>
            </div>
            <small>Captured: ${new Date().toLocaleString()}</small>
        `;
        this.resultsContainer.insertBefore(documentItem, this.resultsContainer.firstChild);
    }
    downloadDocument(imageData, documentNumber) {
        const link = document.createElement('a');
        link.href = imageData;
        link.download = `document_${documentNumber}_${Date.now()}.jpg`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    deleteDocument(documentElement) {
        documentElement.remove();
        // Show "no documents" message if no documents left
        if (this.resultsContainer.children.length === 0) {
            this.resultsContainer.innerHTML = '<p class="no-documents">No documents captured yet. Start by enabling your camera and capturing a document!</p>';
        }
    }
    toggleDetection() {
        this.detectionEnabled = !this.detectionEnabled;
        this.toggleDetectionBtn.textContent = this.detectionEnabled ? '🔍 Disable Detection' : '🔍 Enable Detection';
        if (!this.detectionEnabled) {
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }
        this.showMessage(this.detectionEnabled ? 'Document detection enabled' : 'Document detection disabled', 'info');
    }
    showMessage(message, type = 'info') {
        // Remove existing messages
        const existingMessages = document.querySelectorAll('.status-message, .error-message');
        existingMessages.forEach(msg => msg.remove());
        const messageElement = document.createElement('div');
        messageElement.className = type === 'error' ? 'error-message' : 'status-message';
        messageElement.textContent = message;
        const container = document.querySelector('.camera-section');
        container.insertBefore(messageElement, container.firstChild);
        // Auto-remove after 3 seconds
        setTimeout(() => {
            if (messageElement.parentNode) {
                messageElement.remove();
            }
        }, 3000);
    }
    initializeParameterControls() {
        // Initialize all parameter controls
        this.setupParameterSlider('harris-threshold', 'harris', 'threshold');
        this.setupParameterSlider('harris-k', 'harris', 'k');
        this.setupParameterSlider('nms-radius', 'harris', 'nmsRadius');
        this.setupParameterSlider('blur-kernel-size', 'blur', 'kernelSize');
        this.setupParameterSlider('blur-sigma', 'blur', 'sigma');
        this.setupParameterSlider('edge-threshold', 'edge', 'threshold');
        this.setupParameterSlider('edge-sample-rate', 'edge', 'sampleRate');
        this.setupParameterSlider('corner-size', 'display', 'cornerSize');
        // Setup checkboxes
        this.setupParameterCheckbox('show-edges', 'display', 'showEdges');
        this.setupParameterCheckbox('show-corners', 'display', 'showCorners');
        // Setup action buttons
        const resetBtn = document.getElementById('reset-params');
        const saveBtn = document.getElementById('save-params');
        resetBtn?.addEventListener('click', () => this.resetParameters());
        saveBtn?.addEventListener('click', () => this.saveParameters());
        // Load saved parameters
        this.loadParameters();
    }
    setupParameterSlider(elementId, category, param) {
        const slider = document.getElementById(elementId);
        const valueDisplay = document.getElementById(`${elementId}-value`);
        if (!slider || !valueDisplay)
            return;
        slider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            this.cvParams[category][param] = value;
            valueDisplay.textContent = value.toString();
        });
        // Set initial value
        const initialValue = this.cvParams[category][param];
        slider.value = initialValue.toString();
        valueDisplay.textContent = initialValue.toString();
    }
    setupParameterCheckbox(elementId, category, param) {
        const checkbox = document.getElementById(elementId);
        if (!checkbox)
            return;
        checkbox.addEventListener('change', (e) => {
            const checked = e.target.checked;
            this.cvParams[category][param] = checked;
        });
        // Set initial value
        const initialValue = this.cvParams[category][param];
        checkbox.checked = initialValue;
    }
    resetParameters() {
        // Reset to default values
        this.cvParams = {
            harris: { threshold: 0.005, k: 0.04, nmsRadius: 15 },
            blur: { kernelSize: 5, sigma: 1.0 },
            edge: { threshold: 80, sampleRate: 3 },
            display: { showEdges: true, showCorners: true, cornerSize: 12 }
        };
        // Update all controls
        this.updateAllControls();
        this.showMessage('Parameters reset to defaults', 'info');
    }
    saveParameters() {
        localStorage.setItem('cv-parameters', JSON.stringify(this.cvParams));
        this.showMessage('Parameters saved successfully', 'success');
    }
    loadParameters() {
        const saved = localStorage.getItem('cv-parameters');
        if (saved) {
            try {
                this.cvParams = { ...this.cvParams, ...JSON.parse(saved) };
                this.updateAllControls();
                this.showMessage('Saved parameters loaded', 'info');
            }
            catch (error) {
                console.error('Error loading saved parameters:', error);
            }
        }
    }
    updateAllControls() {
        // Update sliders
        const sliderUpdates = [
            { id: 'harris-threshold', value: this.cvParams.harris.threshold },
            { id: 'harris-k', value: this.cvParams.harris.k },
            { id: 'nms-radius', value: this.cvParams.harris.nmsRadius },
            { id: 'blur-kernel-size', value: this.cvParams.blur.kernelSize },
            { id: 'blur-sigma', value: this.cvParams.blur.sigma },
            { id: 'edge-threshold', value: this.cvParams.edge.threshold },
            { id: 'edge-sample-rate', value: this.cvParams.edge.sampleRate },
            { id: 'corner-size', value: this.cvParams.display.cornerSize }
        ];
        sliderUpdates.forEach(({ id, value }) => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(`${id}-value`);
            if (slider && valueDisplay) {
                slider.value = value.toString();
                valueDisplay.textContent = value.toString();
            }
        });
        // Update checkboxes
        const checkboxUpdates = [
            { id: 'show-edges', checked: this.cvParams.display.showEdges },
            { id: 'show-corners', checked: this.cvParams.display.showCorners }
        ];
        checkboxUpdates.forEach(({ id, checked }) => {
            const checkbox = document.getElementById(id);
            if (checkbox) {
                checkbox.checked = checked;
            }
        });
    }
    updatePerformanceInfo(processingTime, cornerCount) {
        // Calculate FPS
        const currentTime = performance.now();
        if (currentTime - this.lastFrameTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFrameTime = currentTime;
            // Update performance display if it exists
            this.updatePerformanceDisplay(processingTime, cornerCount);
        }
    }
    updatePerformanceDisplay(processingTime, cornerCount) {
        // Create or update performance info display
        let perfDiv = document.getElementById('performance-info');
        if (!perfDiv) {
            perfDiv = document.createElement('div');
            perfDiv.id = 'performance-info';
            perfDiv.className = 'performance-info';
            perfDiv.innerHTML = `
                <h4>⚡ Performance & Detection Stats</h4>
                <div class="performance-stats">
                    <div class="stat-item">
                        <span class="stat-label">FPS</span>
                        <span class="stat-value" id="fps-value">--</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Processing</span>
                        <span class="stat-value" id="processing-time">--</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Corners</span>
                        <span class="stat-value" id="corner-count">--</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Threshold</span>
                        <span class="stat-value" id="current-threshold">--</span>
                    </div>
                </div>
            `;
            const parametersDiv = document.getElementById('cv-parameters');
            if (parametersDiv) {
                parametersDiv.appendChild(perfDiv);
            }
        }
        // Update values
        const fpsElement = document.getElementById('fps-value');
        const processingElement = document.getElementById('processing-time');
        const cornerCountElement = document.getElementById('corner-count');
        const thresholdElement = document.getElementById('current-threshold');
        if (fpsElement)
            fpsElement.textContent = this.fps.toString();
        if (processingElement)
            processingElement.textContent = `${processingTime.toFixed(1)}ms`;
        if (cornerCountElement)
            cornerCountElement.textContent = cornerCount.toString();
        if (thresholdElement)
            thresholdElement.textContent = this.cvParams.harris.threshold.toFixed(3);
    }
}
// Initialize the scanner when the page loads
let scanner;
document.addEventListener('DOMContentLoaded', () => {
    scanner = new DocumentScanner();
});
// Handle page unload to clean up camera stream
window.addEventListener('beforeunload', () => {
    scanner?.stream?.getTracks().forEach(track => track.stop());
});
//# sourceMappingURL=script.js.map