class DocumentScanner {
    constructor() {
        this.video = document.getElementById('video');
        this.overlayCanvas = document.getElementById('overlay-canvas');
        this.captureCanvas = document.getElementById('capture-canvas');
        this.overlayCtx = this.overlayCanvas.getContext('2d');
        this.captureCtx = this.captureCanvas.getContext('2d');
        
        this.startCameraBtn = document.getElementById('start-camera');
        this.captureBtn = document.getElementById('capture');
        this.toggleDetectionBtn = document.getElementById('toggle-detection');
        this.resultsContainer = document.getElementById('results-container');
        
        this.stream = null;
        this.detectionEnabled = true;
        this.isProcessing = false;
        this.documentCount = 0;
        
        this.initializeEventListeners();
        this.waitForOpenCV();
    }
    
    waitForOpenCV() {
        if (typeof cv !== 'undefined') {
            console.log('OpenCV.js is ready');
            this.showMessage('OpenCV.js loaded successfully!', 'success');
        } else {
            console.log('Waiting for OpenCV.js...');
            setTimeout(() => this.waitForOpenCV(), 100);
        }
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
            
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'environment' // Prefer back camera on mobile
                }
            });
            
            this.video.srcObject = this.stream;
            this.startCameraBtn.disabled = true;
            this.captureBtn.disabled = false;
            this.toggleDetectionBtn.disabled = false;
            
            this.showMessage('Camera started successfully!', 'success');
            
        } catch (error) {
            console.error('Error accessing camera:', error);
            this.showMessage('Error accessing camera. Please check permissions.', 'error');
        }
    }
    
    setupCanvases() {
        const rect = this.video.getBoundingClientRect();
        
        // Set overlay canvas size to match video display
        this.overlayCanvas.width = this.video.videoWidth;
        this.overlayCanvas.height = this.video.videoHeight;
        
        // Set capture canvas size to match video resolution
        this.captureCanvas.width = this.video.videoWidth;
        this.captureCanvas.height = this.video.videoHeight;
    }
    
    startDetectionLoop() {
        const detect = () => {
            if (this.video.readyState === 4 && this.detectionEnabled && !this.isProcessing) {
                this.detectDocument();
            }
            requestAnimationFrame(detect);
        };
        detect();
    }
    
    detectDocument() {
        if (typeof cv === 'undefined') return;
        
        try {
            // Clear overlay
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
            
            // Create OpenCV Mat from video
            const src = new cv.Mat(this.video.videoHeight, this.video.videoWidth, cv.CV_8UC4);
            const cap = new cv.VideoCapture(this.video);
            cap.read(src);
            
            // Convert to grayscale
            const gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            
            // Apply Gaussian blur
            const blurred = new cv.Mat();
            cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);
            
            // Edge detection
            const edges = new cv.Mat();
            cv.Canny(blurred, edges, 50, 150);
            
            // Find contours
            const contours = new cv.MatVector();
            const hierarchy = new cv.Mat();
            cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            
            // Find the largest rectangular contour
            let maxArea = 0;
            let bestContour = null;
            
            for (let i = 0; i < contours.size(); i++) {
                const contour = contours.get(i);
                const area = cv.contourArea(contour);
                
                if (area > maxArea && area > 10000) { // Minimum area threshold
                    // Approximate contour to polygon
                    const approx = new cv.Mat();
                    const epsilon = 0.02 * cv.arcLength(contour, true);
                    cv.approxPolyDP(contour, approx, epsilon, true);
                    
                    // Check if it's roughly rectangular (4 points)
                    if (approx.rows === 4) {
                        maxArea = area;
                        if (bestContour) bestContour.delete();
                        bestContour = approx.clone();
                    }
                    approx.delete();
                }
                contour.delete();
            }
            
            // Draw the detected document outline
            if (bestContour) {
                this.drawContour(bestContour);
                bestContour.delete();
            }
            
            // Cleanup
            src.delete();
            gray.delete();
            blurred.delete();
            edges.delete();
            contours.delete();
            hierarchy.delete();
            
        } catch (error) {
            console.error('Detection error:', error);
        }
    }
    
    drawContour(contour) {
        const points = [];
        for (let i = 0; i < contour.rows; i++) {
            const point = {
                x: contour.data32S[i * 2],
                y: contour.data32S[i * 2 + 1]
            };
            points.push(point);
        }
        
        if (points.length === 4) {
            this.overlayCtx.strokeStyle = '#00ff00';
            this.overlayCtx.lineWidth = 3;
            this.overlayCtx.beginPath();
            
            points.forEach((point, index) => {
                if (index === 0) {
                    this.overlayCtx.moveTo(point.x, point.y);
                } else {
                    this.overlayCtx.lineTo(point.x, point.y);
                }
            });
            
            this.overlayCtx.closePath();
            this.overlayCtx.stroke();
            
            // Draw corner circles
            this.overlayCtx.fillStyle = '#00ff00';
            points.forEach(point => {
                this.overlayCtx.beginPath();
                this.overlayCtx.arc(point.x, point.y, 8, 0, 2 * Math.PI);
                this.overlayCtx.fill();
            });
        }
    }
    
    captureDocument() {
        if (!this.video.videoWidth || !this.video.videoHeight) return;
        
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
        
        this.showMessage(
            this.detectionEnabled ? 'Document detection enabled' : 'Document detection disabled',
            'info'
        );
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
}

// Initialize the scanner when the page loads
let scanner;
document.addEventListener('DOMContentLoaded', () => {
    scanner = new DocumentScanner();
});

// Handle page unload to clean up camera stream
window.addEventListener('beforeunload', () => {
    if (scanner && scanner.stream) {
        scanner.stream.getTracks().forEach(track => track.stop());
    }
});
