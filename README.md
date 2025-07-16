# Document Scanner

A simple Genius Scan clone that uses your webcam to capture and scan documents with automatic edge detection.

## Features

- 📷 Real-time webcam access
- 🔍 Automatic document edge detection using OpenCV.js
- 📸 Document capture and download
- 🎨 Modern, responsive web interface
- 💾 Save documents as JPG files
- 📱 Mobile-friendly design

## How to Use

1. **Start the application**:
   ```bash
   npm install
   npm start
   ```

2. **Open your browser** and go to `http://localhost:3000`

3. **Enable camera access** when prompted

4. **Position a document** in front of your camera

5. **Wait for detection** - the app will automatically detect document edges and show a green outline

6. **Capture the document** by clicking the capture button

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
