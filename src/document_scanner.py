import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_edges(img):
    """
    Simple edge detection function
    
    Args:
        img: Input image
        
    Returns:
        gray_img, img_blur, edge_img: Processed images
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)
    edge_img = cv2.Canny(img, 100, 200)
    edge_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
    return gray_img, img_blur, edge_img


def order_corners(pts):
    """
    Order points: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array of corner points
        
    Returns:
        Ordered corner points
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum and difference of coordinates
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # Top-left (smallest sum)
    rect[2] = pts[np.argmax(s)]      # Bottom-right (largest sum)
    rect[1] = pts[np.argmin(diff)]   # Top-right (smallest diff)
    rect[3] = pts[np.argmax(diff)]   # Bottom-left (largest diff)
    
    return rect


def distance(p1, p2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        p1, p2: Points as [x, y] arrays
        
    Returns:
        Distance between points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def document_scanner(image, debug=False):
    """
    Document scanner that detects paper corners within the image.
    
    Args:
        image: Input color image containing a document
        debug: If True, shows intermediate processing steps
        
    Returns:
        Tuple of (original, corners_visualization, scanned_document)
    """
    original = image.copy()
    h, w = image.shape[:2]
    
    # Resize for processing if too large
    if h > 1000:
        scale = 1000 / h
        image = cv2.resize(image, None, fx=scale, fy=scale)
        h, w = image.shape[:2]
    
    # STEP 1: Preprocessing to enhance document edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # STEP 2: Enhanced edge detection for document boundaries
    # Use adaptive threshold to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 10
    )
    
    # Canny edge detection with optimized parameters
    edges = cv2.Canny(filtered, 30, 80, apertureSize=3)
    
    # Combine both edge detection methods
    combined_edges = cv2.bitwise_or(edges, adaptive_thresh)
    
    # Morphological operations to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
    combined_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    
    # STEP 3: Find document contour
    contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    document_contour = None
    
    # Look for the largest rectangular contour (the document)
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Skip very small contours (noise)
        if area < w * h * 0.01:
            continue
            
        # Approximate contour to polygon
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Look for quadrilateral (4 corners)
        if len(approx) == 4:
            # Check if it's a reasonable size (not too small, not the entire image)
            if w * h * 0.05 < area < w * h * 0.95:
                document_contour = approx
                break
    
    # If no good 4-corner contour found, try with more relaxed parameters
    if document_contour is None:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < w * h * 0.01:
                continue
                
            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.05 * perimeter  # More relaxed approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4 and w * h * 0.02 < area < w * h * 0.98:
                document_contour = approx
                break
    
    # Last fallback: use the largest contour and get its bounding rectangle
    if document_contour is None and len(contours) > 0:
        largest_contour = contours[0]
        rect = cv2.minAreaRect(largest_contour)
        document_contour = np.int32(cv2.boxPoints(rect))
    
    # Ultimate fallback: use entire image (but this shouldn't happen with real documents)
    if document_contour is None:
        document_contour = np.array([[50, 50], [w-50, 50], [w-50, h-50], [50, h-50]], dtype=np.float32)
    
    # STEP 4: Extract and order corners
    corners = document_contour.reshape(4, 2).astype(np.float32)
    corners = order_corners(corners)
    
    # STEP 5: Calculate output dimensions
    # Calculate width and height of output rectangle
    width_top = distance(corners[0], corners[1])
    width_bottom = distance(corners[3], corners[2])
    height_left = distance(corners[0], corners[3])
    height_right = distance(corners[1], corners[2])
    
    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))
    
    # STEP 6: Perspective transformation
    dst_corners = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    
    # Apply transformation to original image
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    
    # STEP 7: Enhance scanned document
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for clean text
    scanned = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 10
    )
    
    # STEP 8: Visualize corners and edges
    corners_viz = image.copy()
    
    # Draw detected corners with labels
    corner_labels = ['TL', 'TR', 'BR', 'BL']
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, (corner, label, color) in enumerate(zip(corners, corner_labels, colors)):
        cv2.circle(corners_viz, tuple(corner.astype(int)), 10, color, -1)
        cv2.putText(corners_viz, label, (int(corner[0])+15, int(corner[1])-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw edges between corners
    for i in range(4):
        pt1 = tuple(corners[i].astype(int))
        pt2 = tuple(corners[(i + 1) % 4].astype(int))
        cv2.line(corners_viz, pt1, pt2, (0, 255, 255), 3)
    
    # Debug visualization
    if debug:
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 5, 1), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title('Original')
        plt.subplot(1, 5, 2), plt.imshow(filtered, cmap='gray'), plt.title('Filtered')
        plt.subplot(1, 5, 3), plt.imshow(combined_edges, cmap='gray'), plt.title('Combined Edges')
        plt.subplot(1, 5, 4), plt.imshow(cv2.cvtColor(corners_viz, cv2.COLOR_BGR2RGB)), plt.title('Document Corners')
        plt.subplot(1, 5, 5), plt.imshow(scanned, cmap='gray'), plt.title('Scanned Document')
        plt.tight_layout()
        plt.show()
    
    return original, corners_viz, scanned


def test_scanner(image_path):
    """
    Test the document scanner
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (original, corners_viz, scanned)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None
    
    original, corners_viz, scanned = document_scanner(image, debug=True)
    
    # Display final results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(corners_viz, cv2.COLOR_BGR2RGB)), plt.title('Paper Corner Detection')
    plt.subplot(1, 3, 3), plt.imshow(scanned, cmap='gray'), plt.title('Scanned Paper')
    plt.tight_layout()
    plt.show()
    
    return original, corners_viz, scanned


def simple_quadrilateral_detection(image_path):
    """
    Simple quadrilateral detection function
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of quadrilaterals found
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return []
    
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocess
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter quadrilaterals
    quads = []
    
    for cnt in contours:
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Check if it's a quadrilateral
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 1000:  # discard very small areas
                quads.append((area, approx))
    
    # Sort by area, largest first
    quads = sorted(quads, key=lambda x: x[0], reverse=True)
    
    # Draw all plausible quadrilaterals
    for area, quad in quads:
        cv2.drawContours(original, [quad], -1, (0, 255, 0), 2)
    
    return quads, original


def harris_corner_detection(image_path):
    """
    Harris corner detection function
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image with corners marked
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value, it may vary depending on the image
    img[dst > 0.01 * dst.max()] = [255, 0, 0]
    
    return img
