interface CVParameters {
    harris: {
        threshold: number;
        k: number;
        nmsRadius: number;
    };
    blur: {
        kernelSize: number;
        sigma: number;
    };
    edge: {
        threshold: number;
        sampleRate: number;
    };
    display: {
        showEdges: boolean;
        showCorners: boolean;
        cornerSize: number;
    };
}
interface Point {
    x: number;
    y: number;
}
interface Corner {
    x: number;
    y: number;
    response: number;
}
interface VideoConstraints {
    video: {
        width: {
            ideal: number;
        };
        height: {
            ideal: number;
        };
        facingMode: string;
    };
}
declare class CVUtils {
    static convolve2D(imageData: ImageData, kernel: number[][], stride?: number): ImageData;
    static toGrayscale(imageData: ImageData): ImageData;
    static generateGaussianKernel(size: number, sigma: number): number[][];
    static getSobelKernels(): {
        x: number[][];
        y: number[][];
    };
    static sobelEdgeDetection(imageData: ImageData): ImageData;
    static harrisCornerDetection(imageData: ImageData, threshold?: number, k?: number, nmsRadius?: number): Corner[];
    static nonMaximumSuppression(corners: Corner[], radius: number): Corner[];
    static gaussianBlur(imageData: ImageData, kernelSize?: number, sigma?: number): ImageData;
}
declare class DocumentScanner {
    private video;
    private overlayCanvas;
    private captureCanvas;
    private overlayCtx;
    private captureCtx;
    private hiddenCanvas;
    private hiddenCtx;
    private startCameraBtn;
    private captureBtn;
    private toggleDetectionBtn;
    private resultsContainer;
    stream: MediaStream | null;
    private detectionEnabled;
    private isProcessing;
    private documentCount;
    private cvParams;
    private lastFrameTime;
    private frameCount;
    private fps;
    constructor();
    private initializeEventListeners;
    private startCamera;
    private setupCanvases;
    private startDetectionLoop;
    private detectCorners;
    private drawColoredCorners;
    private drawDocumentOutline;
    private captureDocument;
    private createDocumentItem;
    downloadDocument(imageData: string, documentNumber: number): void;
    deleteDocument(documentElement: HTMLElement): void;
    private toggleDetection;
    private showMessage;
    private initializeParameterControls;
    private setupParameterSlider;
    private setupParameterCheckbox;
    private resetParameters;
    private saveParameters;
    private loadParameters;
    private updateAllControls;
    private updatePerformanceInfo;
    private updatePerformanceDisplay;
}
declare let scanner: DocumentScanner;
//# sourceMappingURL=script.d.ts.map