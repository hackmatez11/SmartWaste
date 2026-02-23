import React, { useState, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

export default function CameraDetection({ isOpen, onClose }) {
    const [isLoading, setIsLoading] = useState(false);
    const [detections, setDetections] = useState([]);
    const [modelLoaded, setModelLoaded] = useState(false);
    const [error, setError] = useState(null);
    const [stream, setStream] = useState(null);

    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const sessionRef = useRef(null);
    const animationFrameRef = useRef(null);

    // ONNX Model configuration
    const MODEL_INPUT_SIZE = 640; // YOLOv8 default input size
    const CONFIDENCE_THRESHOLD = 0.5;
    const IOU_THRESHOLD = 0.45;

    // Waste categories matching the model
    const CLASSES = ['bin', 'garbage', 'spills'];

    // Track detected sites to prevent duplicates
    const detectedSitesRef = useRef(new Set());
    const [taskStats, setTaskStats] = useState({ created: 0, pending: 0 });

    // Load ONNX model
    useEffect(() => {
        if (isOpen && !modelLoaded) {
            loadModel();
        }

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
            }
        };
    }, [isOpen]);

    const loadModel = async () => {
        try {
            setIsLoading(true);
            setError(null);

            // Configure ONNX Runtime
            ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';

            // Load the real model from public folder
            const modelPath = '/waste_detection.onnx';

            try {
                console.log('Loading ONNX model from:', modelPath);
                const session = await ort.InferenceSession.create(modelPath);
                sessionRef.current = session;
                setModelLoaded(true);
                console.log('✅ Model loaded successfully!');
                console.log('Model inputs:', session.inputNames);
                console.log('Model outputs:', session.outputNames);
            } catch (modelError) {
                console.warn('⚠️ Model file not found or failed to load. Running in demo mode.');
                console.error('Model error:', modelError);
                setError('Model not found. Running in demo mode with simulated detections.');
                setModelLoaded(true); // Set to true for demo mode
            }

            setIsLoading(false);
        } catch (err) {
            console.error('Error loading model:', err);
            setError('Failed to load detection model. Running in demo mode.');
            setIsLoading(false);
            setModelLoaded(true); // Continue in demo mode
        }
    };

    const startCamera = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });

            setStream(mediaStream);

            if (videoRef.current) {
                videoRef.current.srcObject = mediaStream;
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current.play();
                    detectObjects();
                };
            }
        } catch (err) {
            console.error('Error accessing camera:', err);
            setError('Failed to access camera. Please ensure camera permissions are granted.');
        }
    };

    const preprocessImage = (imageData) => {
        const { data, width, height } = imageData;
        const input = new Float32Array(3 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE);

        // Resize and normalize image
        for (let y = 0; y < MODEL_INPUT_SIZE; y++) {
            for (let x = 0; x < MODEL_INPUT_SIZE; x++) {
                const srcX = Math.floor(x * width / MODEL_INPUT_SIZE);
                const srcY = Math.floor(y * height / MODEL_INPUT_SIZE);
                const srcIdx = (srcY * width + srcX) * 4;

                // Normalize to [0, 1] and arrange in CHW format
                input[y * MODEL_INPUT_SIZE + x] = data[srcIdx] / 255.0; // R
                input[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + y * MODEL_INPUT_SIZE + x] = data[srcIdx + 1] / 255.0; // G
                input[2 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE + y * MODEL_INPUT_SIZE + x] = data[srcIdx + 2] / 255.0; // B
            }
        }

        return input;
    };

    const detectObjects = async () => {
        if (!videoRef.current || !canvasRef.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // Set canvas size to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw video frame to canvas
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (sessionRef.current) {
            try {
                // Get image data from canvas
                const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

                // Preprocess image
                const inputTensor = preprocessImage(imageData);

                // Run inference
                const tensor = new ort.Tensor('float32', inputTensor, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
                const feeds = { images: tensor };
                const results = await sessionRef.current.run(feeds);

                // Process results (this depends on your model's output format)
                const output = results[Object.keys(results)[0]];
                const detectedObjects = processDetections(output, canvas.width, canvas.height);

                setDetections(detectedObjects);
                drawDetections(ctx, detectedObjects);

                // Create tasks for each detection
                if (detectedObjects.length > 0) {
                    for (const detection of detectedObjects) {
                        await createTask(detection, canvas);
                    }
                }
            } catch (err) {
                console.error('Detection error:', err);
            }
        } else {
            // Demo mode - show random detections
            const demoDetections = generateDemoDetections();
            setDetections(demoDetections);
            drawDetections(ctx, demoDetections);
        }

        // Continue detection loop
        animationFrameRef.current = requestAnimationFrame(detectObjects);
    };

    const generateDemoDetections = () => {
        // Generate random demo detections for demonstration
        if (Math.random() > 0.7) {
            return [{
                class: CLASSES[Math.floor(Math.random() * CLASSES.length)],
                confidence: 0.85 + Math.random() * 0.15,
                bbox: {
                    x: Math.random() * 0.5,
                    y: Math.random() * 0.5,
                    width: 0.2 + Math.random() * 0.2,
                    height: 0.2 + Math.random() * 0.2
                }
            }];
        }
        return [];
    };

    const processDetections = (output, canvasWidth, canvasHeight) => {
        if (!output || !output.data) {
            console.log('No output data from model');
            return [];
        }

        const shape = output.dims;
        const data = output.data;

        console.log('Model output shape:', shape);
        console.log('First 20 values:', Array.from(data.slice(0, 20)));

        let rawDetections = [];

        // Process different YOLO output formats
        if (shape.length === 3) {
            const [batch, values, numDetections] = shape;
            console.log(`Processing format: batch=${batch}, values=${values}, detections=${numDetections}`);

            // YOLOv8 transposed format: [1, 7, 8400] where values < numDetections
            if (values < numDetections) {
                rawDetections = processYOLOv8TransposedFormat(data, numDetections, values);
            }
            // Standard format: [1, 8400, 7]
            else {
                rawDetections = processYOLOStandardFormat(data, numDetections, values);
            }
        }

        // Filter by confidence threshold
        const confidenceThreshold = 0.8;
        const validDetections = rawDetections.filter(det =>
            det.confidence > confidenceThreshold &&
            det.classId >= 0 &&
            det.classId < CLASSES.length
        );

        console.log(`Found ${validDetections.length} valid detections (threshold: ${confidenceThreshold})`);

        // Apply Non-Maximum Suppression to remove duplicate detections
        const nmsDetections = applyNMS(validDetections, 0.45);
        console.log(`After NMS: ${nmsDetections.length} detections`);

        // Convert to display format
        return nmsDetections.map(det => ({
            class: CLASSES[det.classId],
            confidence: det.confidence,
            bbox: {
                x: Math.max(0, det.x - det.w / 2),
                y: Math.max(0, det.y - det.h / 2),
                width: det.w,
                height: det.h
            }
        }));
    };

    // Non-Maximum Suppression to remove overlapping detections
    const applyNMS = (detections, iouThreshold) => {
        if (detections.length === 0) return [];

        // Sort by confidence (highest first)
        const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
        const keep = [];

        while (sorted.length > 0) {
            const current = sorted.shift();
            keep.push(current);

            // Remove detections that overlap significantly with current
            const remaining = [];
            for (const det of sorted) {
                // Only compare detections of the same class
                if (det.classId !== current.classId) {
                    remaining.push(det);
                    continue;
                }

                const iou = calculateIOU(current, det);
                if (iou < iouThreshold) {
                    remaining.push(det);
                }
            }
            sorted.length = 0;
            sorted.push(...remaining);
        }

        return keep;
    };

    // Calculate Intersection over Union (IoU)
    const calculateIOU = (box1, box2) => {
        const x1_min = box1.x - box1.w / 2;
        const y1_min = box1.y - box1.h / 2;
        const x1_max = box1.x + box1.w / 2;
        const y1_max = box1.y + box1.h / 2;

        const x2_min = box2.x - box2.w / 2;
        const y2_min = box2.y - box2.h / 2;
        const x2_max = box2.x + box2.w / 2;
        const y2_max = box2.y + box2.h / 2;

        // Calculate intersection area
        const intersect_x_min = Math.max(x1_min, x2_min);
        const intersect_y_min = Math.max(y1_min, y2_min);
        const intersect_x_max = Math.min(x1_max, x2_max);
        const intersect_y_max = Math.min(y1_max, y2_max);

        const intersect_w = Math.max(0, intersect_x_max - intersect_x_min);
        const intersect_h = Math.max(0, intersect_y_max - intersect_y_min);
        const intersect_area = intersect_w * intersect_h;

        // Calculate union area
        const box1_area = box1.w * box1.h;
        const box2_area = box2.w * box2.h;
        const union_area = box1_area + box2_area - intersect_area;

        // Return IoU
        return union_area > 0 ? intersect_area / union_area : 0;
    };

    // Process YOLOv8 transposed format [1, 7, 8400]
    const processYOLOv8TransposedFormat = (data, numDetections, numValues) => {
        const detections = [];

        for (let i = 0; i < numDetections; i++) {
            // Get bounding box coordinates (normalized 0-1)
            const x = data[0 * numDetections + i] / MODEL_INPUT_SIZE;
            const y = data[1 * numDetections + i] / MODEL_INPUT_SIZE;
            const w = data[2 * numDetections + i] / MODEL_INPUT_SIZE;
            const h = data[3 * numDetections + i] / MODEL_INPUT_SIZE;

            // Get class probabilities (starting from index 4)
            let maxClassConf = 0;
            let bestClass = 0;

            for (let j = 4; j < numValues && j < 7; j++) {
                const classConf = data[j * numDetections + i];
                if (classConf > maxClassConf) {
                    maxClassConf = classConf;
                    bestClass = j - 4;
                }
            }

            detections.push({
                x, y, w, h,
                confidence: maxClassConf,
                classId: bestClass
            });
        }

        return detections;
    };

    // Process standard YOLO format [1, 8400, 7]
    const processYOLOStandardFormat = (data, numDetections, values) => {
        const detections = [];

        for (let i = 0; i < numDetections; i++) {
            const base = i * values;

            // Get bounding box
            const x = data[base] / MODEL_INPUT_SIZE;
            const y = data[base + 1] / MODEL_INPUT_SIZE;
            const w = data[base + 2] / MODEL_INPUT_SIZE;
            const h = data[base + 3] / MODEL_INPUT_SIZE;

            // Get objectness score (if present)
            const objConf = values > 5 ? data[base + 4] : 1.0;

            // Get class probabilities
            let maxClassConf = 0;
            let bestClass = 0;
            const startIdx = values > 5 ? 5 : 4;
            const maxClasses = Math.min(3, values - startIdx);

            for (let j = 0; j < maxClasses; j++) {
                const classConf = data[base + startIdx + j];
                if (classConf > maxClassConf) {
                    maxClassConf = classConf;
                    bestClass = j;
                }
            }

            detections.push({
                x, y, w, h,
                confidence: objConf * maxClassConf,
                classId: bestClass
            });
        }

        return detections;
    };

    const drawDetections = (ctx, detections) => {
        detections.forEach(detection => {
            const { bbox, class: className, confidence } = detection;
            const x = bbox.x * ctx.canvas.width;
            const y = bbox.y * ctx.canvas.height;
            const width = bbox.width * ctx.canvas.width;
            const height = bbox.height * ctx.canvas.height;

            // Draw bounding box
            ctx.strokeStyle = '#1E8449';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, width, height);

            // Draw label background
            const label = `${className} ${(confidence * 100).toFixed(1)}%`;
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(label).width;
            ctx.fillStyle = '#1E8449';
            ctx.fillRect(x, y - 25, textWidth + 10, 25);

            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x + 5, y - 7);
        });
    };

    // ===== TASK CREATION FUNCTIONS =====

    // Calculate severity based on detection size (matching app.py logic)
    const calculateSeverity = (bbox, canvasWidth, canvasHeight) => {
        const detectionArea = bbox.width * bbox.height * canvasWidth * canvasHeight;
        const frameArea = canvasWidth * canvasHeight;
        const coveragePercentage = (detectionArea / frameArea) * 100;

        if (coveragePercentage >= 20) return "High";
        if (coveragePercentage >= 10) return "Medium";
        return "Low";
    };

    // Calculate priority based on class and severity (matching app.py logic)
    const calculatePriority = (className, severity) => {
        const classPriority = {
            "spills": "High",
            "garbage": "Medium",
            "bin": "Low"
        };

        const basePriority = classPriority[className.toLowerCase()] || "Low";
        const priorityLevels = { "High": 3, "Medium": 2, "Low": 1 };

        return priorityLevels[severity] > priorityLevels[basePriority]
            ? severity
            : basePriority;
    };

    // Get GPS coordinates
    const getGPSCoordinates = async () => {
        return new Promise((resolve) => {
            if (!navigator.geolocation) {
                console.log('Geolocation not available');
                resolve({ latitude: null, longitude: null });
                return;
            }

            navigator.geolocation.getCurrentPosition(
                (position) => {
                    console.log('GPS coordinates obtained');
                    resolve({
                        latitude: position.coords.latitude,
                        longitude: position.coords.longitude
                    });
                },
                (error) => {
                    console.warn('GPS error:', error.message);
                    resolve({ latitude: null, longitude: null });
                },
                { timeout: 5000, enableHighAccuracy: false }
            );
        });
    };

    // Create task in MongoDB (matching app.py structure)
    const createTask = async (detection, canvas) => {
        try {
            const { class: className, confidence, bbox } = detection;

            // Calculate center position
            const centerX = (bbox.x + bbox.width / 2) * canvas.width;
            const centerY = (bbox.y + bbox.height / 2) * canvas.height;

            // Check for duplicate detection using spatial hashing (matching app.py)
            const key = `${Math.floor(centerX / 50)}_${Math.floor(centerY / 50)}`;
            if (detectedSitesRef.current.has(key)) {
                console.log('Duplicate detection prevented');
                return null;
            }

            // Mark this location as detected
            detectedSitesRef.current.add(key);
            console.log(`⚠️ New detection at (x=${centerX.toFixed(0)}, y=${centerY.toFixed(0)})`);

            // Calculate severity and priority
            const severity = calculateSeverity(bbox, canvas.width, canvas.height);
            const priority = calculatePriority(className, severity);

            // Get GPS coordinates
            const gps = await getGPSCoordinates();

            // Capture screenshot as base64
            const imageData = canvas.toDataURL('image/jpeg', 0.8);

            // Calculate bounding box coordinates for backend
            const x1 = bbox.x * canvas.width;
            const y1 = bbox.y * canvas.height;
            const x2 = (bbox.x + bbox.width) * canvas.width;
            const y2 = (bbox.y + bbox.height) * canvas.height;

            // Create task payload matching backend API expectations
            const taskData = {
                detectedClass: className,
                x1: x1.toString(),
                y1: y1.toString(),
                x2: x2.toString(),
                y2: y2.toString(),
                confidenceScore: confidence.toString(),
                frameHeight: canvas.height.toString(),
                frameWidth: canvas.width.toString(),
                latitude: gps.latitude?.toString() || '',
                longitude: gps.longitude?.toString() || '',
                cameraId: 'CAM1',
                imageData: imageData // Base64 image for Cloudinary upload
            };

            console.log('Creating task with Cloudinary upload...');

            // Send to backend as JSON
            const response = await fetch(`https://smartwatemobile-1.onrender.com/api/task/detections`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(taskData)
            });

            const result = await response.json();

            if (result.success || response.ok) {
                console.log(`✅ Task created successfully`);
                console.log(`📸 Image uploaded to Cloudinary: ${result.data.imagePath}`);
                setTaskStats(prev => ({
                    created: prev.created + 1,
                    pending: prev.pending + 1
                }));
                return result;
            } else {
                throw new Error(result.error || 'Failed to create task');
            }

        } catch (error) {
            console.error('❌ Task creation failed:', error);
            return null;
        }
    };


    useEffect(() => {
        if (isOpen && modelLoaded) {
            startCamera();
        }
    }, [isOpen, modelLoaded]);

    const handleClose = () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
        }
        setStream(null);
        setDetections([]);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90">
            <div className="relative w-full h-full max-w-6xl max-h-screen p-4">
                {/* Header */}
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-2xl font-bold text-white">
                        Waste Detection Camera
                    </h2>
                    <button
                        onClick={handleClose}
                        className="text-white hover:text-red-500 transition-colors"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-8 w-8"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M6 18L18 6M6 6l12 12"
                            />
                        </svg>
                    </button>
                </div>

                {/* Camera View */}
                <div className="relative bg-neutral-900 rounded-lg overflow-hidden">
                    {isLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-neutral-900">
                            <div className="text-white text-center">
                                <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-[#1E8449] mx-auto mb-4"></div>
                                <p>Loading detection model...</p>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="absolute top-4 left-4 right-4 bg-yellow-500/90 text-white p-4 rounded-lg">
                            {error}
                        </div>
                    )}

                    <video
                        ref={videoRef}
                        className="w-full h-auto"
                        playsInline
                        muted
                        style={{ display: stream ? 'block' : 'none' }}
                    />

                    <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full"
                    />

                    {/* Task Statistics */}
                    <div className="absolute top-4 right-4 bg-neutral-800/90 backdrop-blur-sm rounded-lg p-3">
                        <h3 className="text-white font-semibold mb-2 text-sm">Tasks</h3>
                        <div className="space-y-1 text-xs">
                            <div className="flex justify-between gap-3">
                                <span className="text-neutral-400">Created:</span>
                                <span className="text-[#1E8449] font-semibold">{taskStats.created}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                                <span className="text-neutral-400">Pending:</span>
                                <span className="text-yellow-500 font-semibold">{taskStats.pending}</span>
                            </div>
                        </div>
                    </div>

                    {/* Detection Info Panel */}
                    {detections.length > 0 && (
                        <div className="absolute bottom-4 left-4 right-4 bg-neutral-800/90 p-4 rounded-lg">
                            <h3 className="text-white font-bold mb-2">Detected Items:</h3>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                {detections.map((detection, index) => (
                                    <div
                                        key={index}
                                        className="bg-[#1E8449] text-white px-3 py-2 rounded-md text-sm"
                                    >
                                        <div className="font-semibold capitalize">{detection.class}</div>
                                        <div className="text-xs opacity-90">
                                            {(detection.confidence * 100).toFixed(1)}% confidence
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Instructions */}
                    {!stream && !isLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-neutral-900">
                            <div className="text-white text-center p-8">
                                <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    className="h-24 w-24 mx-auto mb-4 text-[#1E8449]"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
                                    />
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
                                    />
                                </svg>
                                <p className="text-xl mb-2">Point your camera at waste items</p>
                                <p className="text-gray-400">
                                    The AI will automatically detect and classify waste types
                                </p>
                            </div>
                        </div>
                    )}
                </div>

                {/* Info Footer */}
                <div className="mt-4 text-center text-gray-400 text-sm">
                    <p>
                        Using AI-powered waste detection • Real-time classification •
                        {modelLoaded && !sessionRef.current && ' Demo Mode'}
                    </p>
                </div>
            </div>
        </div>
    );
}
