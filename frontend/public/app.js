// DABTTB AI Frontend - Advanced Table Tennis Ball Tracking
class DABTTBAI {
    constructor() {
        this.apiUrl = 'http://localhost:8005';
        this.selectedFile = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkServiceStatus();
        this.loadServiceInfo();
        this.setupAnalytics();
    }

    setupEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');

        // File input change
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e.target.files[0]);
        });

        // Analyze button click
        analyzeBtn.addEventListener('click', (e) => {
            this.analyzeVideo();
        });

        // Analytics event listeners
        this.setupAnalyticsEventListeners();

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('video/')) {
                this.handleFileSelect(file);
            } else {
                this.showNotification('Please select a valid video file', 'error');
            }
        });
    }

    setupAnalyticsEventListeners() {
        // Analytics tab switching
        const tabs = document.querySelectorAll('.analytics-tab');
        tabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchAnalyticsTab(e.target.id);
            });
        });

        // Analytics action buttons
        document.getElementById('refreshAnalyticsBtn').addEventListener('click', () => {
            this.loadAnalytics();
        });

        document.getElementById('exportDataBtn').addEventListener('click', () => {
            this.exportAnalyticsData();
        });

        document.getElementById('generateTrajectoryBtn').addEventListener('click', () => {
            this.generateTrajectory();
        });

        // Chat functionality
        const chatInput = document.getElementById('chatInput');
        const sendChatBtn = document.getElementById('sendChatBtn');
        
        if (chatInput && sendChatBtn) {
            chatInput.addEventListener('input', () => {
                sendChatBtn.disabled = chatInput.value.trim() === '';
            });

            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !sendChatBtn.disabled) {
                    this.sendChatMessage();
                }
            });

            sendChatBtn.addEventListener('click', () => {
                this.sendChatMessage();
            });
        }
    }

    setupAnalytics() {
        // Load analytics data on startup
        this.loadAnalytics();
    }

    async loadAnalytics() {
        try {
            this.showAnalyticsLoading(true);
            
            // Load analytics summary
            try {
                const summaryResponse = await fetch(`${this.apiUrl}/analytics/summary`);
                if (summaryResponse.ok) {
                    const summaryData = await summaryResponse.json();
                    this.displayAnalyticsSummary(summaryData);
                } else {
                    console.warn('Analytics summary not available:', summaryResponse.status);
                }
            } catch (err) {
                console.warn('Analytics summary request failed:', err);
            }

            // Load detection data
            try {
                const detectionsResponse = await fetch(`${this.apiUrl}/analytics/detections`);
                if (detectionsResponse.ok) {
                    const detectionsData = await detectionsResponse.json();
                    this.displayDetections(detectionsData);
                } else {
                    console.warn('Analytics detections not available:', detectionsResponse.status);
                }
            } catch (err) {
                console.warn('Analytics detections request failed:', err);
            }

            // Load Gemma analyses
            try {
                const gemmaResponse = await fetch(`${this.apiUrl}/analytics/gemma`);
                if (gemmaResponse.ok) {
                    const gemmaData = await gemmaResponse.json();
                    this.displayGemmaAnalyses(gemmaData);
                } else {
                    console.warn('Analytics gemma not available:', gemmaResponse.status);
                }
            } catch (err) {
                console.warn('Analytics gemma request failed:', err);
            }

            // Load anomaly data
            try {
                const anomaliesResponse = await fetch(`${this.apiUrl}/analytics/anomalies`);
                if (anomaliesResponse.ok) {
                    const anomaliesData = await anomaliesResponse.json();
                    this.displayAnomalies(anomaliesData);
                } else {
                    console.warn('Analytics anomalies not available:', anomaliesResponse.status);
                }
            } catch (err) {
                console.warn('Analytics anomalies request failed:', err);
            }

            // Load anomaly scores
            try {
                const anomalyScoresResponse = await fetch(`${this.apiUrl}/analytics/anomaly-scores`);
                if (anomalyScoresResponse.ok) {
                    const anomalyScoresData = await anomalyScoresResponse.json();
                    this.displayAnomalyScores(anomalyScoresData);
                } else {
                    console.warn('Analytics anomaly scores not available:', anomalyScoresResponse.status);
                }
            } catch (err) {
                console.warn('Analytics anomaly scores request failed:', err);
            }

            this.showNotification('Analytics interface ready (no data yet - analyze videos to populate)', 'info');
            
        } catch (error) {
            console.error('Failed to load analytics:', error);
            this.showNotification('Analytics interface ready (connect to service)', 'warning');
        } finally {
            this.showAnalyticsLoading(false);
            // Update chat video options after loading analytics
            this.updateChatVideoOptions();
        }
    }

    displayAnalyticsSummary(response) {
        if (response.status === 'success' && response.data) {
            const data = response.data;
            
            // Update summary cards
            if (data.videos) {
                document.getElementById('totalVideos').textContent = data.videos.total || 0;
                document.getElementById('avgDuration').textContent = `Avg: ${data.videos.avg_duration || 0}s`;
                document.getElementById('avgResolution').textContent = data.videos.avg_resolution || 'N/A';
                document.getElementById('avgFps').textContent = data.videos.avg_fps || 0;

                // Populate trajectory video select
                const videoSelect = document.getElementById('trajectoryVideoSelect');
                if (videoSelect) {
                    videoSelect.innerHTML = '<option value="">-- Select a Video --</option>'; // Clear existing options
                    data.videos.list.forEach(video => {
                        const option = document.createElement('option');
                        option.value = video.id;
                        option.textContent = video.filename;
                        videoSelect.appendChild(option);
                    });
                }
            }

            if (data.frames) {
                document.getElementById('totalDetections').textContent = data.frames.with_ball || 0;
                document.getElementById('detectionRate').textContent = `Rate: ${data.frames.detection_rate || 0}%`;
                document.getElementById('timeRange').textContent = data.frames.time_range || 'N/A';
            }

            if (data.gemma) {
                document.getElementById('totalAnalyses').textContent = data.gemma.total_analyses || 0;
                document.getElementById('avgConfidence').textContent = `Avg: ${data.gemma.avg_confidence || 0}`;
            }

            // Update system information
            if (data.system) {
                document.getElementById('dbStatus').textContent = data.system.status || 'Unknown';
                document.getElementById('lastUpdated').textContent = new Date(data.system.last_updated).toLocaleString();
                document.getElementById('analyticsVersion').textContent = data.system.analytics_version || 'N/A';
            }
        }
    }

    displayDetections(response) {
        if (response.status === 'success' && response.data) {
            const tableBody = document.getElementById('detectionsTableBody');
            tableBody.innerHTML = '';

            response.data.forEach(detection => {
                const row = document.createElement('tr');
                row.className = 'border-b border-gray-200 hover:bg-gray-50';
                
                const position = detection.x !== null && detection.y !== null 
                    ? `(${detection.x}, ${detection.y})` 
                    : 'N/A';
                
                const detectedBadge = detection.ball_detected 
                    ? '<span class="bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">Yes</span>'
                    : '<span class="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs">No</span>';

                // Display video filename or video ID
                const videoInfo = detection.video_filename 
                    ? detection.video_filename.substring(0, 20) + (detection.video_filename.length > 20 ? '...' : '')
                    : `Video ${detection.video_id || 'N/A'}`;

                row.innerHTML = `
                    <td class="px-4 py-2 text-sm text-gray-600" title="${detection.video_filename || ''}">${videoInfo}</td>
                    <td class="px-4 py-2 text-sm text-gray-900">${detection.timestamp?.toFixed(1) || 'N/A'}s</td>
                    <td class="px-4 py-2 text-sm text-gray-900">${detection.frame_number || 'N/A'}</td>
                    <td class="px-4 py-2 text-sm">${detectedBadge}</td>
                    <td class="px-4 py-2 text-sm text-gray-900">${detection.confidence?.toFixed(3) || 'N/A'}</td>
                    <td class="px-4 py-2 text-sm text-gray-900">${position}</td>
                `;
                
                tableBody.appendChild(row);
            });
        }
    }

    displayGemmaAnalyses(response) {
        if (response.status === 'success' && response.data) {
            const container = document.getElementById('gemmaAnalyses');
            container.innerHTML = '';

            if (response.data.length === 0) {
                container.innerHTML = '<p class="text-gray-600 text-center py-4">No Gemma 3N analyses found</p>';
                return;
            }

            response.data.forEach(analysis => {
                const analysisCard = document.createElement('div');
                analysisCard.className = 'bg-gray-50 p-4 rounded-lg border border-gray-200';
                
                analysisCard.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <h4 class="font-semibold text-gray-900">${analysis.analysis_type || 'Analysis'}</h4>
                        <span class="bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs">
                            Confidence: ${analysis.confidence?.toFixed(3) || 'N/A'}
                        </span>
                    </div>
                    <p class="text-gray-700 mb-2">${analysis.response || 'No response available'}</p>
                    <div class="text-xs text-gray-500">
                        <span>File: ${analysis.filename || 'Unknown'}</span>
                        ${analysis.created_at ? ` • ${new Date(analysis.created_at).toLocaleString()}` : ''}
                    </div>
                `;
                
                container.appendChild(analysisCard);
            });
        }
    }

    switchAnalyticsTab(tabId) {
        // Remove active class from all tabs
        document.querySelectorAll('.analytics-tab').forEach(tab => {
            tab.classList.remove('active', 'border-blue-500', 'text-blue-600');
            tab.classList.add('border-transparent', 'text-gray-500');
        });

        // Hide all content
        document.querySelectorAll('.analytics-content').forEach(content => {
            content.classList.add('hidden');
        });

        // Activate selected tab
        const activeTab = document.getElementById(tabId);
        activeTab.classList.add('active', 'border-blue-500', 'text-blue-600');
        activeTab.classList.remove('border-transparent', 'text-gray-500');

        // Show corresponding content
        const contentMap = {
            'summaryTab': 'summaryContent',
            'detectionsTab': 'detectionsContent',
            'trajectoryTab': 'trajectoryContent',
            'anomaliesTab': 'anomaliesContent',
            'gemmaTab': 'gemmaContent',
            'chatTab': 'chatContent'
        };

        const contentId = contentMap[tabId];
        if (contentId) {
            document.getElementById(contentId).classList.remove('hidden');
            
            // Initialize chat when chat tab is selected
            if (tabId === 'chatTab') {
                this.updateChatVideoOptions();
            }
        }
    }

    async exportAnalyticsData() {
        try {
            const response = await fetch(`${this.apiUrl}/analytics/export`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'format=json&output_dir=exports'
            });

            if (response.ok) {
                const result = await response.json();
                this.showNotification(`Data exported successfully: ${result.exported_files?.length || 0} files`, 'success');
            } else {
                throw new Error('Export failed');
            }
        } catch (error) {
            console.error('Export failed:', error);
            this.showNotification('Failed to export analytics data', 'error');
        }
    }

    async generateTrajectory() {
        const videoSelect = document.getElementById('trajectoryVideoSelect');
        const videoId = videoSelect.value;

        if (!videoId) {
            this.showNotification('Please select a video first', 'error');
            return;
        }

        try {
            const response = await fetch(`${this.apiUrl}/analytics/trajectory/${videoId}`);
            
            if (response.ok) {
                const result = await response.json();
                
                // Update trajectory visualization area
                const trajectoryDiv = document.getElementById('trajectoryVisualization');
                trajectoryDiv.innerHTML = `
                    <div class="text-center">
                        <h4 class="text-lg font-semibold mb-2 text-green-600">Trajectory Generated Successfully</h4>
                        <p class="text-gray-600 mb-3">Video ID: ${result.video_id}</p>
                        <img src="${this.apiUrl}${result.trajectory_path.replace('/app', '')}" alt="Trajectory Plot" class="mx-auto my-4 rounded-lg shadow-md"/>
                        <p class="text-sm text-gray-500">Path: ${result.trajectory_path}</p>
                        <button id="generateTrajectoryBtn" class="mt-3 bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg">
                            Generate New Trajectory
                        </button>
                    </div>
                `;
                
                // Re-add event listener
                document.getElementById('generateTrajectoryBtn').addEventListener('click', () => {
                    this.generateTrajectory();
                });
                
                this.showNotification('Trajectory visualization generated successfully', 'success');
            } else {
                throw new Error('Trajectory generation failed');
            }
        } catch (error) {
            console.error('Trajectory generation failed:', error);
            this.showNotification('Failed to generate trajectory visualization', 'error');
        }
    }

    showAnalyticsLoading(show) {
        const loadingDiv = document.getElementById('analyticsLoading');
        const contentDiv = document.getElementById('analyticsContent');
        
        if (show) {
            loadingDiv.classList.remove('hidden');
            contentDiv.classList.add('hidden');
        } else {
            loadingDiv.classList.add('hidden');
            contentDiv.classList.remove('hidden');
        }
    }

    async checkServiceStatus() {
        const statusBadge = document.getElementById('statusBadge');
        
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            
            if (data.status === 'healthy') {
                statusBadge.innerHTML = '<i class="fas fa-check-circle mr-1"></i>Online';
                statusBadge.className = 'px-3 py-1 rounded-full text-sm bg-green-500 text-white';
            } else {
                statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle mr-1"></i>Issues';
                statusBadge.className = 'px-3 py-1 rounded-full text-sm bg-yellow-500 text-white';
            }
        } catch (error) {
            statusBadge.innerHTML = '<i class="fas fa-times-circle mr-1"></i>Offline';
            statusBadge.className = 'px-3 py-1 rounded-full text-sm bg-red-500 text-white';
            console.error('Service check failed:', error);
        }
    }

    async loadServiceInfo() {
        try {
            const response = await fetch(`${this.apiUrl}/`);
            const data = await response.json();
            
            const serviceInfo = document.getElementById('serviceInfo');
            serviceInfo.innerHTML = `
                <div class="bg-blue-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-blue-800">Service</h3>
                    <p class="text-blue-600">${data.service || 'TTBall_4 AI'}</p>
                    <p class="text-sm text-blue-500">Version: ${data.version || 'Unknown'}</p>
                </div>
                <div class="bg-green-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-green-800">Device</h3>
                    <p class="text-green-600">${data.device || 'CPU'}</p>
                    <p class="text-sm text-green-500">CUDA: ${data.cuda_available ? 'Available' : 'Not Available'}</p>
                </div>
                <div class="bg-purple-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-purple-800">Capabilities</h3>
                    <p class="text-purple-600">Multimodal Analysis</p>
                    <p class="text-sm text-purple-500">Gemma 3N Ready</p>
                </div>
            `;

            // Load model information
            await this.loadModelInfo();
        } catch (error) {
            console.error('Failed to load service info:', error);
        }
    }

    async loadModelInfo() {
        try {
            const response = await fetch(`${this.apiUrl}/models`);
            const data = await response.json();
            console.log('Models available:', data);
        } catch (error) {
            console.error('Failed to load model info:', error);
        }
    }

    handleFileSelect(file) {
        if (!file) return;

        if (!file.type.startsWith('video/')) {
            this.showNotification('Please select a valid video file', 'error');
            return;
        }

        // Check file size (100MB limit)
        const maxSize = 100 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showNotification('File size must be less than 100MB', 'error');
            return;
        }

        this.selectedFile = file;
        
        // Update UI
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <i class="fas fa-video text-4xl text-green-500 mb-4"></i>
            <p class="text-lg text-green-600 mb-2">Video Selected: ${file.name}</p>
            <p class="text-sm text-gray-500">Size: ${this.formatFileSize(file.size)}</p>
            <button onclick="document.getElementById('fileInput').click()" 
                    class="mt-4 bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded transition-colors">
                <i class="fas fa-exchange-alt mr-2"></i>
                Choose Different File
            </button>
        `;

        // Enable analyze button
        document.getElementById('analyzeBtn').disabled = false;
    }

    async analyzeVideo() {
        if (!this.selectedFile) {
            this.showNotification('Please select a video file first', 'error');
            return;
        }

        const analysisType = document.getElementById('analysisType').value;

        // Show loading
        this.showLoading();

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            
            let endpoint = '/analyze';
            let enhanceEndpoint = null;
            
            if (analysisType === 'gemma-enhanced') {
                // First do standard analysis, then enhance with Gemma
                formData.append('analysis_type', 'full');
            } else {
                formData.append('analysis_type', analysisType);
            }

            const response = await fetch(`${this.apiUrl}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                // If Gemma-Enhanced analysis, run the enhancement after standard analysis
                if (analysisType === 'gemma-enhanced' && result.video_id) {
                    this.showNotification('Standard analysis completed! Running Gemma-Enhanced Detection...', 'info');
                    
                    try {
                        const enhanceResponse = await fetch(`${this.apiUrl}/analytics/gemma-enhance/${result.video_id}`, {
                            method: 'POST'
                        });
                        
                        const enhanceResult = await enhanceResponse.json();
                        
                        if (enhanceResponse.ok) {
                            // Merge enhancement results with original results
                            result.gemma_enhancement = enhanceResult;
                            result.analysis_type = 'Gemma-Enhanced Detection - Physics & AI Validation';
                            this.showNotification('Gemma-Enhanced Detection completed successfully!', 'success');
                        } else {
                            console.warn('Enhancement failed, showing standard results:', enhanceResult);
                            this.showNotification('Standard analysis completed (enhancement failed)', 'warning');
                        }
                    } catch (enhanceError) {
                        console.warn('Enhancement error:', enhanceError);
                        this.showNotification('Standard analysis completed (enhancement error)', 'warning');
                    }
                }
                
                this.displayResults(result);
                if (analysisType !== 'gemma-enhanced') {
                    this.showNotification('Analysis completed successfully!', 'success');
                }
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');

        let html = `
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-gray-800 mb-2">Analysis Overview</h3>
                    <ul class="space-y-1 text-sm">
                        <li><strong>Job ID:</strong> ${result.job_id || 'N/A'}</li>
                        <li><strong>File:</strong> ${result.filename || 'N/A'}</li>
                        <li><strong>Status:</strong> <span class="text-green-600">${result.status || 'N/A'}</span></li>
                        <li><strong>Type:</strong> ${result.analysis_type || result.type || 'N/A'}</li>
                    </ul>
                </div>
                <div class="bg-gray-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-gray-800 mb-2">Model Information</h3>
                    <ul class="space-y-1 text-sm">
        `;

        if (result.gemma_3n) {
            html += `
                        <li><strong>Model:</strong> ${result.gemma_3n.model_used || 'Gemma 3N'}</li>
                        <li><strong>Multimodal:</strong> <span class="text-purple-600">Enabled</span></li>
                        <li><strong>Analysis:</strong> ${result.gemma_3n.multimodal_analysis || 'N/A'}</li>
            `;
        }
        
        // Add Gemma Enhancement information if available
        if (result.gemma_enhancement) {
            html += `
                        <li><strong>Enhancement:</strong> <span class="text-green-600">Gemma 3N Multimodal AI</span></li>
                        <li><strong>Physics Validation:</strong> <span class="text-blue-600">Active</span></li>
                        <li><strong>AI Interpolation:</strong> <span class="text-purple-600">Enabled</span></li>
                        <li><strong>Context Analysis:</strong> <span class="text-orange-600">Table Tennis Domain</span></li>
            `;
        }

        html += `
                    </ul>
                </div>
            </div>
        `;

        // Add results if available
        if (result.results) {
            html += `
                <div class="mt-6 bg-blue-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-blue-800 mb-3">Analysis Results</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div class="text-center">
                            <div class="text-2xl font-bold text-blue-600">${result.results.ball_detections || 0}</div>
                            <div class="text-sm text-blue-500">Ball Detections</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold text-green-600">${result.results.trajectory_points || 0}</div>
                            <div class="text-sm text-green-500">Trajectory Points</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold text-purple-600">${result.results.confidence ? (result.results.confidence * 100).toFixed(1) + '%' : 'N/A'}</div>
                            <div class="text-sm text-purple-500">Confidence</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold text-orange-600">${result.results.video_duration_formatted || result.results.analysis_duration + 's' || '0s'}</div>
                            <div class="text-sm text-orange-500">Video Duration</div>
                        </div>
                    </div>
            `;

            // Add technical stats if available
            if (result.results.technical_stats) {
                html += `
                    <div class="mt-4">
                        <h4 class="font-medium text-blue-800 mb-2">Technical Statistics</h4>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                            <div class="bg-white p-2 rounded">
                                <div class="font-semibold">${result.results.technical_stats.avg_ball_speed || 'N/A'} km/h</div>
                                <div class="text-gray-600">Avg Ball Speed</div>
                            </div>
                            <div class="bg-white p-2 rounded">
                                <div class="font-semibold">${result.results.technical_stats.spin_rate || 'N/A'}</div>
                                <div class="text-gray-600">Spin Rate</div>
                            </div>
                            <div class="bg-white p-2 rounded">
                                <div class="font-semibold">${result.results.technical_stats.shot_accuracy || 'N/A'}</div>
                                <div class="text-gray-600">Shot Accuracy</div>
                            </div>
                            <div class="bg-white p-2 rounded">
                                <div class="font-semibold">${result.results.technical_stats.total_rallies || 'N/A'}</div>
                                <div class="text-gray-600">Total Rallies</div>
                            </div>
                        </div>
                    </div>
                `;
            }

            if (result.results.multimodal_insights) {
                html += `
                    <div class="mt-4">
                        <h4 class="font-medium text-blue-800 mb-2">Analysis Insights</h4>
                        <ul class="list-disc list-inside space-y-1 text-sm text-blue-700">
                `;
                result.results.multimodal_insights.forEach(insight => {
                    html += `<li>${insight}</li>`;
                });
                html += `
                        </ul>
                    </div>
                `;
            }

            html += `</div>`;
        }

        // Add Gemma Enhancement Results if available
        if (result.gemma_enhancement) {
            const enhancement = result.gemma_enhancement;
            html += `
                <div class="mt-6 bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg border border-purple-200">
                    <h3 class="font-semibold text-purple-800 mb-4 flex items-center">
                        <i class="fas fa-brain mr-2"></i>
                        Gemma-Enhanced Detection System Results
                        <span class="ml-2 bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs">AI Enhanced</span>
                    </h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                        <div class="bg-white p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-purple-600">${enhancement.original_detections || 0}</div>
                            <div class="text-sm text-purple-500">Original YOLO Detections</div>
                        </div>
                        <div class="bg-white p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-green-600">${enhancement.enhanced_detections || 0}</div>
                            <div class="text-sm text-green-500">AI Enhanced Detections</div>
                        </div>
                        <div class="bg-white p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-blue-600">${enhancement.improvement_percent ? enhancement.improvement_percent.toFixed(1) + '%' : 'N/A'}</div>
                            <div class="text-sm text-blue-500">Improvement Rate</div>
                        </div>
                    </div>
                    <div class="bg-white p-4 rounded-lg">
                        <h4 class="font-medium text-gray-800 mb-2">Enhancement Features Applied:</h4>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div class="flex items-center"><i class="fas fa-check-circle text-green-500 mr-2"></i>Physics-Based Validation</div>
                            <div class="flex items-center"><i class="fas fa-check-circle text-green-500 mr-2"></i>AI Gap Interpolation</div>
                            <div class="flex items-center"><i class="fas fa-check-circle text-green-500 mr-2"></i>Context-Aware Filtering</div>
                            <div class="flex items-center"><i class="fas fa-check-circle text-green-500 mr-2"></i>Trajectory Smoothing</div>
                        </div>
                    </div>
                    ${result.video_id ? `
                    <div class="mt-4 text-center">
                        <button onclick="window.open('http://localhost:8005/results/gemma_enhanced_dashboard_video_${result.video_id}.png', '_blank')" 
                                class="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white px-4 py-2 rounded-lg transition-all">
                            <i class="fas fa-chart-line mr-2"></i>
                            View Enhanced Analytics Dashboard
                        </button>
                    </div>
                    ` : ''}
                </div>
            `;
        }
        
        // Add Gemma response if available
        if (result.gemma_response) {
            html += `
                <div class="mt-6 bg-purple-50 p-4 rounded-lg">
                    <h3 class="font-semibold text-purple-800 mb-3">
                        <i class="fas fa-robot mr-2"></i>
                        Gemma 3N Analysis
                    </h3>
                    <div class="text-purple-700 whitespace-pre-wrap">${result.gemma_response}</div>
                </div>
            `;
        }

        resultsContent.innerHTML = html;
        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    showLoading() {
        document.getElementById('loadingSection').classList.remove('hidden');
        document.getElementById('analyzeBtn').disabled = true;
        
        // Realistic progress simulation
        let progress = 0;
        const progressBar = document.getElementById('progressBar');
        const statusText = document.getElementById('statusText');
        const progressText = document.getElementById('progressText');
        
        const stages = [
            { progress: 15, text: "📤 Uploading video...", detail: "Transferring file to server...", duration: 800 },
            { progress: 35, text: "🎬 Processing video frames...", detail: "Extracting frames for analysis...", duration: 1200 },
            { progress: 55, text: "🏓 Detecting ball movements...", detail: "AI tracking ball trajectory...", duration: 1500 },
            { progress: 75, text: "📊 Analyzing trajectory data...", detail: "Computing movement patterns...", duration: 1000 },
            { progress: 90, text: "🤖 Generating AI insights...", detail: "Gemma 3N creating analysis...", duration: 1200 },
            { progress: 95, text: "✨ Finalizing analysis...", detail: "Preparing results...", duration: 500 }
        ];
        
        let currentStage = 0;
        
        const updateProgress = () => {
            if (currentStage < stages.length) {
                const stage = stages[currentStage];
                progress = stage.progress;
                progressBar.style.width = `${progress}%`;
                
                if (statusText) {
                    statusText.textContent = stage.text;
                }
                
                if (progressText) {
                    progressText.textContent = `${stage.detail} (${progress}%)`;
                }
                
                currentStage++;
                setTimeout(updateProgress, stage.duration);
            }
        };
        
        // Start progress simulation
        updateProgress();
        
        // Store stage info to clear later
        this.currentStage = currentStage;
    }

    hideLoading() {
        // Complete progress smoothly
        const progressBar = document.getElementById('progressBar');
        const statusText = document.getElementById('statusText');
        const progressText = document.getElementById('progressText');
        
        progressBar.style.width = '100%';
        
        if (statusText) {
            statusText.textContent = "✅ Analysis completed successfully!";
        }
        
        if (progressText) {
            progressText.textContent = "Analysis complete (100%)";
        }
        
        // Hide loading section after a brief moment to show completion
        setTimeout(() => {
            document.getElementById('loadingSection').classList.add('hidden');
            document.getElementById('analyzeBtn').disabled = false;
        }, 800);
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
            type === 'success' ? 'bg-green-500' : 
            type === 'error' ? 'bg-red-500' : 
            'bg-blue-500'
        } text-white`;
        
        notification.innerHTML = `
            <div class="flex items-center">
                <i class="fas ${
                    type === 'success' ? 'fa-check-circle' : 
                    type === 'error' ? 'fa-exclamation-circle' : 
                    'fa-info-circle'
                } mr-2"></i>
                <span>${message}</span>
            </div>
        `;

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    displayAnomalies(response) {
        if (response.status === 'success' && response.data) {
            const data = response.data;
            
            // Calculate summary statistics
            let totalAnomalies = 0;
            let totalBounces = 0;
            let physicsAnomalies = 0;
            let interpolatedFrames = 0;
            
            data.forEach(analysis => {
                totalAnomalies += analysis.total_anomalies || 0;
                totalBounces += analysis.total_bounces || 0;
                physicsAnomalies += analysis.physics_anomalies || 0;
                interpolatedFrames += analysis.interpolated_frames || 0;
            });
            
            // Update anomaly summary cards
            document.getElementById('totalAnomalies').textContent = totalAnomalies;
            document.getElementById('totalBounces').textContent = totalBounces;
            document.getElementById('physicsAnomalies').textContent = physicsAnomalies;
            document.getElementById('interpolatedFrames').textContent = interpolatedFrames;
        }
    }

    displayAnomalyScores(response) {
        if (response.status === 'success' && response.data) {
            const data = response.data;
            const tableBody = document.getElementById('anomaliesTableBody');
            
            if (data.length === 0) {
                tableBody.innerHTML = `
                    <tr>
                        <td colspan="5" class="px-6 py-4 text-center text-gray-500">
                            <i class="fas fa-check-circle text-green-500 mr-2"></i>
                            No anomalies detected in analyzed videos
                        </td>
                    </tr>
                `;
                return;
            }
            
            tableBody.innerHTML = '';
            
            data.forEach(anomaly => {
                const row = document.createElement('tr');
                row.className = 'hover:bg-gray-50';
                
                // Determine severity color
                const severityColor = anomaly.severity > 0.7 ? 'text-red-600 bg-red-100' : 
                                    anomaly.severity > 0.4 ? 'text-orange-600 bg-orange-100' : 
                                    'text-yellow-600 bg-yellow-100';
                
                // Format timestamp
                const timestamp = anomaly.timestamp_seconds ? 
                    `${Math.floor(anomaly.timestamp_seconds / 60)}:${(anomaly.timestamp_seconds % 60).toFixed(1).padStart(4, '0')}` : 
                    '-';
                
                row.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${anomaly.filename || 'Unknown'}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                        ${timestamp}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                            ${anomaly.anomaly_type || 'Unknown'}
                        </span>
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="px-2 py-1 text-xs font-medium rounded-full ${severityColor}">
                            ${(anomaly.severity * 100).toFixed(1)}%
                        </span>
                    </td>
                    <td class="px-6 py-4 text-sm text-gray-900 max-w-xs truncate">
                        ${anomaly.description || 'No description'}
                    </td>
                `;
                
                tableBody.appendChild(row);
            });
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async updateChatVideoOptions() {
        const chatVideoSelect = document.getElementById('chatVideoSelect');
        if (!chatVideoSelect) return;

        // Fetch video list from backend
        let videos = [];
        try {
            const response = await fetch(`${this.apiUrl}/analytics/summary`);
            if (response.ok) {
                const summary = await response.json();
                if (summary.status === 'success' && summary.data && summary.data.videos && summary.data.videos.list) {
                    videos = summary.data.videos.list; // Expecting [{id, filename}, ...]
                }
            }
        } catch (err) {
            console.warn('Could not fetch video list for chat dropdown:', err);
        }

        // Clear existing options except first one
        while (chatVideoSelect.children.length > 1) {
            chatVideoSelect.removeChild(chatVideoSelect.lastChild);
        }

        // Add video options
        videos.forEach(video => {
            const option = document.createElement('option');
            option.value = video.filename;
            option.textContent = video.filename.length > 30 ? video.filename.substring(0, 30) + '...' : video.filename;
            option.setAttribute('data-video-id', video.id);
            chatVideoSelect.appendChild(option);
        });
    }

    async sendChatMessage() {
        const chatInput = document.getElementById('chatInput');
        const chatVideoSelect = document.getElementById('chatVideoSelect');
        const message = chatInput.value.trim();
        if (!message) return;

        const selectedOption = chatVideoSelect.options[chatVideoSelect.selectedIndex];
        const selectedVideo = selectedOption ? selectedOption.value : '';
        const selectedVideoId = selectedOption ? selectedOption.getAttribute('data-video-id') : '';

        // Add user message to chat
        this.addChatMessage(message, 'user');
        chatInput.value = '';
        document.getElementById('sendChatBtn').disabled = true;
        this.addChatMessage('Thinking...', 'ai', 'thinking');

        try {
            const formData = new FormData();
            formData.append('message', message);
            if (selectedVideo) {
                formData.append('video_filename', selectedVideo);
            }
            if (selectedVideoId) {
                formData.append('video_id', selectedVideoId);
            }

            const response = await fetch(`${this.apiUrl}/chat`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.removeChatMessage('thinking');
            this.addChatMessage(data.response, 'ai');
        } catch (error) {
            console.error('Chat error:', error);
            this.removeChatMessage('thinking');
            this.addChatMessage('Sorry, I encountered an error while processing your message. Please try again.', 'ai', 'error');
        }
    }

    addChatMessage(text, sender, messageId = null) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        
        if (messageId) {
            messageDiv.id = `chat-${messageId}`;
        }
        
        if (sender === 'user') {
            messageDiv.className = 'flex justify-end';
            messageDiv.innerHTML = `
                <div class="bg-blue-500 text-white rounded-lg px-3 py-2 max-w-xs lg:max-w-md">
                    <p class="text-sm">${text}</p>
                </div>
            `;
        } else {
            const iconClass = messageId === 'thinking' ? 'fa-spinner fa-spin' : 
                            messageId === 'error' ? 'fa-exclamation-triangle text-red-500' : 
                            'fa-robot text-blue-500';
            
            messageDiv.className = 'flex justify-start';
            messageDiv.innerHTML = `
                <div class="bg-gray-200 text-gray-800 rounded-lg px-3 py-2 max-w-xs lg:max-w-md">
                    <div class="flex items-start space-x-2">
                        <i class="fas ${iconClass} mt-1 flex-shrink-0"></i>
                        <p class="text-sm">${text}</p>
                    </div>
                </div>
            `;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    removeChatMessage(messageId) {
        const message = document.getElementById(`chat-${messageId}`);
        if (message) {
            message.remove();
        }
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new DABTTBAI();
});