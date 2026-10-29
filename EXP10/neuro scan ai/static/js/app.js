/**
 * NeuroScan AI - Medical Machine Learning Web Application Frontend Controller
 * Handles One-Frame Pipeline, Model Training, Evaluation, and Live Code Compiler.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State management
    const state = {
        currentSample: "normal_sample_1.png",
        uploadedFile: null,
        activeStage: "original",
        stagesData: {},
        radarChart: null,
        rocChart: null,
        codeTemplates: {},
        isProcessing: false
    };

    // DOM Elements
    const elements = {
        // Nav tabs
        navTabs: document.querySelectorAll(".nav-tab"),
        frameSections: document.querySelectorAll(".frame-section"),

        // Dropzone & Samples
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("file-input"),
        samplePills: document.querySelectorAll(".sample-pill"),
        btnRunPipeline: document.getElementById("btn-run-pipeline"),

        // Stepper
        stepNodes: [
            document.getElementById("step-node-1"),
            document.getElementById("step-node-2"),
            document.getElementById("step-node-3"),
            document.getElementById("step-node-4"),
            document.getElementById("step-node-5")
        ],
        stepLines: [
            document.getElementById("step-line-1"),
            document.getElementById("step-line-2"),
            document.getElementById("step-line-3"),
            document.getElementById("step-line-4")
        ],

        // Preprocessor display & filmstrip
        mainDisplayImg: document.getElementById("main-display-img"),
        focusStageTitle: document.getElementById("focus-stage-title"),
        stageThumbs: document.querySelectorAll(".stage-thumb"),

        // Diagnosis banner & findings
        diagnosisBanner: document.getElementById("diagnosis-banner"),
        diagnosisIcon: document.getElementById("diagnosis-icon"),
        diagnosisLabel: document.getElementById("diagnosis-label"),
        diagnosisDesc: document.getElementById("diagnosis-desc"),
        diagnosisConf: document.getElementById("diagnosis-conf"),
        probRatio: document.getElementById("prob-ratio"),
        barNormal: document.getElementById("bar-normal"),
        barTumor: document.getElementById("bar-tumor"),
        findingRoi: document.getElementById("finding-roi"),
        findingArea: document.getElementById("finding-area"),
        findingBbox: document.getElementById("finding-bbox"),
        verdictModelName: document.getElementById("verdict-model-name"),
        activeAlgoBadge: document.getElementById("active-algo-badge"),

        // Feature sub-tabs
        subTabs: document.querySelectorAll(".sub-tab"),
        subPanels: document.querySelectorAll(".subpanel"),

        // Feature cells
        fContrast: document.getElementById("f-contrast"),
        fDissimilarity: document.getElementById("f-dissimilarity"),
        fHomogeneity: document.getElementById("f-homogeneity"),
        fEnergy: document.getElementById("f-energy"),
        fCorrelation: document.getElementById("f-correlation"),
        fAsm: document.getElementById("f-asm"),
        fMean: document.getElementById("f-mean"),
        fStd: document.getElementById("f-std"),
        fVariance: document.getElementById("f-variance"),
        fSkewness: document.getElementById("f-skewness"),
        fKurtosis: document.getElementById("f-kurtosis"),
        fEntropy: document.getElementById("f-entropy"),
        fArea: document.getElementById("f-area"),
        fPerimeter: document.getElementById("f-perimeter"),
        fCircularity: document.getElementById("f-circularity"),
        fSolidity: document.getElementById("f-solidity"),
        fRatio: document.getElementById("f-ratio"),

        // Training elements
        selectAlgorithm: document.getElementById("select-algorithm"),
        inputTestSplit: document.getElementById("input-test-split"),
        inputCVal: document.getElementById("input-c-val"),
        valCDisplay: document.getElementById("val-c-display"),
        inputNEstimators: document.getElementById("input-n-estimators"),
        valTreesDisplay: document.getElementById("val-trees-display"),
        inputKNeighbors: document.getElementById("input-k-neighbors"),
        valKDisplay: document.getElementById("val-k-display"),
        groupCVal: document.getElementById("group-c-val"),
        groupEstimators: document.getElementById("group-estimators"),
        groupNeighbors: document.getElementById("group-neighbors"),
        btnTrainModel: document.getElementById("btn-train-model"),
        trainFeedback: document.getElementById("train-feedback"),

        // Metrics & Confusion Matrix
        evalAcc: document.getElementById("eval-acc"),
        evalPrec: document.getElementById("eval-prec"),
        evalRec: document.getElementById("eval-rec"),
        evalSpec: document.getElementById("eval-spec"),
        evalF1: document.getElementById("eval-f1"),
        evalAuc: document.getElementById("eval-auc"),
        cmTn: document.getElementById("cm-tn"),
        cmFp: document.getElementById("cm-fp"),
        cmFn: document.getElementById("cm-fn"),
        cmTp: document.getElementById("cm-tp"),
        rocAucBadge: document.getElementById("roc-auc-badge"),

        // Compiler & Editor
        codeTemplateSelect: document.getElementById("code-template-select"),
        codeEditor: document.getElementById("code-editor"),
        btnRunCode: document.getElementById("btn-run-code"),
        btnResetCode: document.getElementById("btn-reset-code"),
        execStatus: document.getElementById("exec-status"),
        execTiming: document.getElementById("exec-timing"),
        terminalOutput: document.getElementById("terminal-output"),
        copyBtns: document.querySelectorAll(".btn-copy")
    };

    // ==================== INITIALIZATION ====================

    function init() {
        setupNavigation();
        setupDropzone();
        setupSamplePills();
        setupFeatureSubtabs();
        setupFilmstrip();
        setupTrainingControls();
        setupCodePlayground();
        setupCopyButtons();
        initCharts();

        // Run baseline evaluation
        fetchEvaluation();

        // Load templates and initial MRI sample
        fetchCodeTemplates();
        runPipeline();
    }

    // ==================== TAB NAVIGATION ====================

    function setupNavigation() {
        elements.navTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                const targetId = tab.getAttribute("data-tab");
                elements.navTabs.forEach(t => t.classList.remove("active"));
                elements.frameSections.forEach(s => s.classList.remove("active"));

                tab.classList.add("active");
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add("active");
                }
            });
        });
    }

    // ==================== DROPZONE & SAMPLE SELECTION ====================

    function setupDropzone() {
        const dz = elements.dropzone;
        const fi = elements.fileInput;

        dz.addEventListener("click", () => fi.click());

        fi.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                state.uploadedFile = e.target.files[0];
                state.currentSample = null;
                elements.samplePills.forEach(p => p.classList.remove("active"));
                runPipeline();
            }
        });

        dz.addEventListener("dragover", (e) => {
            e.preventDefault();
            dz.classList.add("dragover");
        });

        dz.addEventListener("dragleave", () => {
            dz.classList.remove("dragover");
        });

        dz.addEventListener("drop", (e) => {
            e.preventDefault();
            dz.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                state.uploadedFile = e.dataTransfer.files[0];
                state.currentSample = null;
                elements.samplePills.forEach(p => p.classList.remove("active"));
                runPipeline();
            }
        });

        elements.btnRunPipeline.addEventListener("click", runPipeline);
    }

    function setupSamplePills() {
        elements.samplePills.forEach(pill => {
            pill.addEventListener("click", () => {
                elements.samplePills.forEach(p => p.classList.remove("active"));
                pill.classList.add("active");
                state.currentSample = pill.getAttribute("data-sample");
                state.uploadedFile = null;
                elements.fileInput.value = "";
                runPipeline();
            });
        });
    }

    // ==================== FILMSTRIP & STAGE FOCUS ====================

    const stageTitles = {
        original: "1. Original Axial Brain MRI Scan",
        grayscale: "2. Grayscale Intensity Representation",
        gaussian_blur: "3. Gaussian Filter Noise Smoothing (5x5 Kernel)",
        clahe: "4. Contrast-Limited Adaptive Histogram Equalization (CLAHE)",
        brain_mask: "5. Otsu Skull-Stripped Brain Parenchyma Mask",
        tumor_mask: "6. Abnormal Hyper-Intense ROI Candidate Mask",
        heatmap: "7. Jet Colormap Density Heatmap",
        overlay: "8. Diagnostic ROI Localization Bounding Box Overlay"
    };

    function setupFilmstrip() {
        elements.stageThumbs.forEach(thumb => {
            thumb.addEventListener("click", () => {
                const stage = thumb.getAttribute("data-stage");
                selectStage(stage);
            });
        });
    }

    function selectStage(stage) {
        state.activeStage = stage;
        elements.stageThumbs.forEach(t => {
            t.classList.toggle("active", t.getAttribute("data-stage") === stage);
        });

        if (state.stagesData && state.stagesData[stage]) {
            elements.mainDisplayImg.src = state.stagesData[stage];
        } else if (stage === "original" && state.currentSample) {
            elements.mainDisplayImg.src = `/static/samples/${state.currentSample}`;
        }
        elements.focusStageTitle.textContent = stageTitles[stage] || stage;
    }

    // ==================== FEATURE TABS ====================

    function setupFeatureSubtabs() {
        elements.subTabs.forEach(tab => {
            tab.addEventListener("click", () => {
                const sub = tab.getAttribute("data-sub");
                elements.subTabs.forEach(t => t.classList.remove("active"));
                elements.subPanels.forEach(p => p.classList.remove("active"));

                tab.classList.add("active");
                const panel = document.getElementById(`subpanel-${sub}`);
                if (panel) panel.classList.add("active");
            });
        });
    }

    // ==================== PIPELINE EXECUTION ====================

    async function runPipeline() {
        if (state.isProcessing) return;
        state.isProcessing = true;

        // Animate Stepper
        animateStepper(1);
        elements.btnRunPipeline.disabled = true;
        elements.btnRunPipeline.innerHTML = `<span class="status-indicator"></span> Processing MRI Pipeline...`;

        const formData = new FormData();
        if (state.uploadedFile) {
            formData.append("file", state.uploadedFile);
        } else if (state.currentSample) {
            formData.append("sample", state.currentSample);
        }

        try {
            animateStepper(2);
            const response = await fetch("/api/process_image", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            if (!data.success) {
                alert("Error processing MRI image: " + data.error);
                return;
            }

            animateStepper(3);
            state.stagesData = data.stages;

            // Update Thumbnails in Filmstrip
            updateFilmstrip(data.stages);

            // Select overlay if tumor detected, else clahe/original
            const preferredStage = (data.prediction && data.prediction.predicted_class === 1) ? "overlay" : "clahe";
            selectStage(preferredStage);

            // Populate Extracted Features
            populateFeatures(data.features);

            // Update Radar Chart
            updateRadarChart(data.features);

            animateStepper(4);
            // Update Clinical Diagnosis Verdict
            if (data.prediction) {
                updateDiagnosisVerdict(data.prediction, data.metadata);
            }

            animateStepper(5);

        } catch (err) {
            console.error("Pipeline request failed:", err);
            alert("Failed to execute pipeline: " + err.message);
        } finally {
            state.isProcessing = false;
            elements.btnRunPipeline.disabled = false;
            elements.btnRunPipeline.innerHTML = `
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="5 3 19 12 5 21 5 3"/>
                </svg>
                Execute 5-Step ML Pipeline
            `;
        }
    }

    function animateStepper(activeStep) {
        elements.stepNodes.forEach((node, idx) => {
            const stepNum = idx + 1;
            node.classList.remove("active", "completed");
            if (stepNum < activeStep) {
                node.classList.add("completed");
            } else if (stepNum === activeStep) {
                node.classList.add("active");
            }
        });

        elements.stepLines.forEach((line, idx) => {
            line.classList.toggle("active", idx < activeStep - 1);
        });
    }

    function updateFilmstrip(stages) {
        if (!stages) return;
        document.getElementById("thumb-original").src = stages.original || "";
        document.getElementById("thumb-grayscale").src = stages.grayscale || "";
        document.getElementById("thumb-blur").src = stages.gaussian_blur || "";
        document.getElementById("thumb-clahe").src = stages.clahe || "";
        document.getElementById("thumb-mask").src = stages.brain_mask || "";
        document.getElementById("thumb-tumor").src = stages.tumor_mask || "";
        document.getElementById("thumb-heatmap").src = stages.heatmap || "";
        document.getElementById("thumb-overlay").src = stages.overlay || "";
    }

    function populateFeatures(f) {
        if (!f) return;
        // GLCM
        elements.fContrast.textContent = (f.glcm_contrast || 0).toFixed(4);
        elements.fDissimilarity.textContent = (f.glcm_dissimilarity || 0).toFixed(4);
        elements.fHomogeneity.textContent = (f.glcm_homogeneity || 0).toFixed(4);
        elements.fEnergy.textContent = (f.glcm_energy || 0).toFixed(4);
        elements.fCorrelation.textContent = (f.glcm_correlation || 0).toFixed(4);
        elements.fAsm.textContent = (f.glcm_asm || 0).toFixed(6);

        // Intensity
        elements.fMean.textContent = (f.intensity_mean || 0).toFixed(2);
        elements.fStd.textContent = (f.intensity_std || 0).toFixed(2);
        elements.fVariance.textContent = (f.intensity_variance || 0).toFixed(2);
        elements.fSkewness.textContent = (f.intensity_skewness || 0).toFixed(4);
        elements.fKurtosis.textContent = (f.intensity_kurtosis || 0).toFixed(4);
        elements.fEntropy.textContent = (f.shannon_entropy || 0).toFixed(4);

        // Shape
        elements.fArea.textContent = `${f.tumor_area_px || 0} px`;
        elements.fPerimeter.textContent = `${f.tumor_perimeter_px || 0} px`;
        elements.fCircularity.textContent = (f.circularity || 0).toFixed(4);
        elements.fSolidity.textContent = (f.solidity || 0).toFixed(4);
        elements.fRatio.textContent = `${(f.area_ratio_pct || 0).toFixed(3)}%`;
    }

    function updateDiagnosisVerdict(pred, meta) {
        const isTumor = pred.predicted_class === 1;

        elements.verdictModelName.textContent = pred.model_algorithm;
        elements.activeAlgoBadge.textContent = pred.model_algorithm;

        if (isTumor) {
            elements.diagnosisBanner.classList.add("positive");
            elements.diagnosisLabel.textContent = "TUMOR DETECTED (POSITIVE)";
            elements.diagnosisLabel.style.color = "#f43f5e";
            elements.diagnosisDesc.textContent = "Hyper-intense lesion identified with structural tissue distortion.";
            elements.diagnosisIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
            `;
        } else {
            elements.diagnosisBanner.classList.remove("positive");
            elements.diagnosisLabel.textContent = "NORMAL / HEALTHY (NEGATIVE)";
            elements.diagnosisLabel.style.color = "#10b981";
            elements.diagnosisDesc.textContent = "No focal hyper-intense mass detected. Ventricular symmetry preserved.";
            elements.diagnosisIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <polyline points="20 6 9 17 4 12"/>
                </svg>
            `;
        }

        elements.diagnosisConf.textContent = `${pred.confidence}%`;

        // Probabilities
        const pNorm = Math.round((pred.probabilities.normal || 0) * 100);
        const pTumor = Math.round((pred.probabilities.tumor || 0) * 100);
        elements.probRatio.textContent = `Normal: ${pNorm}% | Tumor: ${pTumor}%`;
        elements.barNormal.style.width = `${pNorm}%`;
        elements.barTumor.style.width = `${pTumor}%`;

        // Findings
        elements.findingRoi.textContent = meta.has_tumor_candidate ? "Yes (Focal Lesion)" : "None";
        elements.findingArea.textContent = `${meta.max_contour_area_px || 0} px`;
        if (meta.bounding_box) {
            const b = meta.bounding_box;
            elements.findingBbox.textContent = `[X:${b.x}, Y:${b.y}, W:${b.w}, H:${b.h}]`;
        } else {
            elements.findingBbox.textContent = "N/A";
        }
    }

    // ==================== CHARTS (RADAR & ROC) ====================

    function initCharts() {
        // 1. Feature Radar Chart
        const radarCtx = document.getElementById("featureRadarChart").getContext("2d");
        state.radarChart = new Chart(radarCtx, {
            type: "radar",
            data: {
                labels: ["Contrast", "Homogeneity", "Energy", "Correlation", "Mean Int.", "Solidity"],
                datasets: [{
                    label: "Normalized Feature Profile",
                    data: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    backgroundColor: "rgba(6, 182, 212, 0.25)",
                    borderColor: "#06b6d4",
                    borderWidth: 2,
                    pointBackgroundColor: "#3b82f6",
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: "rgba(255, 255, 255, 0.08)" },
                        grid: { color: "rgba(255, 255, 255, 0.08)" },
                        pointLabels: {
                            color: "#94a3b8",
                            font: { family: "Outfit", size: 10 }
                        },
                        ticks: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });

        // 2. ROC Curve Chart
        const rocCtx = document.getElementById("rocChart").getContext("2d");
        state.rocChart = new Chart(rocCtx, {
            type: "line",
            data: {
                labels: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                datasets: [
                    {
                        label: "Classifier ROC",
                        data: [
                            { x: 0, y: 0 },
                            { x: 0.05, y: 0.90 },
                            { x: 0.10, y: 0.95 },
                            { x: 0.20, y: 0.98 },
                            { x: 1.0, y: 1.0 }
                        ],
                        borderColor: "#06b6d4",
                        backgroundColor: "rgba(6, 182, 212, 0.15)",
                        fill: true,
                        tension: 0.2,
                        pointRadius: 4,
                        pointBackgroundColor: "#06b6d4"
                    },
                    {
                        label: "Random Chance",
                        data: [
                            { x: 0, y: 0 },
                            { x: 1, y: 1 }
                        ],
                        borderColor: "rgba(255, 255, 255, 0.25)",
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "linear",
                        min: 0,
                        max: 1,
                        title: { display: true, text: "False Positive Rate (1 - Specificity)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#64748b" }
                    },
                    y: {
                        type: "linear",
                        min: 0,
                        max: 1,
                        title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", color: "#94a3b8" },
                        grid: { color: "rgba(255, 255, 255, 0.06)" },
                        ticks: { color: "#64748b" }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: "#cbd5e1", font: { family: "Outfit" } }
                    }
                }
            }
        });
    }

    function updateRadarChart(f) {
        if (!state.radarChart || !f) return;
        // Normalize roughly to 0..1 scale for visual balance
        const c = Math.min(1, (f.glcm_contrast || 0) / 120);
        const h = Math.min(1, f.glcm_homogeneity || 0);
        const e = Math.min(1, (f.glcm_energy || 0) * 4);
        const corr = Math.max(0, (f.glcm_correlation || 0));
        const m = Math.min(1, (f.intensity_mean || 0) / 180);
        const s = Math.min(1, f.solidity || 0);

        state.radarChart.data.datasets[0].data = [c, h, e, corr, m, s];
        state.radarChart.update();
    }

    // ==================== MODEL TRAINING & EVALUATION ====================

    function setupTrainingControls() {
        // Algorithm switch shows/hides hyperparameters
        elements.selectAlgorithm.addEventListener("change", (e) => {
            const algo = e.target.value;
            elements.groupCVal.style.display = (algo.includes("SVM")) ? "flex" : "none";
            elements.groupEstimators.style.display = (algo === "Random Forest") ? "flex" : "none";
            elements.groupNeighbors.style.display = (algo === "K-Nearest Neighbors") ? "flex" : "none";
        });

        // Sliders value display
        elements.inputCVal.addEventListener("input", (e) => {
            elements.valCDisplay.textContent = e.target.value;
        });

        elements.inputNEstimators.addEventListener("input", (e) => {
            elements.valTreesDisplay.textContent = e.target.value;
        });

        elements.inputKNeighbors.addEventListener("input", (e) => {
            elements.valKDisplay.textContent = e.target.value;
        });

        // Train button
        elements.btnTrainModel.addEventListener("click", trainModel);
    }

    async function trainModel() {
        elements.btnTrainModel.disabled = true;
        elements.trainFeedback.textContent = "Training classifier on balanced MRI feature dataset...";
        elements.trainFeedback.style.color = "#06b6d4";

        const payload = {
            algorithm: elements.selectAlgorithm.value,
            test_size: parseFloat(elements.inputTestSplit.value),
            c_val: parseFloat(elements.inputCVal.value),
            n_estimators: parseInt(elements.inputNEstimators.value),
            k_neighbors: parseInt(elements.inputKNeighbors.value)
        };

        try {
            const response = await fetch("/api/train", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            const data = await response.json();
            if (!data.success) {
                elements.trainFeedback.textContent = "Training Error: " + data.error;
                elements.trainFeedback.style.color = "#f43f5e";
                return;
            }

            elements.trainFeedback.textContent = `Trained ${data.algorithm} successfully on ${data.train_samples} samples!`;
            elements.trainFeedback.style.color = "#10b981";
            elements.activeAlgoBadge.textContent = data.algorithm;
            elements.verdictModelName.textContent = data.algorithm;

            // Update evaluation UI
            renderEvaluation(data.evaluation);

            // Re-run pipeline on active image so inference uses new model
            runPipeline();

        } catch (err) {
            elements.trainFeedback.textContent = "Network error: " + err.message;
            elements.trainFeedback.style.color = "#f43f5e";
        } finally {
            elements.btnTrainModel.disabled = false;
        }
    }

    async function fetchEvaluation() {
        try {
            const res = await fetch("/api/evaluate");
            const data = await res.json();
            if (data.success && data.evaluation) {
                renderEvaluation(data.evaluation);
            }
        } catch (err) {
            console.error("Failed to fetch evaluation:", err);
        }
    }

    function renderEvaluation(evalData) {
        if (!evalData) return;
        const m = evalData.metrics;
        const cm = evalData.confusion_matrix;
        const roc = evalData.roc_curve;

        // Metric tiles
        elements.evalAcc.textContent = `${m.accuracy}%`;
        elements.evalPrec.textContent = `${m.precision}%`;
        elements.evalRec.textContent = `${m.recall}%`;
        elements.evalSpec.textContent = `${m.specificity}%`;
        elements.evalF1.textContent = `${m.f1_score}%`;
        elements.evalAuc.textContent = m.roc_auc;
        elements.rocAucBadge.textContent = `AUC: ${m.roc_auc}`;

        // Confusion Matrix
        elements.cmTn.textContent = cm.tn;
        elements.cmFp.textContent = cm.fp;
        elements.cmFn.textContent = cm.fn;
        elements.cmTp.textContent = cm.tp;

        // ROC Curve Chart
        if (state.rocChart && roc) {
            const rocPoints = roc.fpr.map((fprVal, idx) => ({
                x: fprVal,
                y: roc.tpr[idx]
            }));

            state.rocChart.data.datasets[0].data = rocPoints;
            state.rocChart.update();
        }
    }

    // ==================== CODE PLAYGROUND & COMPILER ====================

    async function fetchCodeTemplates() {
        try {
            const res = await fetch("/api/code_templates");
            const data = await res.json();
            if (data.success) {
                state.codeTemplates = data.templates;
                // Pre-populate pipeline code
                elements.codeEditor.value = state.codeTemplates.pipeline || "";
            }
        } catch (err) {
            console.error("Failed to load templates:", err);
        }
    }

    function setupCodePlayground() {
        // Change template
        elements.codeTemplateSelect.addEventListener("change", (e) => {
            const t = e.target.value;
            if (state.codeTemplates[t]) {
                elements.codeEditor.value = state.codeTemplates[t];
            }
        });

        // Reset template
        elements.btnResetCode.addEventListener("click", () => {
            const t = elements.codeTemplateSelect.value;
            if (state.codeTemplates[t]) {
                elements.codeEditor.value = state.codeTemplates[t];
            }
        });

        // Run Code
        elements.btnRunCode.addEventListener("click", runCode);

        // Allow Tab key indentation inside textarea
        elements.codeEditor.addEventListener("keydown", function(e) {
            if (e.key === "Tab") {
                e.preventDefault();
                const start = this.selectionStart;
                const end = this.selectionEnd;
                this.value = this.value.substring(0, start) + "    " + this.value.substring(end);
                this.selectionStart = this.selectionEnd = start + 4;
            }
        });
    }

    async function runCode() {
        const code = elements.codeEditor.value;
        if (!code.trim()) {
            alert("Editor is empty. Please enter or select a Python code snippet.");
            return;
        }

        elements.btnRunCode.disabled = true;
        elements.execStatus.textContent = "Compiling & Running...";
        elements.execStatus.style.color = "#06b6d4";
        elements.terminalOutput.textContent = ">>> Compiling Python program in temporary execution environment...\n";

        const startTime = performance.now();

        try {
            const response = await fetch("/api/run_code", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ code: code })
            });

            const data = await response.json();
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
            elements.execTiming.textContent = `Elapsed: ${data.elapsed_seconds || elapsed}s`;

            if (data.success) {
                let out = "";
                if (data.stdout) out += data.stdout;
                if (data.stderr) out += "\n[STDERR]:\n" + data.stderr;
                if (!out) out = "[Program finished with exit code 0 and no output]";

                elements.terminalOutput.textContent = out;
                elements.execStatus.textContent = "Finished (Exit 0)";
                elements.execStatus.style.color = "#10b981";
            } else {
                elements.terminalOutput.textContent = `[EXECUTION ERROR]:\n${data.error}`;
                elements.execStatus.textContent = "Failed";
                elements.execStatus.style.color = "#f43f5e";
            }
        } catch (err) {
            elements.terminalOutput.textContent = `[NETWORK ERROR]:\n${err.message}`;
            elements.execStatus.textContent = "Network Error";
            elements.execStatus.style.color = "#f43f5e";
        } finally {
            elements.btnRunCode.disabled = false;
        }
    }

    // ==================== COPY BUTTONS ====================

    function setupCopyButtons() {
        elements.copyBtns.forEach(btn => {
            btn.addEventListener("click", () => {
                const text = btn.getAttribute("data-copy");
                if (text) {
                    navigator.clipboard.writeText(text).then(() => {
                        const original = btn.textContent;
                        btn.textContent = "Copied!";
                        btn.style.color = "#10b981";
                        setTimeout(() => {
                            btn.textContent = original;
                            btn.style.color = "";
                        }, 2000);
                    });
                }
            });
        });
    }

    // Run main init
    init();
});
