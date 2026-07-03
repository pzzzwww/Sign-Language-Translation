/**
 * 基于Vision Transformer的手语识别生成语音系统 — 前端主逻辑
 * 每个人调用自己的本地摄像头 → 发帧到服务端 → 服务端识别 → 返回结果
 */
(function () {
    "use strict";

    var els = {
        videoFeed:      document.getElementById("video-feed"),
        detectionCanvas: document.getElementById("detection-canvas"),
        placeholder:    document.getElementById("placeholder"),

        guessDisplay:   document.getElementById("guess-display"),

        statusDot:      document.getElementById("status-dot"),
        statusText:     document.getElementById("status-text"),
        flowBadge:      document.getElementById("flow-badge"),
        loadingOvl:     document.getElementById("loading-overlay"),
        loadingText:    document.getElementById("loading-text"),

        btnCapture:     document.getElementById("btn-capture"),
        btnStop:        document.getElementById("btn-stop"),
        btnClear:       document.getElementById("btn-clear"),

        tokenList:      document.getElementById("token-list"),
        tokenCount:     document.getElementById("token-count"),
        sentPlace:      document.getElementById("sentence-placeholder"),
        errorFeedback:  document.getElementById("error-feedback"),
        errorMessage:   document.getElementById("error-message"),
        transEditor:    document.getElementById("translate-editor"),
        transTextarea:  document.getElementById("translate-textarea"),
        btnConfirm:     document.getElementById("btn-confirm"),
        transConfirm:   document.getElementById("translate-confirmed"),
        confirmText:    document.getElementById("confirmed-text"),
        btnGenerate:    document.getElementById("btn-generate"),
        audioSection:   document.getElementById("audio-player-section"),
        btnPlay:        document.getElementById("btn-play"),
        audioDuration:  document.getElementById("audio-duration"),

        logArea:        document.getElementById("log-area"),
        historyList:    document.getElementById("history-list"),
        btnRefreshHist: document.getElementById("btn-refresh-history"),
    };

    var ws = null;
    var connected = false;
    var capturing = false;
    var flowState = "idle";
    var currentTokens = [];
    var currentHistoryId = null;
    var currentDuration = 0;
    var canvasCtx = null;
    var lastDetection = [];

    // 本地摄像头
    var localStream = null;
    var frameTimer = null;
    var offscreenCanvas = null;
    var offscreenCtx = null;
    var FRAME_INTERVAL_MS = 250;  // ~4 fps（公网优化）

    var STATE = {
        IDLE:             "idle",
        CAPTURING:        "capturing",
        TRANSLATING:      "translating",
        WAITING_GENERATE: "waiting_generate",
        GENERATING_AUDIO: "generating_audio",
        AUDIO_READY:      "audio_ready",
    };

    // ==============================
    // 状态机
    // ==============================

    function setFlowState(newState) {
        flowState = newState;
        updateUI();
        updateButtons();
        var badgeMap = {};
        badgeMap[STATE.WAITING_GENERATE] = "文本已确认";
        badgeMap[STATE.AUDIO_READY]      = "语音已生成";
        if (badgeMap[newState]) {
            els.flowBadge.textContent = badgeMap[newState];
            els.flowBadge.style.display = "inline";
        } else {
            els.flowBadge.style.display = "none";
        }
    }

    function updateUI() {
        var loadingStates = [STATE.TRANSLATING, STATE.GENERATING_AUDIO];
        els.loadingOvl.style.display = loadingStates.indexOf(flowState) !== -1 ? "flex" : "none";

        var isWaitingGen     = flowState === STATE.WAITING_GENERATE;
        var isAudioReady     = flowState === STATE.AUDIO_READY;

        els.sentPlace.style.display     = (!isWaitingGen && !isAudioReady) ? "block" : "none";
        els.transEditor.style.display   = "block";
        els.transConfirm.style.display  = (isWaitingGen || isAudioReady) ? "block" : "none";
        els.audioSection.style.display  = isAudioReady ? "block" : "none";
    }

    function updateStatus(state, message) {
        var dot = els.statusDot;
        var text = els.statusText;
        dot.className = "status-dot";
        switch (state) {
            case "capturing":
                dot.classList.add("active");
                text.textContent = message || "采集中...";
                break;
            case "recognizing":
            case "translating":
            case "generating_audio":
            case "loading_model":
                dot.classList.add("busy");
                text.textContent = message || "处理中...";
                break;
            case "error":
                dot.classList.add("error");
                text.textContent = message || "错误";
                break;
            default:
                text.textContent = message || "就绪";
        }
    }

    function updateButtons() {
        var isBusy = [STATE.TRANSLATING, STATE.GENERATING_AUDIO].indexOf(flowState) !== -1;
        els.btnCapture.disabled   = !connected || isBusy || capturing;
        els.btnStop.disabled      = !connected || isBusy || !capturing;
        els.btnConfirm.disabled   = isBusy;
        els.btnGenerate.disabled  = isBusy || flowState !== STATE.WAITING_GENERATE;
        els.btnPlay.disabled      = isBusy || flowState !== STATE.AUDIO_READY;
    }

    // ==============================
    // WebSocket
    // ==============================

    function connect() {
        var proto = location.protocol === "https:" ? "wss:" : "ws:";
        var url = proto + "//" + location.host + "/ws/stream";
        ws = new WebSocket(url);

        ws.onopen = function () {
            connected = true;
            updateStatus("idle", "已连接，等待模型就绪...");
            log("WebSocket 已连接");
            updateButtons();
        };

        ws.onclose = function () {
            connected = false;
            capturing = false;
            stopLocalCamera();
            setFlowState(STATE.IDLE);
            log("WebSocket 已断开，3秒后重连...");
            updateButtons();
            setTimeout(connect, 3000);
        };

        ws.onmessage = function (event) {
            try {
                handleMessage(JSON.parse(event.data));
            } catch (e) {
                log("消息解析失败: " + e.message, "error");
            }
        };
    }

    // ==============================
    // 本地摄像头
    // ==============================

    function startLocalCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            log("浏览器不支持摄像头访问", "error");
            updateStatus("error", "浏览器不支持摄像头");
            return;
        }
        navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
            audio: false,
        }).then(function (stream) {
            localStream = stream;
            els.videoFeed.srcObject = stream;
            els.videoFeed.style.display = "block";
            // 镜像翻转（照镜子效果）
            els.videoFeed.style.transform = "scaleX(-1)";
            els.detectionCanvas.style.transform = "scaleX(-1)";
            if (els.placeholder) els.placeholder.style.display = "none";

            // 初始化离屏 canvas（用于抓帧发送给服务端）
            offscreenCanvas = document.createElement("canvas");
            offscreenCtx = offscreenCanvas.getContext("2d");

            capturing = true;
            setFlowState(STATE.CAPTURING);
            updateStatus("capturing", "本地摄像头已启动");
            log("本地摄像头已启动");

            // 通知服务端开始新会话
            send("start_capture");

            // 启动帧发送循环
            startFrameLoop();
        }).catch(function (err) {
            log("摄像头启动失败: " + err.message, "error");
            updateStatus("error", "摄像头被拒绝");
            alert("无法打开摄像头: " + err.message + "\n\n请检查:\n1. 是否已允许浏览器摄像头权限\n2. 摄像头是否被其他程序占用");
        });
    }

    function stopLocalCamera() {
        if (frameTimer) { clearInterval(frameTimer); frameTimer = null; }
        if (localStream) {
            localStream.getTracks().forEach(function (t) { t.stop(); });
            localStream = null;
        }
        els.videoFeed.srcObject = null;
        els.videoFeed.style.display = "none";
        els.videoFeed.style.transform = "";
        if (els.detectionCanvas) {
            els.detectionCanvas.style.transform = "";
            if (canvasCtx) canvasCtx.clearRect(0, 0, els.detectionCanvas.width, els.detectionCanvas.height);
        }
        if (els.placeholder) els.placeholder.style.display = "block";
        offscreenCanvas = null;
        offscreenCtx = null;
        lastDetection = [];
    }

    function startFrameLoop() {
        if (frameTimer) clearInterval(frameTimer);
        frameTimer = setInterval(function () {
            if (!capturing || !offscreenCtx || !els.videoFeed.videoWidth) return;
            var vw = els.videoFeed.videoWidth;
            var vh = els.videoFeed.videoHeight;
            offscreenCanvas.width = vw;
            offscreenCanvas.height = vh;
            offscreenCtx.drawImage(els.videoFeed, 0, 0, vw, vh);
            var jpeg = offscreenCanvas.toDataURL("image/jpeg", 0.4);
            var base64 = jpeg.substring(jpeg.indexOf(",") + 1);
            send("process_frame", { data: base64 });
        }, FRAME_INTERVAL_MS);
    }

    // ==============================
    // 消息处理
    // ==============================

    function handleMessage(msg) {
        switch (msg.type) {
            case "status":
                updateStatus(msg.state, msg.message || "");
                break;

            case "detection":
                lastDetection = msg.data || [];
                drawDetectionOverlay(msg.data);
                break;

            case "guess_update":
                updateSubtitle(msg.guess);
                break;

            case "tokens_append":
                currentTokens = msg.total_tokens || [];
                renderTokens(currentTokens);
                if (msg.data && msg.data.length) {
                    log("确认Token: " + msg.data.join(", ") + " (共" + msg.count + "个)");
                }
                break;

            case "tokens_clear":
                currentTokens = [];
                lastDetection = [];
                if (els.guessDisplay) els.guessDisplay.innerHTML = '<span class="guess-placeholder">等待手势...</span>';
                renderTokens([]);
                els.transTextarea.value = "";
                break;

            case "token_deleted":
                currentTokens = msg.tokens || [];
                renderTokens(currentTokens);
                log("Token #" + msg.index + " 已删除");
                break;

            case "auto_translate":
                els.transTextarea.value = msg.data || "";
                els.sentPlace.style.display = "none";
                els.transEditor.style.display = "block";
                break;

            case "audio_ready":
                currentHistoryId = msg.history_id;
                currentDuration  = msg.duration_sec || 0;
                setFlowState(STATE.AUDIO_READY);
                if (currentDuration > 0) {
                    els.audioDuration.textContent = "约 " + Math.round(currentDuration) + " 秒";
                }
                log("语音已生成 (ID: " + currentHistoryId + ")");
                loadHistory();
                break;

            case "error":
                log("错误 [" + msg.code + "]: " + msg.message, "error");
                updateStatus("error", msg.message);
                showError(msg.message);
                if (flowState === STATE.TRANSLATING) {
                    setFlowState(STATE.CAPTURING);
                } else if (flowState === STATE.GENERATING_AUDIO) {
                    setFlowState(STATE.WAITING_GENERATE);
                }
                break;

            case "pong":
                break;
        }
    }

    // ==============================
    // 控制指令
    // ==============================

    function send(action, extra) {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            log("WebSocket 未连接", "error");
            return;
        }
        var payload = { action: action };
        if (extra) {
            for (var k in extra) payload[k] = extra[k];
        }
        ws.send(JSON.stringify(payload));
    }

    function startCapture() {
        startLocalCamera();
    }

    function stopCapture() {
        send("stop");
        stopLocalCamera();
        capturing = false;
        setFlowState(STATE.IDLE);
        updateStatus("idle", "已停止");
    }

    function confirmTranslate() {
        var text = els.transTextarea.value.trim();
        if (!text) { log("翻译文本不能为空", "error"); return; }
        els.confirmText.textContent = text;
        setFlowState(STATE.WAITING_GENERATE);
        send("confirm_translate", { text: text });
        log("文本已确认: " + text);
    }

    function generateAudio() {
        setFlowState(STATE.GENERATING_AUDIO);
        updateStatus("generating_audio", "正在生成语音...");
        send("generate_audio");
    }

    function playAudio() {
        if (!currentHistoryId) { log("没有可播放的音频", "error"); return; }
        var audioUrl = "/audio/" + currentHistoryId + ".wav";
        var audio = new Audio(audioUrl);
        audio.onerror = function () { log("音频播放失败", "error"); };
        audio.play().then(function () { log("开始播放语音"); }).catch(function (e) {
            log("音频播放失败: " + e.message, "error");
        });
    }

    // ==============================
    // 性别显示
    // ==============================

    function showError(message) {
        if (!els.errorFeedback || !els.errorMessage) return;
        els.errorFeedback.style.display = "flex";
        els.errorMessage.textContent = "生成失败：" + message;
    }

    function hideError() {
        if (els.errorFeedback) els.errorFeedback.style.display = "none";
    }

    // ==============================
    // 渲染
    // ==============================

    function renderTokens(tokens) {
        if (!tokens || !tokens.length) {
            els.tokenList.innerHTML = '<span style="color:#484f58;font-size:13px;">等待识别...</span>';
            if (els.tokenCount) els.tokenCount.textContent = "";
            return;
        }
        els.tokenList.innerHTML = "";
        tokens.forEach(function (t) {
            var span = document.createElement("span");
            span.className = "token-tag";
            span.textContent = t;
            els.tokenList.appendChild(span);
        });
        if (els.tokenCount) els.tokenCount.textContent = "(" + tokens.length + ")";
    }

    function updateSubtitle(guess) {
        if (!els.guessDisplay) return;
        els.guessDisplay.innerHTML = "";
        if (guess) {
            var span = document.createElement("span");
            span.className = "guess-token";
            span.textContent = guess;
            span.style.cursor = "pointer";
            span.title = "点击确认此手势";
            span.addEventListener("click", function () {
                send("confirm_token");
            });
            els.guessDisplay.appendChild(span);
        } else {
            els.guessDisplay.innerHTML = '<span class="guess-placeholder">等待手势...</span>';
        }
    }

    // ==============================
    // Canvas 检测框
    // ==============================

    function initCanvas() {
        if (els.detectionCanvas) canvasCtx = els.detectionCanvas.getContext("2d");
    }

    function drawDetectionOverlay(detectionData) {
        if (!canvasCtx || !els.videoFeed || !els.detectionCanvas) return;
        var canvas = els.detectionCanvas;
        var img = els.videoFeed;
        var ctx = canvasCtx;
        var displayW = img.clientWidth;
        var displayH = img.clientHeight;
        if (displayW === 0 || displayH === 0) return;
        canvas.width = displayW;
        canvas.height = displayH;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (!detectionData || !detectionData.length) return;

        var naturalW = img.videoWidth || displayW;
        var naturalH = img.videoHeight || displayH;
        var scaleX = displayW / naturalW;
        var scaleY = displayH / naturalH;

        detectionData.forEach(function (hand) {
            var bbox = hand.bbox;
            var conf = hand.confidence || 0;
            var token = hand.token || null;
            if (!bbox || bbox.length < 4) return;
            var bx = bbox[0], by = bbox[1], bw = bbox[2], bh = bbox[3];
            var sx = bx * scaleX, sy = by * scaleY, sw = bw * scaleX, sh = bh * scaleY;

            var color = "#ff0000";

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx, sy, sw, sh);

            var label = token || hand.handedness || "Hand";
            label += " " + conf.toFixed(2);
            ctx.font = "13px sans-serif";
            ctx.fillStyle = color;
            ctx.fillRect(sx, sy - 20, ctx.measureText(label).width + 8, 20);
            ctx.fillStyle = "#000";
            ctx.fillText(label, sx + 4, sy - 6);

            var landmarks = hand.landmarks || [];
            landmarks.forEach(function (p) {
                ctx.beginPath();
                ctx.arc(p[0] * scaleX, p[1] * scaleY, 3, 0, 2 * Math.PI);
                ctx.fillStyle = color;
                ctx.fill();
            });
        });
    }

    function resetTranslationUI() {
        els.transTextarea.value = "";
        els.confirmText.textContent = "";
        els.audioDuration.textContent = "";
        currentHistoryId = null;
        currentDuration = 0;
        hideError();
        els.sentPlace.style.display    = "block";
        els.transEditor.style.display  = "none";
        els.transConfirm.style.display = "none";
        els.audioSection.style.display = "none";
    }

    function clearResults() {
        currentTokens = [];
        currentHistoryId = null;
        resetTranslationUI();
        els.tokenList.innerHTML = '<span style="color:#484f58;font-size:13px;">等待识别...</span>';
        if (els.tokenCount) els.tokenCount.textContent = "";
        if (els.guessDisplay) els.guessDisplay.innerHTML = '<span class="guess-placeholder">等待手势...</span>';
        lastDetection = [];
        setFlowState(capturing ? STATE.CAPTURING : STATE.IDLE);
        if (canvasCtx && els.detectionCanvas) {
            canvasCtx.clearRect(0, 0, els.detectionCanvas.width, els.detectionCanvas.height);
        }
        send("stop");
        send("start_capture");
        log("已清除当前结果");
    }

    // ==============================
    // 历史记录
    // ==============================

    function loadHistory() {
        fetch("/api/history").then(function (r) { return r.json(); }).then(function (records) {
            renderHistory(records || []);
        }).catch(function (e) {
            log("加载历史记录失败: " + e.message, "error");
        });
    }

    function renderHistory(records) {
        if (!records || !records.length) {
            els.historyList.innerHTML = '<div class="history-empty">暂无记录</div>';
            return;
        }
        els.historyList.innerHTML = "";
        records.forEach(function (rec) {
            var item = document.createElement("div");
            item.className = "history-item";
            var body = document.createElement("div");
            body.className = "history-item-body";
            var time = document.createElement("div");
            time.className = "history-time";
            time.textContent = rec.create_time;
            var text = document.createElement("div");
            text.className = "history-text";
            text.textContent = rec.translated_text;
            body.appendChild(time);
            body.appendChild(text);
            var actions = document.createElement("div");
            actions.className = "history-actions";

            if (rec.status === "completed" && rec.audio_path) {
                var playBtn = document.createElement("button");
                playBtn.className = "btn-history-play";
                playBtn.textContent = "▶ 播放";
                playBtn.addEventListener("click", function () {
                    var a = new Audio("/" + rec.audio_path.replace(/\\/g, "/"));
                    a.play().catch(function (e) { log("历史音频播放失败: " + e.message, "error"); });
                });
                actions.appendChild(playBtn);
            }

            var delBtn = document.createElement("button");
            delBtn.className = "btn-history-delete";
            delBtn.textContent = "✕ 删除";
            delBtn.addEventListener("click", function () {
                fetch("/api/history/" + rec.id, { method: "DELETE" }).then(function (r) { return r.json(); }).then(function () {
                    log("已删除历史记录 #" + rec.id);
                    loadHistory();
                }).catch(function (e) { log("删除失败: " + e.message, "error"); });
            });
            actions.appendChild(delBtn);
            item.appendChild(body);
            item.appendChild(actions);
            els.historyList.appendChild(item);
        });
    }

    function log(message, level) {
        var div = document.createElement("div");
        div.className = "log-entry";
        var time = new Date().toLocaleTimeString();
        div.textContent = "[" + time + "] " + message;
        els.logArea.appendChild(div);
        els.logArea.scrollTop = els.logArea.scrollHeight;
    }

    // ==============================
    // 事件绑定
    // ==============================

    els.btnCapture.addEventListener("click", startCapture);
    els.btnStop.addEventListener("click", stopCapture);
    els.btnClear.addEventListener("click", clearResults);
    els.btnConfirm.addEventListener("click", confirmTranslate);
    els.btnGenerate.addEventListener("click", generateAudio);
    els.btnPlay.addEventListener("click", playAudio);
    els.btnRefreshHist.addEventListener("click", loadHistory);

    // ==============================
    // 启动
    // ==============================

    initCanvas();
    log("系统启动中...");
    updateStatus("idle", "连接中...");
    setFlowState(STATE.IDLE);
    connect();
    loadHistory();
})();
