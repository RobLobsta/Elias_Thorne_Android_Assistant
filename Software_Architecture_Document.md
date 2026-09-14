## Software Architecture Document (SAD)
Project Name: Elias Thorne On-Device Agentic PWA
Target Platform: Android (Trusted Web Activity via Bubblewrap)
Core Stack: ONNX Runtime Web (WebGPU), Orama Vector DB, Web Audio API, Web Workers
## 1. System Overview & Agentic Core
Elias Thorne is an autonomous, on-device agent running a BitNet 2B4T (Ternary quantized) model via WebGPU. The architecture separates heavy local neural processing, real-time audio constraints, and volatile JIT tool execution into sandboxed threads.
Elias is uniquely designed for recursive self-improvement. When encountering processing errors, optimizing its own tools, or structuring learned facts, Elias triggers a self-correction loop. To maintain user agency, this loop must broadcast structured progress telemetry to the UI thread before executing system updates.

                      +---------------------------------------+

                      |        UI Thread (Main View)          |
                      |  - Audio State Visualizer / Canvas   |
                      |  - Web Speech API (STT / TTS Engine)  |
                      |  - Telemetry Box (Self-Improve Logs)  |
                      +-------------------+-------------------+
                                          |
                                   PostMessage API
            +-----------------------------+-----------------------------+

            |                                                           |
+-----------v-------------+                               +-------------v------------+

|   Background Worker A   |                               |    Background Worker B   |
|   (ONNX Runtime Web)    |                               | (JIT Tool Execution)      |
|                         |                               |                           |
| - BitNet 2B4T Inference |                               | - Isolated Sandbox Core   |
| - System Persona Block  |----- Request dynamic tool --->| - Dynamic ESM Fetcher     |
| - Orama Vector / Memory |<---- Tool Execution / Logs ---| - Error Linters & Specs   |
+-------------------------+                               +--------------------------+

------------------------------
## 2. Core Technical Specifications## 2.1 Identity, Wake-Word, & Interruption

* System Persona: Explicitly locks identity to Elias Thorne. Valid wake anchors are "Hey Elias", "Elias", or "Thorne!".
* Acoustic Echo Cancellation (AEC): The PWA enforces low-level Android hardware constraints to subtract system speaker output from mic input:

{ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: false } }

* Interruptible State Machine: When Elias is in the [Speaking] state, a lightweight, non-LLM streaming event listener checks microphone arrays exclusively for the string token "Stop" or "Wait". If triggered, it instantly executes window.speechSynthesis.cancel() and rolls back to [Listening].

## 2.2 Inference & Storage Engine

* Execution Provider: ONNX Runtime Web explicitly targeting the webgpu backend.
* Persistent Web Memory: Uses Orama DB for local full-text and vector RAG. To prevent storage eviction by Android cache cleanup, the wrapper layer implements a persistent storage lock via navigator.storage.persist(). Memory arrays auto-save to IndexedDB hooks during idle states.

## 2.3 Just-In-Time (JIT) Tooling & Self-Improvement Loops

* Agentic Creation Schema: Elias outputs structured system actions using a explicit schema layout:

{
  "action": "create_tool" | "invoke_tool" | "self_improve",
  "toolName": "string",
  "dependencies": ["https://esm.sh"],
  "code": "string",
  "telemetryMessage": "User-facing summary of what Elias is optimizing or building"
}

* Sandboxed Execution Environment: Worker B handles runtime execution via a scoped AsyncFunction constructor wrapper. It blocks access to global window variables, exposing only a strictly throttled network fetch controller and secure local database interfaces.
* Self-Improvement Telemetry Loop: Before a self-correction pass or tool configuration update commits to the persistent registry, Worker A fires a postMessage event to the UI thread reading { type: "TELEMETRY_LOG", message: data.telemetryMessage }. This ensures the user is kept fully in the loop during autonomous updates.

------------------------------
## 3. Directory & File Blueprint for Claude Code
This is the recommended structural scaffold for your workspace directory.

elias-thorne-pwa/
├── android/                  # Bubblewrap TWA native configuration files
│   └── twa-manifest.json     # Android App Package configurations
├── public/
│   ├── index.html            # Core HTML5 visualizer wrapper
│   ├── manifest.json         # PWA capability descriptors
│   └── models/               # BitNet 2B4T .onnx weights (Cache-managed)
├── src/
│   ├── main.js               # UI Orchestrator, Audio State Machine & TTS bindings
│   ├── styles.css            # Lightweight minimalist visual display
│   ├── workers/
│   │   ├── inference.worker.js  # Worker A: ONNX WebGPU instance & Orama DB Layer
│   │   └── sandbox.worker.js    # Worker B: JIT runtime engine & ESM dynamic import hooks
│   └── utils/
│       ├── audio.js          # Hardware mic constraint setups & echo managers
│       └── storage.js        # IndexedDB state resiliency service
└── package.json              # Dependency manifests (Orama, ONNX Runtime Web)
