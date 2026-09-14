/**
 * UI thread — orchestrator, audio state machine, TTS bindings (SAD §1, §2.1).
 *
 * This thread owns three things and delegates everything else: the state
 * machine, the speech I/O, and the lifecycles of the two workers. It never
 * blocks on inference, so the visualiser stays at frame rate while BitNet is
 * mid-token.
 *
 * On the single recognition instance: Android's Web Speech implementation backs
 * every SpeechRecognition with one shared speech service, and a second
 * concurrent instance either fails to start or steals the first one's audio. So
 * rather than running a separate recogniser for barge-in, one continuous
 * recogniser feeds a router that behaves differently per state. While
 * [Speaking], that router is the interruption gate described in SAD §2.1: it
 * looks at interim results only, matches nothing but "stop" and "wait", and
 * never consults the model.
 */

import { AudioFrontEnd, microphonePermissionState } from './utils/audio.js';
import { AGENT_NAME, INTERRUPT_TOKENS, isInterruptToken, matchWakeAnchor } from './utils/persona.js';
import { MESSAGE } from './utils/schema.js';
import { formatBytes, requestPersistentLock } from './utils/storage.js';

/** The audio state machine. */
const STATE = Object.freeze({
  BOOT: 'boot',
  IDLE: 'idle',
  LISTENING: 'listening',
  CAPTURING: 'capturing',
  THINKING: 'thinking',
  SPEAKING: 'speaking',
  ERROR: 'error',
});

const STATE_COPY = {
  [STATE.BOOT]: 'Waking up',
  [STATE.IDLE]: 'Asleep — tap to arm',
  [STATE.LISTENING]: 'Listening for "Hey Elias"',
  [STATE.CAPTURING]: 'Listening',
  [STATE.THINKING]: 'Thinking',
  [STATE.SPEAKING]: 'Speaking — say "stop" to interrupt',
  [STATE.ERROR]: 'Something went wrong',
};

/** Trailing silence that ends a captured utterance when STT gives no final. */
const CAPTURE_TIMEOUT_MS = 6000;
/** Telemetry entries retained in the DOM. */
const MAX_TELEMETRY_ROWS = 200;

const dom = {};
const app = {
  state: STATE.BOOT,
  audio: new AudioFrontEnd(),
  recognition: null,
  recognitionRunning: false,
  wantRecognition: false,
  restartDelay: 250,
  captureBuffer: '',
  captureTimer: 0,
  inference: null,
  sandbox: null,
  speaking: null,
  utteranceQueue: [],
  voice: null,
  streamBuffer: '',
  backend: 'starting',
  muted: false,
  synthesisFailed: false,
};

/* ------------------------------------------------------------------- boot */

function cacheDom() {
  const ids = [
    'visualizer',
    'state-label',
    'state-detail',
    'backend-pill',
    'storage-pill',
    'memory-pill',
    'transcript',
    'telemetry',
    'tool-list',
    'composer',
    'composer-input',
    'arm-button',
    'mute-button',
    'stop-button',
    'permission-notice',
  ];
  for (const id of ids) dom[camel(id)] = document.getElementById(id);
}

function camel(id) {
  return id.replace(/-([a-z])/g, (_, ch) => ch.toUpperCase());
}

async function start() {
  cacheDom();
  bindControls();
  startVisualizer();
  spawnWorkers();

  const lock = await requestPersistentLock();
  dom.storagePill.textContent = lock.persisted
    ? `storage locked · ${formatBytes(lock.usage)}`
    : 'storage evictable';
  dom.storagePill.dataset.warn = String(!lock.persisted);

  const permission = await microphonePermissionState();
  if (permission === 'denied') {
    showPermissionNotice(
      'Microphone access is blocked. Enable it in the app settings to talk to Elias; typing still works.',
    );
  }

  if (!('speechSynthesis' in window)) {
    showPermissionNotice('This device has no speech synthesiser, so Elias will reply in text only.');
  }
  loadVoices();

  setState(STATE.IDLE);
  registerServiceWorker();
}

function bindControls() {
  dom.armButton.addEventListener('click', () => {
    if (app.state === STATE.IDLE) void arm();
    else void disarm();
  });

  dom.muteButton.addEventListener('click', () => {
    app.muted = !app.muted;
    app.audio.setMuted(app.muted);
    dom.muteButton.dataset.active = String(app.muted);
    dom.muteButton.textContent = app.muted ? 'Unmute mic' : 'Mute mic';
  });

  dom.stopButton.addEventListener('click', () => interrupt('button'));

  dom.composer.addEventListener('submit', (event) => {
    event.preventDefault();
    const text = dom.composerInput.value.trim();
    if (!text) return;
    dom.composerInput.value = '';
    submitTurn(text, 'typed');
  });

  // Space bar is the hardware-free barge-in for desktop testing.
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && (app.state === STATE.SPEAKING || app.state === STATE.THINKING)) {
      interrupt('escape');
    }
  });
}

/* ---------------------------------------------------------------- workers */

function spawnWorkers() {
  app.inference = new Worker(new URL('./workers/inference.worker.js', import.meta.url), {
    type: 'module',
    name: 'elias-inference',
  });
  app.inference.addEventListener('message', (event) => onInferenceMessage(event.data ?? {}));
  app.inference.addEventListener('error', (event) => {
    pushTelemetry(`Inference worker error: ${event.message}`, 'error');
    setState(STATE.ERROR, event.message);
  });

  spawnSandbox();
  app.inference.postMessage({ type: MESSAGE.INIT });
}

/**
 * Worker B is created here, not by Worker A, because only the creating thread
 * can terminate a worker — and a tool that overruns its budget cannot be
 * stopped any other way.
 */
function spawnSandbox() {
  app.sandbox?.terminate();
  app.sandbox = new Worker(new URL('./workers/sandbox.worker.js', import.meta.url), {
    type: 'module',
    name: 'elias-sandbox',
  });
  app.sandbox.addEventListener('error', (event) => {
    pushTelemetry(`Sandbox worker error: ${event.message}`, 'error');
  });

  const channel = new MessageChannel();
  app.inference.postMessage({ type: MESSAGE.BIND_SANDBOX, port: channel.port1 }, [channel.port1]);
  app.sandbox.postMessage({ type: MESSAGE.BIND_INFERENCE, port: channel.port2 }, [channel.port2]);
}

function onInferenceMessage(message) {
  switch (message.type) {
    case MESSAGE.STATUS:
      onStatus(message);
      return;

    case MESSAGE.TOKEN:
      app.streamBuffer += message.text;
      updateStreamingBubble(app.streamBuffer);
      return;

    case MESSAGE.REPLY:
      app.streamBuffer = '';
      finaliseStreamingBubble(message.text);
      if (message.text && !message.aborted) speak(message.text);
      else setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
      return;

    case MESSAGE.TELEMETRY_LOG:
      pushTelemetry(message.message, message.channel ?? 'self-improve', message.at);
      return;

    case MESSAGE.TOOLS:
      renderTools(message.tools ?? []);
      return;

    case MESSAGE.MEMORY_RESULT:
      if (message.reason === 'recall' && message.hits?.length) {
        pushTelemetry(
          `Recalled ${message.hits.length} memor${message.hits.length === 1 ? 'y' : 'ies'}: ${message.hits[0].text.slice(0, 80)}…`,
          'memory',
        );
      }
      return;

    case MESSAGE.SANDBOX_RESET:
      pushTelemetry(`Respawning the sandbox — ${message.reason}`, 'error');
      spawnSandbox();
      return;

    case MESSAGE.ERROR:
      pushTelemetry(`${message.context ?? 'error'}: ${message.message}`, 'error');
      if (app.state === STATE.THINKING) setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
      return;

    default:
  }
}

function onStatus(message) {
  const { stage, detail } = message;

  if (stage === 'storage' && detail && typeof detail === 'object') {
    dom.storagePill.textContent = detail.persisted
      ? `storage locked · ${formatBytes(detail.usage)}`
      : 'storage evictable';
    dom.storagePill.dataset.warn = String(!detail.persisted);
    return;
  }

  if (stage === 'online' && detail && typeof detail === 'object') {
    app.backend = detail.backend;
    dom.backendPill.textContent = backendLabel(detail.backend);
    dom.backendPill.dataset.warn = String(detail.backend === 'heuristic');
    dom.memoryPill.textContent = `${detail.memories} memories`;
    pushTelemetry(
      `Online on ${backendLabel(detail.backend)} with ${detail.memories} memories and ${detail.tools.length} tools.`,
      'system',
    );
    // Boot progress lines race with setState(IDLE); clear the last one so the
    // caption does not read "starting inference engine" while sitting idle.
    if (app.state === STATE.IDLE) dom.stateDetail.textContent = '';
    return;
  }

  if (stage === 'ready' && message.degraded) {
    pushTelemetry(`Weights unavailable — ${message.reason}`, 'error');
    return;
  }

  if (stage === 'memory' && typeof detail === 'string') {
    dom.memoryPill.textContent = detail;
    return;
  }

  if (stage === 'thinking') {
    setState(STATE.THINKING, typeof detail === 'string' ? detail : undefined);
    return;
  }

  if (typeof detail === 'string') dom.stateDetail.textContent = detail;
}

function backendLabel(backend) {
  if (backend === 'webgpu') return 'BitNet · WebGPU';
  if (backend === 'wasm') return 'BitNet · WASM';
  if (backend === 'heuristic') return 'fallback reasoner';
  return backend ?? 'starting';
}

/* ------------------------------------------------------------ state machine */

function setState(next, detail) {
  if (app.state === next && detail === undefined) return;
  app.state = next;
  document.body.dataset.state = next;
  dom.stateLabel.textContent = STATE_COPY[next] ?? next;
  dom.stateDetail.textContent = detail ?? '';
  dom.armButton.textContent = next === STATE.IDLE ? 'Arm wake word' : 'Stand down';
  dom.stopButton.disabled = next !== STATE.SPEAKING && next !== STATE.THINKING;

  // Residual echo must not re-trigger the VAD while the speaker is live.
  app.audio.setEchoGuard(next === STATE.SPEAKING);
}

async function arm() {
  try {
    setState(STATE.LISTENING, 'acquiring microphone');
    const settings = await app.audio.start();
    hidePermissionNotice();

    pushTelemetry(
      `Microphone armed — echo cancellation ${settings.echoCancellation ? 'on' : 'unavailable'}, ` +
        `noise suppression ${settings.noiseSuppression ? 'on' : 'off'}, ` +
        `auto gain ${settings.autoGainControl ? 'on (unexpected)' : 'off'}.`,
      'system',
    );

    app.wantRecognition = true;
    startRecognition();
    setState(STATE.LISTENING);
  } catch (error) {
    setState(STATE.IDLE);
    showPermissionNotice(
      error?.name === 'NotAllowedError'
        ? 'Microphone permission was refused. Elias can still be typed to.'
        : `Microphone unavailable: ${error.message}`,
    );
  }
}

async function disarm() {
  app.wantRecognition = false;
  stopRecognition();
  window.speechSynthesis?.cancel();
  await app.audio.stop();
  setState(STATE.IDLE);
}

/* --------------------------------------------------- speech recognition */

function recognitionCtor() {
  return window.SpeechRecognition ?? window.webkitSpeechRecognition ?? null;
}

function startRecognition() {
  const Ctor = recognitionCtor();
  if (!Ctor) {
    showPermissionNotice('This browser has no Web Speech recognition. Type to Elias instead.');
    return;
  }
  if (app.recognitionRunning) return;

  const recognition = new Ctor();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = navigator.language || 'en-US';
  recognition.maxAlternatives = 1;

  recognition.onstart = () => {
    app.recognitionRunning = true;
    app.restartDelay = 250;
  };

  recognition.onresult = (event) => routeRecognition(event);

  recognition.onerror = (event) => {
    if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
      app.wantRecognition = false;
      showPermissionNotice('Speech recognition was blocked by the system.');
      return;
    }
    if (event.error !== 'no-speech' && event.error !== 'aborted') {
      pushTelemetry(`Recognition error: ${event.error}`, 'error');
    }
  };

  // Android ends recognition sessions on its own schedule; a supervisor with
  // backoff keeps the wake word armed without hot-looping if the service is
  // genuinely unavailable.
  recognition.onend = () => {
    app.recognitionRunning = false;
    if (!app.wantRecognition) return;
    setTimeout(() => {
      if (app.wantRecognition) startRecognition();
    }, app.restartDelay);
    app.restartDelay = Math.min(app.restartDelay * 2, 4000);
  };

  app.recognition = recognition;
  try {
    recognition.start();
  } catch {
    // start() throws if a previous session has not fully released the service.
    app.recognitionRunning = false;
  }
}

function stopRecognition() {
  app.wantRecognition = false;
  try {
    app.recognition?.stop();
  } catch {
    /* already stopped */
  }
  app.recognitionRunning = false;
}

/**
 * Route one recognition event according to the current state.
 *
 * While [Speaking] this is the interruption gate from SAD §2.1: interim results
 * only, matched against nothing but the interrupt tokens.
 */
function routeRecognition(event) {
  let interim = '';
  let final = '';
  for (let i = event.resultIndex; i < event.results.length; i += 1) {
    const result = event.results[i];
    if (result.isFinal) final += result[0].transcript;
    else interim += result[0].transcript;
  }

  if (app.state === STATE.SPEAKING || app.state === STATE.THINKING) {
    if (isInterruptToken(interim) || isInterruptToken(final)) interrupt('voice');
    return;
  }

  if (app.state === STATE.LISTENING) {
    const heard = `${final} ${interim}`.trim();
    const wake = matchWakeAnchor(heard);
    if (!wake) return;

    pushTelemetry(`Wake anchor "${wake.anchor}" matched.`, 'system');
    app.captureBuffer = wake.remainder;
    setState(STATE.CAPTURING, wake.remainder || 'go ahead');
    armCaptureTimeout();

    // The wake anchor and the command often arrive in one breath.
    if (final && wake.remainder) commitCapture();
    return;
  }

  if (app.state === STATE.CAPTURING) {
    if (interim) {
      dom.stateDetail.textContent = `${app.captureBuffer} ${interim}`.trim();
      armCaptureTimeout();
    }
    if (final) {
      app.captureBuffer = `${app.captureBuffer} ${final}`.trim();
      commitCapture();
    }
  }
}

function armCaptureTimeout() {
  clearTimeout(app.captureTimer);
  app.captureTimer = setTimeout(() => {
    if (app.state === STATE.CAPTURING) commitCapture();
  }, CAPTURE_TIMEOUT_MS);
}

function commitCapture() {
  clearTimeout(app.captureTimer);
  const text = app.captureBuffer.trim();
  app.captureBuffer = '';
  if (!text) {
    setState(STATE.LISTENING);
    return;
  }
  submitTurn(text, 'voice');
}

function submitTurn(text, origin) {
  appendBubble('user', text, origin);
  app.streamBuffer = '';
  setState(STATE.THINKING);
  app.inference.postMessage({ type: MESSAGE.PROMPT, text });
}

/**
 * Interrupt whatever Elias is doing. SAD §2.1: cancel synthesis instantly and
 * roll back to [Listening].
 */
function interrupt(source) {
  if (app.state !== STATE.SPEAKING && app.state !== STATE.THINKING) return;

  window.speechSynthesis?.cancel();
  app.utteranceQueue.length = 0;
  app.speaking = null;
  app.inference.postMessage({ type: MESSAGE.CANCEL });

  pushTelemetry(`Interrupted by ${source}.`, 'system');
  setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
}

/* ------------------------------------------------------------------- TTS */

function loadVoices() {
  const pick = () => {
    const voices = window.speechSynthesis?.getVoices() ?? [];
    if (voices.length === 0) return;
    const language = navigator.language || 'en-US';
    app.voice =
      voices.find((v) => v.lang === language && v.localService) ??
      voices.find((v) => v.lang.startsWith(language.slice(0, 2)) && v.localService) ??
      voices.find((v) => v.lang.startsWith(language.slice(0, 2))) ??
      voices[0];
  };
  pick();
  if (window.speechSynthesis) window.speechSynthesis.onvoiceschanged = pick;
}

/**
 * Speak a reply, one sentence per utterance.
 *
 * Chunking matters for interruption: cancel() takes effect between utterances
 * on some Android builds, so short utterances bound the worst-case latency
 * between "stop" and silence.
 */
function speak(text) {
  if (!window.speechSynthesis) {
    setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
    return;
  }

  const sentences = text
    .replace(/```[\s\S]*?```/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
  if (sentences.length === 0) {
    setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
    return;
  }

  app.utteranceQueue = sentences;
  setState(STATE.SPEAKING);
  speakNext();
}

function speakNext() {
  const sentence = app.utteranceQueue.shift();
  if (!sentence) {
    app.speaking = null;
    setState(app.wantRecognition ? STATE.LISTENING : STATE.IDLE);
    return;
  }

  const utterance = new SpeechSynthesisUtterance(sentence);
  if (app.voice) utterance.voice = app.voice;
  utterance.rate = 1.02;
  utterance.pitch = 0.95;

  utterance.onend = () => {
    if (app.state === STATE.SPEAKING) speakNext();
  };
  utterance.onerror = (event) => {
    if (event.error === 'interrupted' || event.error === 'canceled') return;
    // A device with no installed voice fails every utterance. Say so once
    // rather than filling the telemetry box with the same line per sentence.
    if (!app.synthesisFailed) {
      app.synthesisFailed = true;
      pushTelemetry(
        `Speech synthesis unavailable (${event.error}) — replying in text only.`,
        'error',
      );
    }
    if (app.state === STATE.SPEAKING) speakNext();
  };

  app.speaking = utterance;
  window.speechSynthesis.speak(utterance);
}

/* ------------------------------------------------------------- transcript */

let streamingBubble = null;

function appendBubble(role, text, origin) {
  const bubble = document.createElement('article');
  bubble.className = `bubble bubble--${role}`;
  if (origin) bubble.dataset.origin = origin;

  const body = document.createElement('p');
  body.textContent = text;
  bubble.append(body);

  dom.transcript.append(bubble);
  dom.transcript.scrollTop = dom.transcript.scrollHeight;
  return bubble;
}

function updateStreamingBubble(text) {
  if (!streamingBubble) {
    streamingBubble = appendBubble('agent', '');
    streamingBubble.dataset.streaming = 'true';
  }
  streamingBubble.querySelector('p').textContent = text;
  dom.transcript.scrollTop = dom.transcript.scrollHeight;
}

function finaliseStreamingBubble(text) {
  if (!text) {
    streamingBubble?.remove();
    streamingBubble = null;
    return;
  }
  if (!streamingBubble) streamingBubble = appendBubble('agent', '');
  streamingBubble.querySelector('p').textContent = text;
  delete streamingBubble.dataset.streaming;
  streamingBubble = null;
  dom.transcript.scrollTop = dom.transcript.scrollHeight;
}

/* -------------------------------------------------------------- telemetry */

function pushTelemetry(message, channel = 'system', at = Date.now()) {
  const row = document.createElement('li');
  row.className = 'telemetry__row';
  row.dataset.channel = channel;

  const time = document.createElement('time');
  time.textContent = new Date(at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  const body = document.createElement('span');
  body.textContent = message;

  row.append(time, body);
  dom.telemetry.append(row);

  while (dom.telemetry.childElementCount > MAX_TELEMETRY_ROWS) {
    dom.telemetry.firstElementChild.remove();
  }
  dom.telemetry.scrollTop = dom.telemetry.scrollHeight;
}

function renderTools(tools) {
  dom.toolList.replaceChildren();
  if (tools.length === 0) {
    const empty = document.createElement('li');
    empty.className = 'tool tool--empty';
    empty.textContent = 'No tools yet — ask Elias to build one.';
    dom.toolList.append(empty);
    return;
  }

  for (const tool of tools) {
    const item = document.createElement('li');
    item.className = 'tool';

    const name = document.createElement('strong');
    name.textContent = tool.toolName;

    const meta = document.createElement('span');
    meta.textContent = `rev ${tool.revision} · ${tool.codeBytes} B${
      tool.dependencies?.length ? ` · ${tool.dependencies.length} dep` : ''
    }`;

    const summary = document.createElement('p');
    summary.textContent = tool.telemetryMessage ?? '';

    item.append(name, meta, summary);
    dom.toolList.append(item);
  }
}

function showPermissionNotice(text) {
  dom.permissionNotice.textContent = text;
  dom.permissionNotice.hidden = false;
}

function hidePermissionNotice() {
  dom.permissionNotice.hidden = true;
}

/* ------------------------------------------------------------- visualiser */

function startVisualizer() {
  const canvas = dom.visualizer;
  const ctx = canvas.getContext('2d');
  let width = 0;
  let height = 0;
  let phase = 0;

  const palette = {
    [STATE.BOOT]: ['#3b3f52', '#5a6076'],
    [STATE.IDLE]: ['#2f3446', '#464c63'],
    [STATE.LISTENING]: ['#1f6f8b', '#37b3c9'],
    [STATE.CAPTURING]: ['#2d8f6f', '#4fd6a2'],
    [STATE.THINKING]: ['#7a5cc4', '#b38ef5'],
    [STATE.SPEAKING]: ['#c07b2f', '#f5b451'],
    [STATE.ERROR]: ['#8b2f3b', '#d95c6a'],
  };

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    width = rect.width;
    height = rect.height;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  new ResizeObserver(resize).observe(canvas);

  const draw = () => {
    requestAnimationFrame(draw);
    if (width === 0 || height === 0) return;

    const [dark, bright] = palette[app.state] ?? palette[STATE.IDLE];
    const centreX = width / 2;
    const centreY = height / 2;
    const baseRadius = Math.min(width, height) * 0.22;

    ctx.clearRect(0, 0, width, height);
    phase += app.state === STATE.THINKING ? 0.045 : 0.012;

    const spectrum = app.audio.spectrum();
    // While speaking there is no useful mic signal (the AEC has removed it), so
    // the ring is driven by a synthetic envelope instead.
    const energy =
      app.state === STATE.SPEAKING
        ? 0.35 + 0.25 * Math.abs(Math.sin(phase * 3.1))
        : Math.min(app.audio.level * 6, 1);

    const gradient = ctx.createRadialGradient(centreX, centreY, baseRadius * 0.2, centreX, centreY, baseRadius * 2.2);
    gradient.addColorStop(0, bright);
    gradient.addColorStop(1, 'transparent');
    ctx.globalAlpha = 0.18 + energy * 0.28;
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    ctx.globalAlpha = 1;

    const points = 128;
    ctx.beginPath();
    for (let i = 0; i <= points; i += 1) {
      const angle = (i / points) * Math.PI * 2;
      const bin = spectrum ? spectrum[Math.floor((i / points) * spectrum.length)] / 255 : 0;
      const wobble = Math.sin(angle * 3 + phase * 2) * 0.04 + Math.sin(angle * 7 - phase) * 0.02;
      const radius = baseRadius * (1 + wobble + energy * 0.35 + bin * 0.45);
      const x = centreX + Math.cos(angle) * radius;
      const y = centreY + Math.sin(angle) * radius;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = bright;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = `${dark}55`;
    ctx.fill();

    ctx.beginPath();
    ctx.arc(centreX, centreY, baseRadius * (0.45 + energy * 0.12), 0, Math.PI * 2);
    ctx.fillStyle = bright;
    ctx.globalAlpha = 0.55 + energy * 0.35;
    ctx.fill();
    ctx.globalAlpha = 1;
  };

  requestAnimationFrame(draw);
}

/* -------------------------------------------------------- service worker */

function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((error) => {
      pushTelemetry(`Service worker registration failed: ${error.message}`, 'error');
    });
  });
}

/* ------------------------------------------------------------------ init */

document.title = `${AGENT_NAME} — on-device agent`;
document.addEventListener('DOMContentLoaded', () => {
  void start();
  pushTelemetry(
    `Interruption tokens armed: ${INTERRUPT_TOKENS.map((t) => `"${t}"`).join(', ')}.`,
    'system',
  );
});
