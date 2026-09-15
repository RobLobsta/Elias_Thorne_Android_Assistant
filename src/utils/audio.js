/**
 * Hardware mic constraints and echo management (SAD §2.1).
 *
 * The PWA asks Android for hardware acoustic echo cancellation so the speaker
 * output is subtracted from the mic input at the platform layer. That is what
 * makes barge-in possible at all: without it, the microphone hears Elias's own
 * synthesised voice and "stop" is triggered by his own speech.
 *
 * Auto gain control is deliberately OFF. AGC ramps the mic gain up during the
 * silences between sentences, which both floods the VAD with room noise and
 * partially defeats the echo canceller's gain model.
 */

/** The exact constraint set the SAD mandates. */
export const MIC_CONSTRAINTS = Object.freeze({
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: false,
  },
});

/**
 * Optional vendor hints, applied via `advanced` so that a device that does not
 * understand them ignores the entry instead of failing the whole request.
 * `echoCancellationType: 'system'` asks Chrome for the Android hardware AEC
 * rather than its own WebRTC software canceller.
 */
const ADVANCED_HINTS = [
  { echoCancellationType: 'system' },
  { googEchoCancellation: true },
  { googAutoGainControl: false },
  { channelCount: 1 },
];

/** RMS above which we consider the user to be speaking. */
const SPEECH_THRESHOLD = 0.035;
/** While Elias speaks, residual echo can leak through; demand more energy. */
const ECHO_GUARD_MULTIPLIER = 2.6;
/** Milliseconds of sub-threshold audio before we call the utterance finished. */
const SILENCE_HANGOVER_MS = 900;

/**
 * Owns the microphone stream, the AudioContext graph and a lightweight VAD.
 *
 * The graph is analysis-only — the mic is never routed to the destination, so
 * there is no feedback path of our own making.
 */
export class AudioFrontEnd extends EventTarget {
  stream = null;
  context = null;
  analyser = null;
  #source = null;
  #timeData = null;
  #freqData = null;
  #raf = 0;
  #speaking = false;
  #silenceSince = 0;
  #echoGuard = false;
  #lastLevel = 0;

  get active() {
    return Boolean(this.stream);
  }

  get level() {
    return this.#lastLevel;
  }

  /** True when the platform confirmed it applied echo cancellation. */
  get echoCancellationActive() {
    const track = this.stream?.getAudioTracks?.()[0];
    if (!track?.getSettings) return false;
    return track.getSettings().echoCancellation === true;
  }

  /**
   * Acquire the microphone and start the analysis loop.
   * @returns {Promise<MediaTrackSettings>} the settings the platform actually applied
   */
  async start() {
    if (this.stream) return this.settings();

    const constraints = {
      audio: { ...MIC_CONSTRAINTS.audio, advanced: ADVANCED_HINTS },
    };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      if (error?.name === 'OverconstrainedError' || error?.name === 'NotSupportedError') {
        // Fall back to the bare SAD constraint set without the vendor hints.
        this.stream = await navigator.mediaDevices.getUserMedia(MIC_CONSTRAINTS);
      } else {
        throw error;
      }
    }

    const AudioCtx = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    this.context = new AudioCtx({ latencyHint: 'interactive' });
    if (this.context.state === 'suspended') await this.context.resume();

    this.#source = this.context.createMediaStreamSource(this.stream);
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = 1024;
    this.analyser.smoothingTimeConstant = 0.82;
    this.#source.connect(this.analyser);

    this.#timeData = new Float32Array(this.analyser.fftSize);
    this.#freqData = new Uint8Array(this.analyser.frequencyBinCount);
    this.#silenceSince = performance.now();
    this.#loop();

    return this.settings();
  }

  settings() {
    const track = this.stream?.getAudioTracks?.()[0];
    return track?.getSettings ? track.getSettings() : {};
  }

  /** Raise the VAD threshold while the TTS engine is producing output. */
  setEchoGuard(active) {
    this.#echoGuard = Boolean(active);
    if (active && this.#speaking) {
      this.#speaking = false;
      this.dispatchEvent(new CustomEvent('speechend', { detail: { reason: 'echo-guard' } }));
    }
  }

  /** Frequency bins for the visualiser; safe to call before start(). */
  spectrum() {
    if (!this.analyser || !this.#freqData) return null;
    this.analyser.getByteFrequencyData(this.#freqData);
    return this.#freqData;
  }

  /** Mute the capture track without tearing down the graph. */
  setMuted(muted) {
    for (const track of this.stream?.getAudioTracks?.() ?? []) {
      track.enabled = !muted;
    }
  }

  async stop() {
    cancelAnimationFrame(this.#raf);
    this.#raf = 0;
    for (const track of this.stream?.getAudioTracks?.() ?? []) track.stop();
    this.stream = null;
    this.#source?.disconnect();
    this.#source = null;
    this.analyser = null;
    if (this.context && this.context.state !== 'closed') {
      await this.context.close().catch(() => {});
    }
    this.context = null;
    this.#speaking = false;
    this.#lastLevel = 0;
  }

  #loop = () => {
    this.#raf = requestAnimationFrame(this.#loop);
    if (!this.analyser) return;

    this.analyser.getFloatTimeDomainData(this.#timeData);
    let sum = 0;
    for (let i = 0; i < this.#timeData.length; i += 1) {
      sum += this.#timeData[i] * this.#timeData[i];
    }
    const rms = Math.sqrt(sum / this.#timeData.length);
    this.#lastLevel = rms;
    this.dispatchEvent(new CustomEvent('level', { detail: { rms } }));

    const threshold = this.#echoGuard ? SPEECH_THRESHOLD * ECHO_GUARD_MULTIPLIER : SPEECH_THRESHOLD;
    const now = performance.now();

    if (rms >= threshold) {
      this.#silenceSince = now;
      if (!this.#speaking) {
        this.#speaking = true;
        this.dispatchEvent(new CustomEvent('speechstart', { detail: { rms } }));
      }
      return;
    }

    if (this.#speaking && now - this.#silenceSince >= SILENCE_HANGOVER_MS) {
      this.#speaking = false;
      this.dispatchEvent(new CustomEvent('speechend', { detail: { reason: 'silence' } }));
    }
  };
}

/**
 * Report on the microphone permission without triggering a prompt, so the UI
 * can show an accurate state before the user taps anything.
 */
export async function microphonePermissionState() {
  if (typeof navigator === 'undefined' || !navigator.permissions?.query) return 'unknown';
  try {
    const status = await navigator.permissions.query({ name: 'microphone' });
    return status.state;
  } catch {
    return 'unknown';
  }
}
