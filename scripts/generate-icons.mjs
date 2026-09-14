/**
 * Generate the PWA / TWA icon set.
 *
 * Bubblewrap needs a real 512px PNG to build a launcher icon, and the manifest
 * needs 192/512 plus a maskable variant. Rather than committing binaries nobody
 * can diff, the icons are drawn here from the same palette as styles.css and
 * written with a minimal PNG encoder (node:zlib does the compression, so there
 * is no image dependency).
 *
 * Run: npm run icons
 */

import { deflateSync } from 'node:zlib';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'icons');

const GROUND = [0x0b, 0x0d, 0x14];
const RING = [0x4f, 0xd6, 0xa2];
const CORE = [0x37, 0xb3, 0xc9];
const HALO = [0x7a, 0x5c, 0xc4];

/**
 * Draw the Elias mark: a listening ring around a core, matching the visualiser.
 *
 * @param {number} size edge length in pixels
 * @param {number} safeRatio fraction of the edge the artwork may occupy — the
 *   maskable variant keeps everything inside Android's 80% safe zone so an
 *   aggressive launcher mask cannot clip the ring.
 */
function drawIcon(size, safeRatio) {
  const pixels = new Uint8Array(size * size * 4);
  const centre = (size - 1) / 2;
  const artwork = (size * safeRatio) / 2;

  const ringRadius = artwork * 0.78;
  const ringWidth = artwork * 0.13;
  const coreRadius = artwork * 0.32;
  const haloRadius = artwork * 1.02;

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const dx = x - centre;
      const dy = y - centre;
      const distance = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);

      let colour = GROUND;
      let alpha = 1;

      // Soft halo behind everything.
      const haloFalloff = clamp01(1 - distance / haloRadius);
      colour = mix(GROUND, HALO, haloFalloff * 0.35);

      // The ring is modulated the way the visualiser modulates its waveform, so
      // the icon reads as a still frame of the running app.
      const wobble = 1 + Math.sin(angle * 3) * 0.045 + Math.sin(angle * 7) * 0.02;
      const ringDistance = Math.abs(distance - ringRadius * wobble);
      if (ringDistance < ringWidth) {
        colour = mix(colour, RING, smoothstep(1 - ringDistance / ringWidth));
      }

      // A gap at the top-right, so the mark is not a plain circle at small sizes.
      const gap = Math.abs(normaliseAngle(angle + Math.PI / 4));
      if (gap < 0.28 && ringDistance < ringWidth) {
        colour = mix(colour, GROUND, smoothstep(1 - gap / 0.28));
      }

      if (distance < coreRadius) {
        colour = mix(colour, CORE, smoothstep(1 - distance / coreRadius) * 0.95 + 0.05);
      }

      const offset = (y * size + x) * 4;
      pixels[offset] = colour[0];
      pixels[offset + 1] = colour[1];
      pixels[offset + 2] = colour[2];
      pixels[offset + 3] = Math.round(alpha * 255);
    }
  }

  return pixels;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}

function smoothstep(t) {
  const x = clamp01(t);
  return x * x * (3 - 2 * x);
}

function normaliseAngle(angle) {
  let a = angle;
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
}

function mix(a, b, t) {
  const k = clamp01(t);
  return [
    Math.round(a[0] + (b[0] - a[0]) * k),
    Math.round(a[1] + (b[1] - a[1]) * k),
    Math.round(a[2] + (b[2] - a[2]) * k),
  ];
}

/* ------------------------------------------------------------ PNG encoder */

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

function encodePng(pixels, size) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // colour type: RGBA
  ihdr[10] = 0; // deflate
  ihdr[11] = 0; // adaptive filtering
  ihdr[12] = 0; // no interlace

  // One filter byte per scanline; filter 0 (None) keeps the encoder trivial and
  // still compresses well on flat artwork.
  const stride = size * 4;
  const raw = Buffer.alloc((stride + 1) * size);
  for (let y = 0; y < size; y += 1) {
    raw[y * (stride + 1)] = 0;
    Buffer.from(pixels.buffer, y * stride, stride).copy(raw, y * (stride + 1) + 1);
  }

  return Buffer.concat([
    signature,
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 9 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/* ------------------------------------------------------------------- run */

const TARGETS = [
  { file: 'icon-192.png', size: 192, safeRatio: 0.92 },
  { file: 'icon-512.png', size: 512, safeRatio: 0.92 },
  // Android may crop a maskable icon to a circle inscribed in 80% of the edge.
  { file: 'icon-maskable-512.png', size: 512, safeRatio: 0.66 },
];

await mkdir(OUT_DIR, { recursive: true });
for (const { file, size, safeRatio } of TARGETS) {
  const png = encodePng(drawIcon(size, safeRatio), size);
  await writeFile(resolve(OUT_DIR, file), png);
  console.log(`wrote ${file} (${size}x${size}, ${png.length} bytes)`);
}
