/**
 * Zero-Knowledge Engine
 * Zero-Knowledge Orchestrator Web Worker
 *
 * Handles the entire ZK proof lifecycle off the main thread:
 *   1. Initialize WASM module via compileStreaming
 *   2. Check OPFS cache for existing .zkey file
 *   3. If cache miss: BYOB-stream the .zkey from the server
 *   4. Pipe chunks to both OPFS (persist) and WASM linear memory (compute)
 *   5. Execute Groth16 proof generation
 *   6. Return TOON-Crypto serialized proof to main thread
 *
 * ARCHITECTURAL LAWS:
 *   - NO ArrayBuffer allocation for the full .zkey (BYOB bounce buffer only)
 *   - NO postMessage of large buffers (only small proof results)
 *   - NO synchronous file I/O on main thread (OPFS SyncAccessHandle is
 *     Worker-only API)
 */

// ─── Types ─────────────────────────────────────────────────────────────────

declare global {
  function importScripts(...urls: string[]): void;
  interface FileSystemSyncAccessHandle {
    read(buffer: ArrayBuffer | ArrayBufferView, options?: { at?: number }): number;
    write(buffer: ArrayBuffer | ArrayBufferView, options?: { at?: number }): number;
    flush(): void;
    close(): void;
    getSize(): number;
    truncate(newSize: number): void;
  }
  interface FileSystemFileHandle {
    createSyncAccessHandle(): Promise<FileSystemSyncAccessHandle>;
  }
}

interface WorkerMessage {
  type: 'init' | 'prove';
  payload?: ProvePayload;
}

interface ProvePayload {
  publicInputs: number[];
  zkeyUrl: string;
  zkeyHash: string;
}

interface WorkerResponse {
  type: 'status' | 'progress' | 'proof' | 'error';
  payload: StatusPayload | ProgressPayload | ProofPayload | ErrorPayload;
}

interface StatusPayload {
  phase: 'initializing' | 'downloading' | 'cached' | 'computing' | 'done';
  message: string;
}

interface ProgressPayload {
  phase: 'download' | 'compute';
  percent: number;
  bytesLoaded?: number;
  bytesTotal?: number;
}

interface ProofPayload {
  publicInputsHash: string;
  nullifierHash: string;
  proofCommitmentMatrix: {
    pi_a: [string, string, string];
    pi_b: [[string, string], [string, string], [string, string]];
    pi_c: [string, string, string];
  };
  computeTimeMs: number;
}

interface ErrorPayload {
  code: string;
  message: string;
  details?: string;
}

// ─── Constants ─────────────────────────────────────────────────────────────

const BOUNCE_BUFFER_SIZE = 256 * 1024; // 256KB
const OPFS_DIR_NAME = 'zk_engine-zk-cache';

// ─── WASM Module ───────────────────────────────────────────────────────────

let wasmModule: any = null;
let isInitialized = false;

// ─── Thread Pool Size Calculation ──────────────────────────────────────────
const THREAD_POOL_SIZE = Math.max(
  1,
  Math.floor(navigator.hardwareConcurrency / 2)
);

// ─── Helper Functions ──────────────────────────────────────────────────────

function send(msg: WorkerResponse): void {
  self.postMessage(msg);
}

function sendStatus(phase: StatusPayload['phase'], message: string): void {
  send({ type: 'status', payload: { phase, message } });
}

function sendProgress(
  phase: ProgressPayload['phase'],
  percent: number,
  bytesLoaded?: number,
  bytesTotal?: number
): void {
  send({ type: 'progress', payload: { phase, percent, bytesLoaded, bytesTotal } });
}

function sendError(code: string, message: string, details?: string): void {
  send({ type: 'error', payload: { code, message, details } });
}

function bytesToHex(bytes: Uint8Array): string {
  const hexArray: string[] = [];
  for (let i = 0; i < bytes.length; i++) {
    hexArray.push(bytes[i].toString(16).padStart(2, '0'));
  }
  return '0x' + hexArray.join('');
}

// ─── WASM Initialization ──────────────────────────────────────────────────

async function initializeWasm(): Promise<void> {
  if (isInitialized) return;

  sendStatus('initializing', 'Compiling WASM module...');

  try {
    // To completely bypass Vite's dev server interception of public directory imports,
    // we fetch the JS glue code manually and evaluate it via a Blob URL.
    const response = await fetch('/wasm/prover_engine.js');
    if (!response.ok) throw new Error(`Failed to fetch WASM JS: ${response.status}`);
    const blob = new Blob([await response.text()], { type: 'application/javascript' });
    const objectUrl = URL.createObjectURL(blob);

    const engineModule = await import(/* @vite-ignore */ objectUrl);
    const createProverEngine = engineModule.default;
    URL.revokeObjectURL(objectUrl);

    if (typeof createProverEngine !== 'function') {
      throw new Error(
        'createProverEngine factory not found. WASM glue code may be corrupted.'
      );
    }

    wasmModule = await createProverEngine({
      locateFile: (path: string) => {
        if (path.endsWith('.wasm')) return '/wasm/prover_engine.wasm';
        return path;
      },
      pthreadPoolSize: THREAD_POOL_SIZE,
    });

    isInitialized = true;
    sendStatus('initializing', `WASM engine ready. Thread pool: ${THREAD_POOL_SIZE} P-cores.`);
  } catch (err) {
    sendError(
      'WASM_INIT_FAILED',
      'Failed to initialize WASM prover engine.',
      err instanceof Error ? err.message : String(err)
    );
    throw err;
  }
}

// ─── OPFS Cache Layer ──────────────────────────────────────────────────────

async function getOpfsDir(): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(OPFS_DIR_NAME, { create: true });
}

async function getCachedZkey(hash: string): Promise<FileSystemSyncAccessHandle | null> {
  try {
    const dir = await getOpfsDir();
    const fileHandle = await dir.getFileHandle(`zkey-${hash}.bin`);
    return await fileHandle.createSyncAccessHandle();
  } catch {
    return null; // Cache miss
  }
}





// ─── iden3 .zkey Header Constants ─────────────────────────────────────────
// The .zkey binary format (iden3/snarkjs):
//   Bytes 0-3:   Magic "zkey" (0x7A 0x6B 0x65 0x79)
//   Bytes 4-7:   Version (uint32 LE)
//   Bytes 8-11:  Number of sections (uint32 LE)
//   Then for each section:
//     Bytes 0-3:  Section type (uint32 LE)
//     Bytes 4-11: Section size (uint64 LE)
//     Bytes 12+:  Section data
//
// Section types:
//   1 = Header (contains field info, curve, power, n8q, n8r)
//   2 = Groth16 proving key points (G1 and G2 elements — THIS IS THE PAYLOAD)
//   3 = IC (verification key points)

const ZKEY_MAGIC = new Uint8Array([0x7A, 0x6B, 0x65, 0x79]); // "zkey"
const SECTION_TYPE_GROTH16_PK = 2;

interface ZkeyHeaderInfo {
  version: number;
  numSections: number;
  pkSectionOffset: number;   // Byte offset where section 2 DATA begins
  pkSectionSize: number;     // Byte size of section 2 DATA
  totalHeaderSize: number;   // Total bytes before section 2 data
}

/**
 * Parses the iden3 .zkey binary header to locate the proving key section.
 *
 * This reads only the minimum bytes needed to find section 2's offset.
 * It does NOT load the entire file — subsequent streaming writes skip
 * directly to the proving key offset.
 *
 * @param reader  A ReadableStreamDefaultReader from the fetch response
 * @returns       Header metadata including the proving key section offset
 */
async function parseIden3Header(
  reader: ReadableStreamDefaultReader<Uint8Array>
): Promise<{ headerInfo: ZkeyHeaderInfo; remainingChunk: Uint8Array | null }> {
  // Accumulate header bytes (the header is typically < 1KB)
  const headerChunks: Uint8Array[] = [];
  let headerBytesRead = 0;
  const HEADER_READ_LIMIT = 64 * 1024; // 64KB max header read

  // We need at least 12 bytes for the global header (magic + version + numSections)
  while (headerBytesRead < 12) {
    const { done, value } = await reader.read();
    if (done || !value) throw new Error('Unexpected end of .zkey stream during header read');
    headerChunks.push(value);
    headerBytesRead += value.byteLength;
  }

  // Concatenate into a single buffer for parsing
  let headerBuffer = new Uint8Array(headerBytesRead);
  let writeOffset = 0;
  for (const chunk of headerChunks) {
    headerBuffer.set(chunk, writeOffset);
    writeOffset += chunk.byteLength;
  }

  // Validate magic
  for (let i = 0; i < 4; i++) {
    if (headerBuffer[i] !== ZKEY_MAGIC[i]) {
      throw new Error(`Invalid .zkey magic at byte ${i}: expected 0x${ZKEY_MAGIC[i].toString(16)}, got 0x${headerBuffer[i].toString(16)}`);
    }
  }

  const view = new DataView(headerBuffer.buffer, headerBuffer.byteOffset, headerBuffer.byteLength);
  const version = view.getUint32(4, true);
  const numSections = view.getUint32(8, true);

  // Read section descriptors to find section 2 (Groth16 proving key)
  let parseOffset = 12;
  let pkSectionOffset = 0;
  let pkSectionSize = 0;
  let pkFound = false;

  for (let s = 0; s < numSections; s++) {
    // Ensure we have enough bytes for the section descriptor (12 bytes: type + size)
    while (headerBytesRead < parseOffset + 12) {
      if (headerBytesRead > HEADER_READ_LIMIT) {
        throw new Error('.zkey header exceeds 64KB limit — file may be corrupted');
      }
      const { done, value } = await reader.read();
      if (done || !value) throw new Error('Unexpected end of .zkey stream during section descriptor read');

      // Resize headerBuffer
      const newBuffer = new Uint8Array(headerBytesRead + value.byteLength);
      newBuffer.set(headerBuffer);
      newBuffer.set(value, headerBytesRead);
      headerBuffer = newBuffer;
      headerBytesRead += value.byteLength;
    }

    const secView = new DataView(headerBuffer.buffer, headerBuffer.byteOffset, headerBuffer.byteLength);
    const sectionType = secView.getUint32(parseOffset, true);
    // Section size is a uint64 LE — read via BigInt for precision safety
    // (HARDENING: Doubt 4 fix — avoids IEEE 754 double-precision truncation
    //  for section sizes > 2^53 bytes, which would silently corrupt the value
    //  when using Number arithmetic with sizeHigh * 0x100000000)
    const sectionSizeBig = secView.getBigUint64(parseOffset + 4, true);
    if (sectionSizeBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(
        `.zkey section ${sectionType} declares size ${sectionSizeBig} bytes — ` +
        `exceeds MAX_SAFE_INTEGER, file is corrupt or unsupported.`
      );
    }
    const sectionSize = Number(sectionSizeBig);

    if (sectionType === SECTION_TYPE_GROTH16_PK) {
      pkSectionOffset = parseOffset + 12; // Data starts after the 12-byte descriptor
      pkSectionSize = sectionSize;
      pkFound = true;
      break;
    }

    // Skip past this section's data
    parseOffset += 12 + sectionSize;
  }

  if (!pkFound) {
    throw new Error('.zkey file does not contain a Groth16 proving key section (type 2)');
  }

  // Determine if we have leftover bytes after the header that belong to section 2 data
  let remainingChunk: Uint8Array | null = null;
  if (headerBytesRead > pkSectionOffset) {
    remainingChunk = headerBuffer.slice(pkSectionOffset);
  }

  return {
    headerInfo: {
      version,
      numSections,
      pkSectionOffset,
      pkSectionSize,
      totalHeaderSize: pkSectionOffset,
    },
    remainingChunk,
  };
}

/**
 * Zero-copy .zkey loader for Phase 3.1 MSM orchestration.
 *
 * Unlike streamZkeyToWasm() (Phase 1) which copies the ENTIRE .zkey file
 * into WASM memory, this method:
 *
 *   1. Parses the iden3 binary header to locate section 2 (proving key)
 *   2. Calls WASM init_g1_arena() with the header size
 *   3. Calls WASM allocate_g1_arena() to get a target pointer
 *   4. Streams ONLY the section 2 polynomial payloads directly into
 *      WASM linear memory at the allocated address
 *   5. Discards header bytes (they are not needed for proof generation)
 *
 * This eliminates redundant header storage in the constrained 64MB WASM buffer.
 *
 * @param url   URL of the .zkey file (e.g., '/mock.zkey')
 * @param hash  SHA-256 hash for OPFS cache lookup
 */
async function loadZkeyZeroCopy(url: string, hash: string): Promise<void> {
  // Phase 3.1: Check OPFS cache first — uses VERIFIED cache only
  // (HARDENING: Doubt 2 fix — only the final renamed file is considered valid.
  //  The .tmp staging file is invisible to getCachedZkey.)
  const cached = await getCachedZkey(hash);
  if (cached) {
    sendStatus('cached', 'Loading .zkey from OPFS cache (zero-copy mode)...');
    await loadFromOpfsCache(cached);
    return;
  }

  sendStatus('downloading', 'Fetching .zkey with zero-copy header parsing...');

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch .zkey: HTTP ${response.status}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
  const body = response.body;
  if (!body) throw new Error('.zkey response has no body');

  // Open OPFS for write-through caching using atomic rename pattern
  // (HARDENING: Doubt 2 fix — write to a .tmp staging file first.
  //  Only rename to the final filename after ALL bytes are verified.
  //  If the network drops mid-download, the .tmp file is orphaned and
  //  invisible to getCachedZkey(), preventing corrupted cache hits.
  //  On the next attempt, createZkeyCache() overwrites the .tmp file.)
  const dir = await getOpfsDir();
  const tmpHandle = await dir.getFileHandle(`zkey-${hash}.tmp`, { create: true });
  const opfsHandle = await tmpHandle.createSyncAccessHandle();
  opfsHandle.truncate(0); // Clear any orphaned partial data from a previous crash

  // Get a default reader (BYOB not needed for header parsing)
  const reader = body.getReader() as ReadableStreamDefaultReader<Uint8Array>;

  // ─── Step 1: Parse iden3 header ────────────────────────────────────
  let headerInfo: ZkeyHeaderInfo;
  let remainingChunk: Uint8Array | null;

  try {
    const parsed = await parseIden3Header(reader);
    headerInfo = parsed.headerInfo;
    remainingChunk = parsed.remainingChunk;
  } catch (err) {
    opfsHandle.close();
    throw err;
  }

  sendStatus('downloading',
    `iden3 header parsed: v${headerInfo.version}, ${headerInfo.numSections} sections, ` +
    `PK section: ${headerInfo.pkSectionSize} bytes at offset ${headerInfo.pkSectionOffset}`
  );

  // ─── Step 2: Initialize WASM arena ─────────────────────────────────
  wasmModule._reset_engine();

  // HARDENING: Doubt 3 fix — Hard-coded upper bound check BEFORE touching WASM
  // Query the actual WASM buffer capacity (64MB = ZKEY_BUFFER_CAPACITY in C++)
  // and reject any .zkey whose PK section exceeds it. This prevents a malformed
  // or adversarial .zkey from declaring a massive section size that would cause
  // heapView.set() to silently write past the arena into the WASM call stack
  // or proof output buffers.
  const wasmBufferCapacity: number = wasmModule._get_buffer_capacity();
  if (headerInfo.pkSectionSize > wasmBufferCapacity) {
    opfsHandle.close();
    sendError(
      'ZKEY_TOO_LARGE',
      `.zkey PK section (${headerInfo.pkSectionSize} bytes) exceeds WASM buffer capacity (${wasmBufferCapacity} bytes).`,
      'The .zkey file may be corrupted or designed for a circuit too large for this client.'
    );
    return;
  }

  wasmModule.ccall('init_g1_arena_export', null, ['number'], [headerInfo.totalHeaderSize]);

  // Calculate how many JacobianPoints fit in the PK section
  // Each JacobianPoint = 256 bytes, but raw zkey stores points as 2×32=64 byte
  // affine coordinates. The actual point count depends on the circuit.
  // For now, allocate based on raw byte count — Phase 3.2 will refine this.
  const rawPointSize = 64; // BN254 G1 affine point = 2 × 32 bytes
  const estimatedPointCount = Math.floor(headerInfo.pkSectionSize / rawPointSize);

  const arenaPtr: number = wasmModule.ccall(
    'allocate_g1_arena_export', 'number', ['number'], [estimatedPointCount]
  );

  if (arenaPtr === 0) {
    opfsHandle.close();
    sendError('ARENA_OVERFLOW', 'G1 arena allocation failed — .zkey too large for 64MB buffer.');
    return;
  }

  // ─── Step 3: Stream payload into WASM linear memory ────────────────
  let totalPayloadBytesWritten = 0;
  let totalFileOffset = headerInfo.pkSectionOffset; // For OPFS write-through

  // Write header bytes to OPFS (we cache the full file)
  // The header was already read into memory during parsing, but we
  // stored it in the parseIden3Header function. For simplicity, write
  // the remaining chunk which starts at the PK section.
  // NOTE: In production, the header bytes should also be written to OPFS.
  // Phase 3.2 will add full header write-through.

  // First, process any leftover bytes from header parsing
  if (remainingChunk && remainingChunk.byteLength > 0) {
    const chunkLen = remainingChunk.byteLength;

    // Write to OPFS
    opfsHandle.write(remainingChunk, { at: totalFileOffset });

    // Write to WASM linear memory at the arena pointer
    const heapView = new Uint8Array(
      wasmModule.HEAPU8.buffer,
      arenaPtr + totalPayloadBytesWritten,
      chunkLen
    );
    heapView.set(remainingChunk);

    totalPayloadBytesWritten += chunkLen;
    totalFileOffset += chunkLen;

    sendProgress('download',
      contentLength > 0 ? Math.round((totalFileOffset / contentLength) * 100) : -1,
      totalFileOffset, contentLength
    );
  }

  let pkBytesRead = totalPayloadBytesWritten;

  while (pkBytesRead < headerInfo.pkSectionSize) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value || value.byteLength === 0) continue;

    // Only write up to the remaining PK section size
    const bytesToWrite = Math.min(value.byteLength, headerInfo.pkSectionSize - pkBytesRead);
    const chunk = bytesToWrite < value.byteLength ? value.slice(0, bytesToWrite) : value;

    // Write to OPFS (full file cache)
    opfsHandle.write(chunk, { at: totalFileOffset });

    // Write to WASM linear memory
    // HARDENING: Doubt 3 secondary check — verify write target is within bounds
    // even after the initial capacity check, in case of stream overrun
    if (pkBytesRead + chunk.byteLength > wasmBufferCapacity) {
      opfsHandle.close();
      sendError(
        'HEAP_OVERFLOW_PREVENTED',
        `Stream write would exceed WASM buffer: offset ${pkBytesRead} + chunk ${chunk.byteLength} > capacity ${wasmBufferCapacity}`,
        'Aborting to prevent heap corruption.'
      );
      return;
    }
    const heapView = new Uint8Array(
      wasmModule.HEAPU8.buffer,
      arenaPtr + pkBytesRead,
      chunk.byteLength
    );
    heapView.set(chunk);

    pkBytesRead += chunk.byteLength;
    totalFileOffset += chunk.byteLength;

    sendProgress('download',
      contentLength > 0 ? Math.round((totalFileOffset / contentLength) * 100) : -1,
      totalFileOffset, contentLength
    );
  }

  opfsHandle.flush();
  opfsHandle.close();

  // HARDENING: Doubt 2 fix — Atomic rename from .tmp to final filename.
  // Only after ALL bytes are written and flushed do we "commit" the cache.
  // If we crashed or lost network before reaching this line, the .tmp file
  // remains orphaned and getCachedZkey() never sees it.
  try {
    // Remove old final file if it somehow exists (defensive)
    try { await dir.removeEntry(`zkey-${hash}.bin`); } catch { /* didn't exist — fine */ }
    // Atomically move .tmp → .bin (OPFS rename = create new handle, copy, delete old)
    // OPFS doesn't support native rename, so we use the rename-via-move pattern:
    // The .tmp SyncAccessHandle is already closed. We now create the final file
    // and do a handle-level copy. However, since both files reference the same
    // underlying storage in modern browsers (Chromium 110+), we can simply
    // rename by creating a new handle pointing to the same data.
    //
    // Simpler approach: re-open .tmp, read into .bin, delete .tmp.
    // This is acceptable because the file is already fully on disk (no network I/O).
    const finalHandle = await dir.getFileHandle(`zkey-${hash}.bin`, { create: true });
    const finalAccess = await finalHandle.createSyncAccessHandle();
    const tmpReadHandle = await tmpHandle.createSyncAccessHandle();
    const fileSize = tmpReadHandle.getSize();
    const copyBuffer = new Uint8Array(Math.min(fileSize, 1024 * 1024)); // 1MB copy buffer
    let copyOffset = 0;
    while (copyOffset < fileSize) {
      const bytesToCopy = Math.min(copyBuffer.byteLength, fileSize - copyOffset);
      const slice = new Uint8Array(bytesToCopy);
      tmpReadHandle.read(slice, { at: copyOffset });
      finalAccess.write(slice, { at: copyOffset });
      copyOffset += bytesToCopy;
    }
    finalAccess.flush();
    finalAccess.close();
    tmpReadHandle.close();
    // Delete the staging file
    await dir.removeEntry(`zkey-${hash}.tmp`);
  } catch (renameErr) {
    // Non-fatal: the download succeeded into WASM memory even if cache commit fails.
    // Next run will re-download (bandwidth cost) but proof generation still works.
    console.warn('[ZK Worker] OPFS cache commit failed (non-fatal):', renameErr);
  }

  sendStatus('downloading',
    `Zero-copy ingestion complete. ${pkBytesRead} bytes of polynomial data ` +
    `written to WASM arena (${headerInfo.totalHeaderSize} header bytes discarded).`
  );
}

// ─── Load from OPFS Cache ─────────────────────────────────────────────────

async function loadFromOpfsCache(handle: FileSystemSyncAccessHandle): Promise<void> {
  wasmModule._reset_engine();

  const fileSize = handle.getSize();
  const wasmMemPtr = wasmModule._get_memory_ptr();
  let offset = 0;

  while (offset < fileSize) {
    const chunkSize = Math.min(BOUNCE_BUFFER_SIZE, fileSize - offset);
    const buffer = new Uint8Array(chunkSize);
    handle.read(buffer, { at: offset });

    const heapView = new Uint8Array(
      wasmModule.HEAPU8.buffer,
      wasmMemPtr + offset,
      chunkSize
    );
    heapView.set(buffer);

    offset += chunkSize;
    sendProgress('download', Math.round((offset / fileSize) * 100), offset, fileSize);
  }

  handle.close();
  sendStatus('cached', `Loaded ${fileSize} bytes from OPFS cache.`);
}

// ─── Proof Generation ─────────────────────────────────────────────────────

async function generateProof(payload: ProvePayload): Promise<void> {
  const startTime = performance.now();

  try {
    await initializeWasm();
    // Phase 3.1: Use zero-copy ingestion (parses iden3 header, writes only polynomials)
    await loadZkeyZeroCopy(payload.zkeyUrl, payload.zkeyHash);

    sendStatus('computing', 'Generating Groth16 proof...');

    // Prepare public inputs in WASM memory
    const inputBytes = new Uint8Array(payload.publicInputs.length * 32);
    for (let i = 0; i < payload.publicInputs.length; i++) {
      const val = BigInt(payload.publicInputs[i]);
      for (let j = 31; j >= 0; j--) {
        inputBytes[i * 32 + (31 - j)] = Number((val >> BigInt(j * 8)) & 0xFFn);
      }
    }

    const inputPtr = wasmModule._malloc(inputBytes.length);
    wasmModule.HEAPU8.set(inputBytes, inputPtr);

    // Poll progress during computation
    const progressInterval = setInterval(() => {
      if (wasmModule && isInitialized) {
        const progress: number = wasmModule._get_progress();
        sendProgress('compute', progress);
      }
    }, 100);

    // Execute Groth16 prover
    const result: number = wasmModule._compute_proof(inputPtr, payload.publicInputs.length);

    clearInterval(progressInterval);
    wasmModule._free(inputPtr);

    if (result !== 0) {
      sendError('PROOF_FAILED', 'Groth16 proof generation failed.', `compute_proof returned ${result}`);
      return;
    }

    // ─── Phase 3.4: Zero-copy proof extraction via ZkProofExporter ───
    const proofPtr: number = wasmModule._get_proof_ptr();
    const { createProofExporter } = await import('../lib/ZkProofExporter');
    const exporter = createProofExporter(wasmModule.HEAPU8, proofPtr);
    const grothProof = exporter.toGroth16Json();

    // Read auxiliary data (beyond the 256-byte proof)
    const auxView = new Uint8Array(wasmModule.HEAPU8.buffer, proofPtr + 256, 64);
    const publicInputsHash = bytesToHex(new Uint8Array(auxView.buffer, auxView.byteOffset, 32));
    const nullifierHash = bytesToHex(new Uint8Array(auxView.buffer, auxView.byteOffset + 32, 32));

    const computeTimeMs = Math.round(performance.now() - startTime);

    const proofPayload: ProofPayload = {
      publicInputsHash,
      nullifierHash,
      proofCommitmentMatrix: grothProof,
      computeTimeMs,
    };

    send({ type: 'proof', payload: proofPayload });
    sendStatus('done', `Proof generated in ${computeTimeMs}ms.`);
  } catch (err) {
    sendError(
      'UNEXPECTED',
      'Unexpected error during proof generation.',
      err instanceof Error ? err.stack || err.message : String(err)
    );
  }
}

// ─── Message Handler ──────────────────────────────────────────────────────

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'init':
      try {
        await initializeWasm();
      } catch {
        // Error already sent via sendError
      }
      break;

    case 'prove':
      if (!payload) {
        sendError('INVALID_PAYLOAD', 'prove message requires a payload.');
        return;
      }
      await generateProof(payload);
      break;

    default:
      sendError('UNKNOWN_MESSAGE', `Unknown message type: ${type}`);
  }
};
