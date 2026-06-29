/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ZK Engine Phase 2.3
 * Pthread Orchestration Test Harness (Node.js ESM)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This script tests the multi-threaded futex synchronization pipeline:
 *   1. Load WASM module with pthread support
 *   2. Call orchestrate_parallel_proof_matrix() (spawns C++ thread → futex sleep)
 *   3. Write mock payload into shared memory via HEAPU8 (zero-copy)
 *   4. Flip synchronization flag from 0 → 1
 *   5. Call Atomics.notify() to wake the C++ thread
 *   6. Poll for completion flag (2)
 *   7. Verify the XOR-rotate checksum
 *   8. MANDATORY: Call PThread.terminateAllThreads() to prevent Node hang
 *
 * Usage: node --experimental-wasm-threads --experimental-wasm-bulk-memory \
 *             pthread_harness.mjs
 *
 * EXIT CODES:
 *   0 = All checks passed
 *   1 = One or more checks failed
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// The Emscripten glue code is CommonJS, so we need createRequire
const require = createRequire(import.meta.url);

console.log('═══════════════════════════════════════════════════════════');
console.log('  ZK Engine Phase 2.3 — Pthread Orchestration Test');
console.log('  Futex Synchronization & Shared Memory Verification');
console.log('═══════════════════════════════════════════════════════════\n');

let exitCode = 0;

// ─── STEP 1: Load the WASM Module ─────────────────────────────────────────
console.log('[STEP 1] Loading pthread-enabled WASM module...');

const modulePath = join(__dirname, '..', 'build', 'pthread_prover.js');
const createModule = require(modulePath);

let Module;
try {
    Module = await createModule({
        // Emscripten pthread configuration for Node.js
        // These are critical for headless thread spawning
        locateFile: (path) => {
            return join(__dirname, '..', 'build', path);
        },
    });
    console.log('  Module loaded successfully ✓');
    console.log(`  HEAPU8 length: ${Module.HEAPU8.length} bytes`);
} catch (err) {
    console.error(`  [FATAL] Module load failed: ${err.message}`);
    process.exit(1);
}

// ─── STEP 2: Initialize Field Constants ───────────────────────────────────
console.log('\n[STEP 2] Initializing field constants...');
Module._zk_engine_field_init_full();
console.log('  Field init complete ✓');

// ─── STEP 3: Get Shared Arena Pointer ─────────────────────────────────────
console.log('\n[STEP 3] Resolving shared memory arena...');
const arenaPtr = Module._get_shared_arena_ptr();
const arenaCapacity = Module._get_shared_arena_capacity();
console.log(`  Arena pointer:  ${arenaPtr} (WASM linear memory offset)`);
console.log(`  Arena capacity: ${arenaCapacity} bytes (${arenaCapacity / (1024 * 1024)} MB)`);

if (arenaCapacity !== 64 * 1024 * 1024) {
    console.error(`  [FAIL] Expected 64MB arena, got ${arenaCapacity}`);
    exitCode = 1;
} else {
    console.log('  Arena capacity check: 64MB ✓');
}

// ─── STEP 4: Get Sync Flag Pointer ────────────────────────────────────────
console.log('\n[STEP 4] Resolving synchronization flag...');
const flagPtr = Module._get_sync_flag_ptr();
console.log(`  Flag pointer: ${flagPtr} (WASM linear memory offset)`);

// The flag is a uint32 (4 bytes), so its Int32Array index is flagPtr / 4
const flagIndex = flagPtr >> 2; // Equivalent to Math.floor(flagPtr / 4)
console.log(`  Flag Int32Array index: ${flagIndex}`);

// Verify initial state is 0 (WAITING)
const flagView = new Int32Array(Module.HEAPU8.buffer);
const initialFlag = Atomics.load(flagView, flagIndex);
console.log(`  Initial flag value: ${initialFlag} (expected: 0)`);
if (initialFlag !== 0) {
    console.error(`  [FAIL] Flag should be 0 (WAITING), got ${initialFlag}`);
    exitCode = 1;
} else {
    console.log('  Initial flag state: WAITING (0) ✓');
}

// ─── STEP 5: Spawn C++ Thread ─────────────────────────────────────────────
console.log('\n[STEP 5] Spawning C++ orchestration thread...');
console.log('  Calling orchestrate_parallel_proof_matrix(arenaPtr, 256)...');
console.log('  (Thread will immediately enter futex sleep)');

// The payload length we'll write
const PAYLOAD_LEN = 256;

// Call the function — this spawns a C++ worker thread that immediately
// enters futex sleep, and returns control to JavaScript immediately.
Module._start_orchestration(PAYLOAD_LEN);

console.log('  Thread spawned and parked in futex sleep ✓');

// ─── STEP 6: Write Mock Payload (Zero-Copy) ──────────────────────────────
console.log('\n[STEP 6] Writing mock payload to shared memory arena...');

// Simulate 250ms delay (as mandated by the sprint directive)
console.log('  Waiting 250ms to simulate async payload arrival...');
await sleep(250);

// Write a deterministic pattern into the shared arena via HEAPU8
// Pattern: bytes 0x00, 0x01, 0x02, ..., 0xFF (repeating)
for (let i = 0; i < PAYLOAD_LEN; i++) {
    Module.HEAPU8[arenaPtr + i] = i & 0xFF;
}
console.log(`  Wrote ${PAYLOAD_LEN} bytes to arena at offset ${arenaPtr}`);
console.log('  First 16 bytes: ' +
    Array.from(Module.HEAPU8.slice(arenaPtr, arenaPtr + 16))
        .map(b => '0x' + b.toString(16).padStart(2, '0'))
        .join(' '));

// ─── STEP 7: Compute Expected Checksum ────────────────────────────────────
// Mirror the C++ XOR-rotate-left algorithm
let expectedChecksum = 0;
for (let i = 0; i < PAYLOAD_LEN; i++) {
    expectedChecksum ^= (i & 0xFF);
    expectedChecksum = ((expectedChecksum << 1) | (expectedChecksum >>> 31)) >>> 0;
}
console.log(`  Expected checksum: 0x${expectedChecksum.toString(16).padStart(8, '0').toUpperCase()}`);

// ─── STEP 8: Wake the C++ Thread ──────────────────────────────────────────
console.log('\n[STEP 8] Waking C++ thread via Atomics...');

// Step 8a: Flip the flag from 0 → 1 (READY)
Atomics.store(flagView, flagIndex, 1);
console.log('  Flag set to 1 (READY)');

// Step 8b: Wake any thread waiting on this address
const wokenCount = Atomics.notify(flagView, flagIndex, 1);
console.log(`  Atomics.notify() woke ${wokenCount} thread(s)`);

// ─── STEP 9: Wait for C++ Thread Completion ──────────────────────────────
console.log('\n[STEP 9] Waiting for C++ thread completion...');

// Poll for the flag to transition to 2 (COMPLETE)
// We use a short polling loop with sleep intervals — NOT a busy-spin.
// This is the JS side, so we can't use Atomics.wait (it blocks the main thread).
const MAX_WAIT_MS = 5000;
const POLL_INTERVAL_MS = 50;
let elapsed = 0;

while (elapsed < MAX_WAIT_MS) {
    const currentFlag = Atomics.load(flagView, flagIndex);
    if (currentFlag === 2) {
        console.log(`  Thread completed after ${elapsed}ms ✓`);
        break;
    }
    await sleep(POLL_INTERVAL_MS);
    elapsed += POLL_INTERVAL_MS;
}

const finalFlag = Atomics.load(flagView, flagIndex);
if (finalFlag !== 2) {
    console.error(`  [FAIL] Thread did not complete within ${MAX_WAIT_MS}ms. Flag = ${finalFlag}`);
    exitCode = 1;
}

// ─── STEP 10: Verify Checksum ─────────────────────────────────────────────
console.log('\n[STEP 10] Verifying orchestration result...');

const result = Module._get_orchestration_result();
console.log(`  C++ checksum:    0x${(result >>> 0).toString(16).padStart(8, '0').toUpperCase()}`);
console.log(`  JS expected:     0x${expectedChecksum.toString(16).padStart(8, '0').toUpperCase()}`);

if ((result >>> 0) === (expectedChecksum >>> 0)) {
    console.log('  Checksum verification: MATCH ✓');
} else {
    console.error('  Checksum verification: MISMATCH ✗');
    exitCode = 1;
}

// ─── STEP 11: Terminate All Threads ───────────────────────────────────────
console.log('\n[STEP 11] Terminating pthread pool...');

// CRITICAL: Without this call, the Node.js event loop hangs indefinitely
// because the Web Worker threads spawned by Emscripten's pthread implementation
// keep the event loop alive. PThread.terminateAllThreads() sends a termination
// message to each worker and allows Node to exit cleanly.
if (Module.PThread && typeof Module.PThread.terminateAllThreads === 'function') {
    Module.PThread.terminateAllThreads();
    console.log('  All threads terminated ✓');
} else {
    console.warn('  [WARN] PThread.terminateAllThreads not available');
    console.warn('  This may cause the Node.js process to hang.');
}

// ─── Final Summary ────────────────────────────────────────────────────────
console.log('\n═══════════════════════════════════════════════════════════');
if (exitCode === 0) {
    console.log('  ✅ ALL PTHREAD ORCHESTRATION CHECKS PASSED');
} else {
    console.log('  ❌ PTHREAD ORCHESTRATION FAILED — See errors above');
}
console.log('═══════════════════════════════════════════════════════════\n');

process.exit(exitCode);
