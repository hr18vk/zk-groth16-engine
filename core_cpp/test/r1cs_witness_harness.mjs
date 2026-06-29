/**
 * ZK Engine Phase 3.3
 * R1CS Witness Engine — Node.js Integration Test Harness
 *
 * This harness loads the WASM module compiled from witness_engine.cpp
 * and verifies the R1CS constraint evaluation pipeline.
 *
 * Tests:
 *   1. C++ self-test (signal sanitizer, R1CS eval, DAG scheduler, prime bound)
 *   2. ABI export verification (all functions accessible)
 *   3. Memory layout validation (arena pointer, capacity)
 *   4. Sequential R1CS evaluation from JavaScript
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const require = createRequire(import.meta.url);

async function main() {
  console.log('');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('  ZK Engine Phase 3.3 — R1CS Witness Engine Test');
  console.log('═══════════════════════════════════════════════════════════');
  console.log('');

  // Load the WASM module
  const modulePath = join(__dirname, '..', 'build', 'r1cs_witness_test.js');
  const createR1csTest = require(modulePath);

  const wasmModule = await createR1csTest();

  let testsRun = 0;
  let testsPassed = 0;

  function check(name, condition) {
    testsRun++;
    if (condition) {
      testsPassed++;
      console.log(`  ✅ ${name}`);
    } else {
      console.log(`  ❌ ${name}`);
    }
  }

  // ─── Test 1: C++ Self-Test ────────────────────────────────────────
  console.log('─── C++ Self-Tests ────────────────────────────────────');
  const selfTestResult = wasmModule._r1cs_selftest();
  check('r1cs_selftest() returns 0 (all C++ tests passed)', selfTestResult === 0);
  if (selfTestResult !== 0) {
    console.log(`     Self-test returned error code: ${selfTestResult}`);
  }

  // ─── Test 2: ABI Export Verification ──────────────────────────────
  console.log('');
  console.log('─── ABI Export Verification ────────────────────────────');

  const expectedExports = [
    '_r1cs_init',
    '_r1cs_ingest_constraint',
    '_r1cs_assign_signal',
    '_r1cs_finalize',
    '_r1cs_evaluate',
    '_r1cs_get_signal_ptr',
    '_r1cs_get_constraint_count',
    '_r1cs_get_nnz',
    '_r1cs_get_arena_ptr',
    '_r1cs_get_arena_capacity',
    '_r1cs_register_subcircuit',
    '_r1cs_add_dependency',
    '_r1cs_selftest',
    '_malloc',
    '_free',
  ];

  for (const exportName of expectedExports) {
    check(
      `Export ${exportName} is a function`,
      typeof wasmModule[exportName] === 'function'
    );
  }

  // ─── Test 3: Memory Layout Validation ─────────────────────────────
  console.log('');
  console.log('─── Memory Layout Validation ──────────────────────────');

  // Initialize with small capacity for validation
  const initResult = wasmModule._r1cs_init(4, 8, 16);
  check('r1cs_init(4, 8, 16) succeeds', initResult === 0);

  const arenaPtr = wasmModule._r1cs_get_arena_ptr();
  check('Arena pointer is non-zero', arenaPtr !== 0);

  const arenaCapacity = wasmModule._r1cs_get_arena_capacity();
  check('Arena capacity is 16 MB (16777216)', arenaCapacity === 16 * 1024 * 1024);

  // ─── Test 4: JavaScript-Side R1CS Evaluation ──────────────────────
  console.log('');
  console.log('─── JavaScript-Side R1CS Evaluation ────────────────────');

  // Re-init for a clean test: 1 constraint, 4 signals, 4 nnz per matrix
  const init2 = wasmModule._r1cs_init(1, 4, 4);
  check('Re-init r1cs_init(1, 4, 4) succeeds', init2 === 0);

  // Allocate a 32-byte buffer in WASM memory for signal assignment
  const sigBufPtr = wasmModule._malloc(32);
  const sigBuf = new Uint8Array(wasmModule.HEAPU8.buffer, sigBufPtr, 32);

  // Signal 0 = 1 (constant one)
  sigBuf.fill(0);
  sigBuf[31] = 1;
  check('Assign signal 0 = 1', wasmModule._r1cs_assign_signal(0, sigBufPtr, 32) === 0);

  // Signal 1 = 7
  sigBuf.fill(0);
  sigBuf[31] = 7;
  check('Assign signal 1 = 7', wasmModule._r1cs_assign_signal(1, sigBufPtr, 32) === 0);

  // Signal 2 = 11
  sigBuf.fill(0);
  sigBuf[31] = 11;
  check('Assign signal 2 = 11', wasmModule._r1cs_assign_signal(2, sigBufPtr, 32) === 0);

  // Signal 3 = 77 (7 * 11)
  sigBuf.fill(0);
  sigBuf[31] = 77;
  check('Assign signal 3 = 77', wasmModule._r1cs_assign_signal(3, sigBufPtr, 32) === 0);

  // Allocate 32-byte coefficient buffer
  const coeffPtr = wasmModule._malloc(32);
  const coeffBuf = new Uint8Array(wasmModule.HEAPU8.buffer, coeffPtr, 32);

  // Coefficient = 1
  coeffBuf.fill(0);
  coeffBuf[31] = 1;

  // A[0][1] = 1
  check('Ingest A[0][1]=1',
    wasmModule._r1cs_ingest_constraint(0, 0, 1, coeffPtr) === 0);
  // B[0][2] = 1
  check('Ingest B[0][2]=1',
    wasmModule._r1cs_ingest_constraint(1, 0, 2, coeffPtr) === 0);
  // C[0][3] = 1
  check('Ingest C[0][3]=1',
    wasmModule._r1cs_ingest_constraint(2, 0, 3, coeffPtr) === 0);

  // Finalize
  check('r1cs_finalize succeeds', wasmModule._r1cs_finalize() === 0);

  // Evaluate: 7 * 11 = 77 → should pass
  const evalResult = wasmModule._r1cs_evaluate();
  check('R1CS evaluation passes (7 * 11 = 77)', evalResult === 0);

  // Verify nnz counts
  check('Matrix A has 1 non-zero entry', wasmModule._r1cs_get_nnz(0) === 1);
  check('Matrix B has 1 non-zero entry', wasmModule._r1cs_get_nnz(1) === 1);
  check('Matrix C has 1 non-zero entry', wasmModule._r1cs_get_nnz(2) === 1);

  // Clean up
  wasmModule._free(sigBufPtr);
  wasmModule._free(coeffPtr);

  // ─── Summary ──────────────────────────────────────────────────────
  console.log('');
  console.log('═══════════════════════════════════════════════════════════');
  if (testsRun === testsPassed) {
    console.log(`  ✅ ALL ${testsRun} PHASE 3.3 VERIFICATION CHECKS PASSED`);
  } else {
    console.log(`  ❌ PHASE 3.3 VERIFICATION FAILED: ${testsPassed}/${testsRun} passed`);
    process.exit(1);
  }
  console.log('═══════════════════════════════════════════════════════════');
}

main().catch((err) => {
  console.error('FATAL:', err);
  process.exit(1);
});
