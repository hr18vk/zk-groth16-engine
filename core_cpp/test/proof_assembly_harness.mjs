/**
 * Proof Assembly Test Harness -- Sprint 44 (Phase 3.4)
 *
 * Loads the WASM module compiled from prover_engine.cpp (which includes
 * proof_assembly.cpp), initializes the BN254 field, and runs the
 * comprehensive self-test suite (8 tests).
 *
 * Usage: node test/proof_assembly_harness.mjs
 * Build: make test-proof-assembly
 */

import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function main() {
  console.log('+=======================================================+');
  console.log('|  ZK Engine Phase 3.4 -- Proof Assembly Tests   |');
  console.log('+=======================================================+');
  console.log('');

  // Dynamically import the Emscripten glue module
  const modulePath = path.join(__dirname, '..', 'build', 'proof_assembly_test.js');
  const { default: createModule } = await import(modulePath);

  const wasmModule = await createModule({
    locateFile: (file) => path.join(__dirname, '..', 'build', file),
  });

  // Initialize BN254 field (P_LIMBS, P_PRIME_0, R_SQUARED)
  wasmModule._zk_engine_field_init_full();
  console.log('  > Field initialized (zk_engine_field_init_full)');
  console.log('');

  // Execute self-test suite
  const result = wasmModule._proof_assembly_selftest();
  const total  = (result >>> 16) & 0xFFFF;
  const passed = result & 0xFFFF;

  const labels = [
    '1. Fp2 multiplication identity',
    '2. Fp2 squaring consistency',
    '3. fp_inv correctness (a*a^-1=1)',
    '4. fp2_inv correctness',
    '5. BE swizzle reversal',
    '6. Double swizzle = identity',
    '7. Batch G1 inversion round-trip',
    '8. Quotient computation (A*B=C)',
  ];

  for (let i = 0; i < labels.length; i++) {
    const status = passed > i ? 'PASS' : 'FAIL';
    const icon = passed > i ? '+' : 'X';
    console.log(`    [${icon}] ${labels[i]}:  ${status}`);
  }

  console.log('');
  console.log(`  Score: ${passed}/${total}`);
  console.log('');

  if (passed === total) {
    console.log('  ALL 8 TESTS PASSED -- Phase 3.4 verified.');
    process.exit(0);
  } else {
    console.log(`  ${total - passed} TEST(S) FAILED -- investigation required.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error in test harness:', err);
  process.exit(1);
});
