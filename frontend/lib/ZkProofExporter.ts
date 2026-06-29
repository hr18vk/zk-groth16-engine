/**
 * ZkProofExporter -- Zero-Copy Proof Extraction from WASM Linear Memory
 *
 * Zero-Knowledge Engine
 *
 * Creates a Uint8Array view directly into the SharedArrayBuffer at the
 * proof pointer offset. No memory copy occurs until explicit export methods
 * are called.
 *
 * Proof Memory Layout (256 bytes, big-endian after SIMD swizzle):
 *   [0..31]    pi_A.x       [32..63]    pi_A.y
 *   [64..95]   pi_B.x.c0    [96..127]   pi_B.x.c1
 *   [128..159] pi_B.y.c0    [160..191]  pi_B.y.c1
 *   [192..223] pi_C.x       [224..255]  pi_C.y
 */

/** snarkjs-compatible Groth16 proof structure */
export interface Groth16Proof {
  readonly pi_a: [string, string, string];
  readonly pi_b: [[string, string], [string, string], [string, string]];
  readonly pi_c: [string, string, string];
  readonly protocol: 'groth16';
  readonly curve: 'bn128';
}

/** Handle for zero-copy proof access */
export interface ProofExportHandle {
  /** Zero-copy view into WASM memory. Valid until next compute_proof() call. */
  readonly rawView: Uint8Array;
  /** Parse proof into snarkjs-compatible JSON format */
  toGroth16Json(): Groth16Proof;
  /** Export as hex-encoded calldata for Solidity verifier */
  toSolidityCalldata(): string;
  /** Get raw 256-byte proof as Uint8Array (copies on first access, cached) */
  toBytes(): Uint8Array;
}

/**
 * Extracts a 32-byte field element at the given offset as a hex string.
 * Reads from the zero-copy view -- no intermediate allocation.
 */
function fieldToHex(view: Uint8Array, offset: number): string {
  let hex = '0x';
  for (let i = offset; i < offset + 32; i++) {
    hex += view[i].toString(16).padStart(2, '0');
  }
  return hex;
}

/**
 * Creates a zero-copy proof export handle.
 *
 * @param wasmHeap - The WASM module's HEAPU8 typed array
 * @param proofPtr - Pointer returned by _get_proof_ptr()
 * @returns ProofExportHandle with lazy parsing
 */
export function createProofExporter(
  wasmHeap: Uint8Array,
  proofPtr: number
): ProofExportHandle {
  // Zero-copy view: 256 bytes starting at proofPtr
  // WARNING: In Emscripten with SHARED_MEMORY=1, wasmHeap.buffer is a SharedArrayBuffer.
  // Passing `rawView` directly to standard ArrayBuffer APIs may throw `TypeError`.
  const rawView = new Uint8Array(wasmHeap.buffer, proofPtr, 256);

  let cachedBytes: Uint8Array | null = null;

  return {
    get rawView() {
      return rawView;
    },

    toGroth16Json(): Groth16Proof {
      return {
        // SnarkJS strictly requires projective representation with Z="1"
        pi_a: [fieldToHex(rawView, 0), fieldToHex(rawView, 32), "1"],
        // SnarkJS expects G2 points in [c1, c0] (imaginary, real) order
        pi_b: [
          [fieldToHex(rawView, 96), fieldToHex(rawView, 64)],
          [fieldToHex(rawView, 160), fieldToHex(rawView, 128)],
          ["1", "0"],
        ],
        pi_c: [fieldToHex(rawView, 192), fieldToHex(rawView, 224), "1"],
        protocol: 'groth16',
        curve: 'bn128',
      };
    },

    toSolidityCalldata(): string {
      // EVM Precompile 0x08 strictly requires G2 elements in [c1, c0] order!
      let calldata = '0x';
      const order = [0, 32, 96, 64, 160, 128, 192, 224];
      for (const offset of order) {
        calldata += fieldToHex(rawView, offset).slice(2).padStart(64, '0');
      }
      return calldata;
    },

    toBytes(): Uint8Array {
      if (!cachedBytes) {
        cachedBytes = new Uint8Array(256);
        cachedBytes.set(rawView);
      }
      return cachedBytes;
    },
  };
}
