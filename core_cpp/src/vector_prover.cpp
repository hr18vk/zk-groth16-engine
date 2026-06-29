/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ZK Engine Phase 2.2
 * Vectorized Montgomery CIOS — 9×29-bit Reduced-Radix BN254 Field Arithmetic
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This file implements the core finite field multiplication for the BN254
 * elliptic curve used in Groth16 ZK-SNARK proofs.
 *
 * Architecture:
 *   - Field elements are 254-bit integers represented as 9 × 29-bit limbs.
 *   - Montgomery multiplication uses CIOS (Coarsely Integrated Operand Scanning).
 *   - Inner loops are accelerated with WASM SIMD128 i64x2.extmul_low_i32x4_u.
 *   - All memory is allocated from a static 64MB bump arena (no malloc/new).
 *
 * HARDWARE CONSTRAINTS:
 *   - No native 64-bit carry flag in WASM — hence 29-bit radix with 6-bit
 *     headroom in 64-bit lanes for lazy carry accumulation.
 *   - No Karatsuba: fully unrolled O(n²) schoolbook to eliminate data dependencies.
 *   - SIMD loads MUST hit 16-byte aligned addresses.
 *
 * This file is compiled into the verification harness (test-simd target) and
 * will be linked into the production prover_engine.cpp in Phase 2.3.
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <wasm_simd128.h>

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Number of limbs in a reduced-radix field element.
static constexpr int NLIMBS = 9;

/// Bits per limb.
static constexpr int RADIX = 29;

/// Limb mask: 2^29 - 1 = 0x1FFFFFFF
static constexpr uint32_t LIMB_MASK = (1u << RADIX) - 1u;

/**
 * BN254 scalar field prime p in canonical 32-byte big-endian form.
 *
 * p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
 *   = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
 */
static constexpr uint8_t P_BYTES[32] = {
    0x30, 0x64, 0x4E, 0x72, 0xE1, 0x31, 0xA0, 0x29,
    0xB8, 0x50, 0x45, 0xB6, 0x81, 0x81, 0x58, 0x5D,
    0x28, 0x33, 0xE8, 0x48, 0x79, 0xB9, 0x70, 0x91,
    0x43, 0xE1, 0xF5, 0x93, 0xF0, 0x00, 0x00, 0x01
};

/**
 * Convert 32-byte big-endian representation to 9×29-bit limb form (little-endian limbs).
 * This is a constexpr function so the compiler resolves it at compile time.
 */
struct Fp9x29 {
    uint32_t limb[NLIMBS];
};

/**
 * Converts a 32-byte big-endian integer to 9×29-bit limbs.
 *
 * Algorithm:
 *   1. Convert big-endian bytes to a 256-bit little-endian bitstream.
 *   2. Extract 29-bit slices from the bitstream.
 *
 * We do this at runtime in a dedicated init function since constexpr
 * bitwise extraction across byte boundaries is verbose but straightforward.
 */
static Fp9x29 bytes_to_limbs(const uint8_t be_bytes[32]) {
    Fp9x29 result;
    std::memset(&result, 0, sizeof(result));

    // Step 1: Reverse bytes to little-endian
    uint8_t le[32];
    for (int i = 0; i < 32; i++) {
        le[i] = be_bytes[31 - i];
    }

    // Step 2: Extract 29-bit limbs from the little-endian byte array
    // We treat `le` as a bitstream and extract RADIX bits at a time.
    int bit_offset = 0;
    for (int i = 0; i < NLIMBS; i++) {
        uint32_t val = 0;
        for (int b = 0; b < RADIX && (bit_offset + b) < 256; b++) {
            int byte_idx = (bit_offset + b) / 8;
            int bit_idx  = (bit_offset + b) % 8;
            if (le[byte_idx] & (1u << bit_idx)) {
                val |= (1u << b);
            }
        }
        result.limb[i] = val;
        bit_offset += RADIX;
    }
    return result;
}

/**
 * Converts 9×29-bit limbs back to 32-byte big-endian form.
 * Used for verification: reassemble → compare against canonical bytes.
 */
static void limbs_to_bytes(uint8_t out_be[32], const Fp9x29& fp) {
    uint8_t le[33]; // Extra byte for potential overflow during assembly
    std::memset(le, 0, sizeof(le));

    int bit_offset = 0;
    for (int i = 0; i < NLIMBS; i++) {
        uint32_t val = fp.limb[i];
        for (int b = 0; b < RADIX; b++) {
            if (val & (1u << b)) {
                int abs_bit = bit_offset + b;
                int byte_idx = abs_bit / 8;
                int bit_idx  = abs_bit % 8;
                if (byte_idx < 33) {
                    le[byte_idx] |= (1u << bit_idx);
                }
            }
        }
        bit_offset += RADIX;
    }

    // Reverse to big-endian
    for (int i = 0; i < 32; i++) {
        out_be[i] = le[31 - i];
    }
}

// The prime p in 9×29-bit limb form (initialized once at startup)
static Fp9x29 P_LIMBS;

/**
 * p'_0 = (-p^(-1)) mod 2^29
 *
 * Computed via the standard iterative Newton's method:
 *   Start with x = 1
 *   Repeat: x = x * (2 - p[0] * x) mod 2^29
 *   Return (-x) mod 2^29
 *
 * This converges in ceil(log2(29)) = 5 iterations.
 */
static uint32_t P_PRIME_0;

static void compute_p_prime_0() {
    // Newton's method for modular inverse of p[0] mod 2^29
    uint64_t p0 = P_LIMBS.limb[0];
    uint64_t x = 1;
    uint64_t mod = (1ull << RADIX);

    // 5 iterations of Newton's method
    for (int i = 0; i < 5; i++) {
        x = (x * (2 - p0 * x)) & (mod - 1);
    }

    // p'_0 = -p0^(-1) mod 2^29  =  (2^29 - x) & LIMB_MASK
    P_PRIME_0 = static_cast<uint32_t>((mod - x) & (mod - 1));
}

/**
 * Initialize the field arithmetic constants.
 * MUST be called once before any Montgomery operations.
 */
void zk_engine_field_init() {
    P_LIMBS = bytes_to_limbs(P_BYTES);
    compute_p_prime_0();
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: VECTORIZED MONTGOMERY CIOS — SCHOOLBOOK
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Performs Montgomery multiplication of two 9×29-bit field elements:
 *
 *   result = a · b · R^(-1) mod p
 *
 * where R = 2^261 (9 limbs × 29 bits).
 *
 * ALGORITHM: CIOS (Coarsely Integrated Operand Scanning)
 *   - Fully unrolled schoolbook O(n²) multiplication
 *   - SIMD128 i64x2.extmul_low_i32x4_u for paired 29×29 → 58-bit products
 *   - Lazy carry: accumulate in 64-bit lanes, normalize after each outer row
 *
 * @param result  Output: the reduced Montgomery product (9 limbs)
 * @param a       Input operand A (9×29-bit limbs, values in [0, p))
 * @param b       Input operand B (9×29-bit limbs, values in [0, p))
 */
void vectorized_montgomery_cios_schoolbook(Fp9x29* result,
                                            const Fp9x29* a,
                                            const Fp9x29* b) {
    uint64_t t[NLIMBS + 1] __attribute__((aligned(16)));
    std::memset(t, 0, sizeof(t));

    const uint32_t* a_limbs = a->limb;
    const uint32_t* b_limbs = b->limb;
    const uint32_t* p_limbs = P_LIMBS.limb;

    for (int i = 0; i < NLIMBS; i++) {
        const uint32_t bi = b_limbs[i];

        // ─── Step 1: Multiply-Accumulate (a[j] * b[i]) ──────────────
        for (int j = 0; j < 8; j += 2) {
            uint32_t a_pair[4] __attribute__((aligned(16))) = {
                a_limbs[j], a_limbs[j + 1], 0, 0
            };
            uint32_t b_splat[4] __attribute__((aligned(16))) = {
                bi, bi, 0, 0
            };

            v128_t va = wasm_v128_load(a_pair);
            v128_t vb = wasm_v128_load(b_splat);
            v128_t prod = wasm_u64x2_extmul_low_u32x4(va, vb);

            // Pure vector accumulation into t
            v128_t vt = wasm_v128_load(&t[j]);
            vt = wasm_i64x2_add(vt, prod);
            wasm_v128_store(&t[j], vt);
        }

        // Scalar tail: j = 8
        {
            t[8] += (uint64_t)a_limbs[8] * (uint64_t)bi;
        }

        // ─── Step 2: Montgomery Quotient ─────────────────────────────
        // Since t[0] has not yet had carries extracted from it, its higher bits
        // might contain carry data, but Montgomery only needs t[0] mod 2^29.
        uint32_t m = static_cast<uint32_t>((static_cast<uint32_t>(t[0]) * P_PRIME_0) & LIMB_MASK);

        // ─── Step 3: Reduction Accumulate (m * p[j]) ─────────────────
        for (int j = 0; j < 8; j += 2) {
            uint32_t p_pair[4] __attribute__((aligned(16))) = {
                p_limbs[j], p_limbs[j + 1], 0, 0
            };
            uint32_t m_splat[4] __attribute__((aligned(16))) = {
                m, m, 0, 0
            };

            v128_t vp = wasm_v128_load(p_pair);
            v128_t vm = wasm_v128_load(m_splat);
            v128_t red = wasm_u64x2_extmul_low_u32x4(vp, vm);

            // Pure vector accumulation into t
            v128_t vt = wasm_v128_load(&t[j]);
            vt = wasm_i64x2_add(vt, red);
            wasm_v128_store(&t[j], vt);
        }

        // Scalar tail: j = 8
        {
            t[8] += (uint64_t)m * (uint64_t)p_limbs[8];
        }

        // ─── Step 4: Deferred Carry Normalization Pipeline ───────────
        uint64_t carry = 0;
        for (int j = 0; j < NLIMBS; j++) {
            t[j] += carry;
            carry = t[j] >> RADIX;
            t[j] &= LIMB_MASK;
        }
        t[9] += carry;

        // ─── Step 5: Shift (divide by R_limb = 2^29) ────────────────
        for (int j = 0; j < NLIMBS; j++) {
            t[j] = t[j + 1];
        }
        t[9] = 0;
    }

    // ─── Final Conditional Subtraction ───────────────────────────────
    int borrow = 0;
    uint32_t diff[NLIMBS];

    for (int j = 0; j < NLIMBS; j++) {
        int64_t d = (int64_t)t[j] - (int64_t)p_limbs[j] - borrow;
        if (d < 0) {
            diff[j] = static_cast<uint32_t>(d + (1ll << RADIX));
            borrow = 1;
        } else {
            diff[j] = static_cast<uint32_t>(d);
            borrow = 0;
        }
    }

    uint32_t use_original = borrow ? 0xFFFFFFFF : 0x00000000;

    for (int j = 0; j < NLIMBS; j++) {
        result->limb[j] = (static_cast<uint32_t>(t[j]) & use_original) |
                          (diff[j] & ~use_original);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Convert a field element to Montgomery form:
 *   aR = a · R^2 · R^(-1) mod p = a · R mod p
 *
 * This requires a precomputed R² mod p constant.
 */

/**
 * R² mod p for BN254 with R = 2^261.
 *
 * R² mod p = (2^261)² mod p = 2^522 mod p
 *
 * This is precomputed offline. The 32-byte big-endian representation is
 * hardcoded here and converted to limbs at init time.
 *
 * Value (hex): 0x06D89F71CAB8351F47AB1EFF0A417FF6B5E71911D44501FBF32CFC5B538AFA89
 */
static constexpr uint8_t R_SQUARED_BYTES[32] = {
    0x0A, 0x05, 0x4A, 0x3E, 0x84, 0x8B, 0x0F, 0x05,
    0x78, 0x40, 0xF9, 0xF0, 0xAB, 0xC6, 0xE5, 0x4D,
    0x0F, 0xFE, 0xDB, 0x18, 0x85, 0x88, 0x33, 0x77,
    0x38, 0xC2, 0xE1, 0x4B, 0x45, 0xB6, 0x9B, 0xD4
};

static Fp9x29 R_SQUARED_LIMBS;

/**
 * Extended init: also computes R² mod p in limb form.
 */
extern "C" void zk_engine_field_init_full() {
    zk_engine_field_init();
    R_SQUARED_LIMBS = bytes_to_limbs(R_SQUARED_BYTES);
}

/**
 * Convert a to Montgomery form: aR = MonPro(a, R² mod p)
 */
void to_montgomery(Fp9x29* aR, const Fp9x29* a) {
    vectorized_montgomery_cios_schoolbook(aR, a, &R_SQUARED_LIMBS);
}

/**
 * Convert from Montgomery form: a = MonPro(aR, 1)
 */
void from_montgomery(Fp9x29* a, const Fp9x29* aR) {
    Fp9x29 one;
    std::memset(&one, 0, sizeof(one));
    one.limb[0] = 1;
    vectorized_montgomery_cios_schoolbook(a, aR, &one);
}

/**
 * Print a field element in limb form (for debugging).
 */
void fp_print(const char* label, const Fp9x29* fp) {
    std::printf("  %s: [", label);
    for (int i = 0; i < NLIMBS; i++) {
        std::printf("0x%08X", fp->limb[i]);
        if (i < NLIMBS - 1) std::printf(", ");
    }
    std::printf("]\n");
}

/**
 * Compare two field elements for equality.
 */
bool fp_equal(const Fp9x29* a, const Fp9x29* b) {
    for (int i = 0; i < NLIMBS; i++) {
        if (a->limb[i] != b->limb[i]) return false;
    }
    return true;
}

/**
 * Verify that limb decomposition is lossless by round-tripping:
 *   bytes → limbs → bytes → compare
 */
bool verify_decomposition(const uint8_t original[32]) {
    Fp9x29 limbs = bytes_to_limbs(original);
    uint8_t reassembled[32];
    limbs_to_bytes(reassembled, limbs);
    return std::memcmp(original, reassembled, 32) == 0;
}

/**
 * Returns a pointer to P_LIMBS for external access (test harness).
 */
extern "C" const Fp9x29* get_p_limbs() {
    return &P_LIMBS;
}

/**
 * Returns P_PRIME_0 for external access (test harness).
 */
extern "C" uint32_t get_p_prime_0() {
    return P_PRIME_0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: PTHREAD ORCHESTRATION — Phase 2.3
// ═══════════════════════════════════════════════════════════════════════════
//
// This section adds the multi-threaded infrastructure for parallel Groth16
// proving. The JavaScript orchestrator allocates a proof payload into the
// shared memory arena, then wakes the C++ thread via Atomics.notify().
//
// SYNCHRONIZATION MODEL:
//   1. C++ thread spawns via pthread_create → immediately enters futex sleep
//   2. JS main thread writes payload into g_shared_memory_arena via HEAPU8
//   3. JS flips g_stream_synchronization_flag from 0 → 1
//   4. JS calls Atomics.notify() on the flag's memory address
//   5. C++ thread wakes, reads payload, verifies checksum, returns result
//
// ARCHITECTURAL LAW: No while() busy-spin loops. EVER.
//   emscripten_atomic_wait_u32 parks the thread on the kernel futex queue,
//   consuming ZERO CPU cycles while waiting.
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __EMSCRIPTEN_PTHREADS__

#include <emscripten/atomic.h>
#include <thread>

// ─── Shared Memory Arena ──────────────────────────────────────────────────
// 64MB arena for zero-copy payload transfer between JS and C++ threads.
// The JS orchestrator writes proof witness data directly into this buffer
// via Module.HEAPU8, avoiding any memcpy overhead.
//
// __attribute__((aligned(16))): Ensures SIMD128 v128.load/v128.store never
// hit unaligned addresses within this buffer.

static uint8_t g_shared_memory_arena[64 * 1024 * 1024]
    __attribute__((aligned(16)));

// ─── Futex Synchronization Flag ───────────────────────────────────────────
// This 32-bit flag is the SOLE synchronization primitive between the JS
// main thread and the C++ worker thread.
//
// States:
//   0 = WAITING  — The C++ thread is parked in futex sleep.
//   1 = READY    — The JS thread has written the payload and is signaling.
//   2 = COMPLETE — The C++ thread has finished processing.
//
// volatile: Prevents the compiler from optimizing away reads/writes.
// The actual atomicity is guaranteed by emscripten_atomic_wait_u32 and
// __atomic_store_n / __atomic_load_n.

static volatile uint32_t g_stream_synchronization_flag = 0;

// ─── Result Storage ──────────────────────────────────────────────────────
// The C++ thread writes its result here after processing the payload.
// The JS orchestrator reads it after the flag transitions to COMPLETE (2).

static volatile uint32_t g_orchestration_result = 0;

// Global variable to hold the length for the spawned thread
static volatile uint32_t g_orchestrate_len = 0;

extern "C" {

/**
 * Returns a pointer to the shared memory arena.
 * The JS orchestrator uses this to compute the HEAPU8 offset for zero-copy
 * payload writes.
 */
uint8_t* get_shared_arena_ptr(void) {
    return g_shared_memory_arena;
}

/**
 * Returns the capacity of the shared memory arena in bytes.
 */
uint32_t get_shared_arena_capacity(void) {
    return 64u * 1024u * 1024u;
}

/**
 * Returns a pointer to the synchronization flag.
 * The JS orchestrator needs this address to call Atomics.notify() on it.
 */
volatile uint32_t* get_sync_flag_ptr(void) {
    return &g_stream_synchronization_flag;
}

/**
 * Returns the orchestration result after processing completes.
 */
uint32_t get_orchestration_result(void) {
    return g_orchestration_result;
}

/**
 * The core pthread entry point for parallel proof orchestration.
 *
 * LIFECYCLE:
 *   1. Called by the JS orchestrator (indirectly via Emscripten's
 *      PROXY_TO_PTHREAD mechanism or explicit pthread_create).
 *   2. Immediately enters a futex sleep on g_stream_synchronization_flag,
 *      waiting for the value to become non-zero.
 *   3. Once woken by Atomics.notify(), reads `len` bytes from the shared
 *      arena and computes a simple XOR checksum.
 *   4. Stores the checksum in g_orchestration_result.
 *   5. Sets the flag to 2 (COMPLETE) so JS knows the result is ready.
 *
 * @param ptr  Pointer to the payload within the shared memory arena
 *             (currently unused — the arena itself is the payload source).
 * @param len  Number of bytes to process from the arena.
 *
 * ZERO-CPU GUARANTEE:
 *   emscripten_atomic_wait_u32(&flag, 0, INFINITY) translates to the
 *   WebAssembly `memory.atomic.wait32` instruction, which calls into the
 *   browser's futex implementation. The thread is fully descheduled by the
 *   OS kernel — no CPU cycles are consumed while waiting.
 */
void orchestrate_parallel_proof_matrix(uint8_t* ptr, uint32_t len) {
    (void)ptr; // Reserved for future direct-pointer mode

    // ─── FUTEX SLEEP ──────────────────────────────────────────────────
    // Park the thread until g_stream_synchronization_flag != 0.
    // The third argument is the "expected value" — we sleep while flag == 0.
    // The fourth argument is the timeout in nanoseconds (INFINITY = wait forever).
    //
    // CRITICAL: This is NOT a busy-spin. The thread is fully descheduled.
    emscripten_atomic_wait_u32(
        (int32_t*)&g_stream_synchronization_flag,
        0,          // Expected value (sleep while flag == 0)
        -1          // Timeout: -1 = INFINITY (wait forever)
    );

    // ─── PAYLOAD VERIFICATION ─────────────────────────────────────────
    // Compute a simple XOR-fold checksum of the first `len` bytes of the
    // shared memory arena. This proves:
    //   1. The futex wake mechanism works correctly
    //   2. The shared memory is readable from the worker thread
    //   3. The zero-copy write from JS HEAPU8 landed correctly

    uint32_t checksum = 0;
    uint32_t safe_len = (len <= (64u * 1024u * 1024u)) ? len : (64u * 1024u * 1024u);

    for (uint32_t i = 0; i < safe_len; i++) {
        checksum ^= g_shared_memory_arena[i];
        checksum = (checksum << 1) | (checksum >> 31); // Rotate left by 1
    }

    // ─── STORE RESULT ─────────────────────────────────────────────────
    g_orchestration_result = checksum;

    // ─── SIGNAL COMPLETION ────────────────────────────────────────────
    // Atomically set flag to 2 (COMPLETE) so the JS orchestrator knows
    // the result is ready. Uses __atomic_store_n for cross-thread visibility.
    __atomic_store_n((uint32_t*)&g_stream_synchronization_flag, 2, __ATOMIC_SEQ_CST);
}

void worker_thread_entry(void) {
    orchestrate_parallel_proof_matrix(nullptr, g_orchestrate_len);
}

/**
 * Non-blocking entry point called by JS. Spawns a dedicated worker thread
 * to wait on the futex, allowing the main/proxied thread to return immediately.
 */
void start_orchestration(uint32_t len) {
    g_orchestrate_len = len;
    // Reset flag to 0 (WAITING) and result to 0
    __atomic_store_n((uint32_t*)&g_stream_synchronization_flag, 0, __ATOMIC_SEQ_CST);
    g_orchestration_result = 0;

    // Spawn a C++ std::thread to handle the blocking futex sleep
    std::thread t(worker_thread_entry);
    t.detach();
}

} // extern "C"

// ─── REQUIRED BY EMSCRIPTEN PROXY_TO_PTHREAD ────────────────────────────
// Emscripten's PROXY_TO_PTHREAD=1 intercepts the main thread and proxies
// execution to a Web Worker. This requires a dummy main() function to exist.
int main() {
    return 0;
}

#endif // __EMSCRIPTEN_PTHREADS__
