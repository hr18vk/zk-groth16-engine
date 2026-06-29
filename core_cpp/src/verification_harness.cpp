/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ZK Engine Phase 2.1 + 2.2
 * SIMD128 Verification Matrix & Memory Arena Harness
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This is a STANDALONE test harness (with main()) that proves:
 *   1. The bump allocator correctly aligns all pointers to 16-byte boundaries
 *   2. WASM SIMD128 i64x2.extmul_low_i32x4_u produces correct widened products
 *   3. No overflow truncation occurs during 32-bit → 64-bit limb multiplication
 *
 * This file does NOT export any functions to JavaScript. It prints to stdout
 * and is executed via Node.js after Emscripten compilation.
 *
 * Compilation: em++ -std=c++17 -Oz -msimd128 verification_harness.cpp -o harness.js
 * Execution:   node harness.js
 *
 * ARCHITECTURAL LAWS:
 *   - NO malloc(), NO new, NO dynamic allocation
 *   - ALL allocations from static bump arena with 16-byte alignment mask
 *   - SIMD loads MUST hit 16-byte aligned addresses (ARM SIGBUS prevention)
 * ═══════════════════════════════════════════════════════════════════════════
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <wasm_simd128.h>

// ZK Engine Phase 2.2 — Montgomery CIOS
#include "vector_prover.cpp"

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: STATIC BUMP ALLOCATOR (Zero-Overhead Memory Arena)
// ═══════════════════════════════════════════════════════════════════════════
//
// The arena is a flat 64MB buffer. Allocations bump a monotonic offset pointer.
// There is no free(). To "reset," set g_arena_offset = 0.
//
// Every allocation is 16-byte aligned via the bitwise mask:
//   offset = (offset + 15) & ~15
//
// This guarantees that SIMD128 v128.load/v128.store never hit an unaligned
// address, which would cause SIGBUS on ARM or undefined behavior on x86
// (split cache-line penalty even if it doesn't fault).
// ═══════════════════════════════════════════════════════════════════════════

static constexpr uint32_t ARENA_SIZE = 64 * 1024 * 1024; // 64 MB
static uint8_t g_arena[ARENA_SIZE] __attribute__((aligned(64)));
static uint32_t g_arena_offset = 0;

/**
 * Aligns the current offset to a 16-byte boundary.
 *
 * Mathematical proof:
 *   ~15 in binary = ...11110000 (masks off the low 4 bits)
 *   (offset + 15) rounds up to the next multiple of 16
 *   & ~15 clears the low 4 bits, snapping to the 16-byte boundary
 *
 * Examples:
 *   align16(0)  = 0   (already aligned)
 *   align16(1)  = 16  (bumped to next boundary)
 *   align16(15) = 16  (bumped to next boundary)
 *   align16(16) = 16  (already aligned)
 *   align16(17) = 32  (bumped to next boundary)
 */
static inline uint32_t align16(uint32_t offset) {
    return (offset + 15u) & ~15u;
}

/**
 * Allocates `size` bytes from the bump arena.
 * Returns a pointer to the allocated region, or nullptr if arena is exhausted.
 *
 * The returned pointer is ALWAYS 16-byte aligned.
 * This function NEVER calls malloc(), new, or any libc allocator.
 */
static uint8_t* arena_alloc(uint32_t size) {
    // Step 1: Align the current offset to 16-byte boundary BEFORE allocating
    uint32_t aligned_offset = align16(g_arena_offset);

    // Step 2: Bounds check
    if (aligned_offset + size > ARENA_SIZE) {
        std::printf("  [FATAL] Arena exhausted: requested %u bytes at offset %u, capacity %u\n",
                    size, aligned_offset, ARENA_SIZE);
        return nullptr;
    }

    // Step 3: Get the pointer
    uint8_t* ptr = g_arena + aligned_offset;

    // Step 4: Bump the offset past the allocated region
    g_arena_offset = aligned_offset + size;

    return ptr;
}

/**
 * Returns the total number of bytes consumed (including alignment padding).
 */
static uint32_t arena_used(void) {
    return g_arena_offset;
}

/**
 * Resets the arena. All previous allocations become invalid.
 */
static void arena_reset(void) {
    g_arena_offset = 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: SIMD128 EXTENDED MULTIPLICATION TEST
// ═══════════════════════════════════════════════════════════════════════════
//
// The BN254 scalar field prime is:
//   p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
//
// This is a 254-bit number. We represent it as 4 × 64-bit limbs (Montgomery form).
// Each 64-bit limb is stored as 2 × 32-bit halves in little-endian order.
//
// To multiply two limbs, we need the full 64-bit product of two 32-bit values.
// A naive i32x4.mul truncates to 32 bits. We MUST use:
//   i64x2.extmul_low_i32x4_u
// which widens the low 2 lanes of each i32x4 operand to i64x2, then multiplies.
//
// Test: 0xFFFFFFFF × 2 = 0x1FFFFFFFE (exceeds 32-bit range)
// If the result is 0xFFFFFFFE (truncated), SIMD widening is broken.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Executes the SIMD128 i64x2.extmul_low_i32x4_u instruction on test vectors
 * and writes the 2 × 64-bit results into the provided output array.
 *
 * @param result_out  Pointer to a 16-byte aligned buffer for 2 × uint64_t results
 * @param a           Pointer to a 16-byte aligned buffer containing 4 × uint32_t
 * @param b           Pointer to a 16-byte aligned buffer containing 4 × uint32_t
 */
static void test_simd_extmul(uint64_t* result_out,
                              const uint32_t* a,
                              const uint32_t* b) {
    // Load 4 × 32-bit unsigned integers from aligned memory
    v128_t va = wasm_v128_load(a);
    v128_t vb = wasm_v128_load(b);

    // Extended multiply: widen low 2 lanes of each operand to 64-bit, then multiply
    // Input:  va = [a0, a1, a2, a3] (u32)
    //         vb = [b0, b1, b2, b3] (u32)
    // Output: result = [(u64)a0 * (u64)b0, (u64)a1 * (u64)b1]
    v128_t result = wasm_u64x2_extmul_low_u32x4(va, vb);

    // Store result to aligned memory
    wasm_v128_store(result_out, result);
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: MAIN — Verification Harness Entry Point
// ═══════════════════════════════════════════════════════════════════════════

int main() {
    int exit_code = 0;

    std::printf("═══════════════════════════════════════════════════════════\n");
    std::printf("  ZK Engine Phase 2.1 — Verification Harness\n");
    std::printf("  SIMD128 Memory Alignment & Extended Multiplication\n");
    std::printf("═══════════════════════════════════════════════════════════\n\n");

    // ─── TEST 1: Arena Initialization ──────────────────────────────────
    std::printf("[TEST 1] Arena Initialization\n");
    arena_reset();
    std::printf("  Arena initialized: %u bytes (%u MB)\n", ARENA_SIZE, ARENA_SIZE / (1024 * 1024));
    std::printf("  Arena base address: %p\n", static_cast<void*>(g_arena));

    // Verify base address is 64-byte aligned (it should be due to __attribute__)
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(g_arena);
    if (base_addr % 64 == 0) {
        std::printf("  Base alignment: 64-byte ALIGNED ✓\n\n");
    } else {
        std::printf("  Base alignment: UNALIGNED ✗ (address %% 64 = %u)\n\n",
                    static_cast<unsigned>(base_addr % 64));
        exit_code = 1;
    }

    // ─── TEST 2: Unaligned Payload Ingestion ──────────────────────────
    // Simulate 3 payloads with deliberately awkward sizes to stress the
    // alignment mask. The allocator MUST round up to 16-byte boundaries.
    std::printf("[TEST 2] Bump Allocator — 3 Unaligned Payload Allocations\n");

    // Payload 1: 37 bytes (not a multiple of 16)
    uint8_t* p1 = arena_alloc(37);
    uintptr_t addr1 = reinterpret_cast<uintptr_t>(p1);
    bool a1_ok = (addr1 % 16 == 0);
    std::printf("  Alloc 1: %u bytes → address %p → %% 16 = %u → %s\n",
                37u, static_cast<void*>(p1),
                static_cast<unsigned>(addr1 % 16),
                a1_ok ? "ALIGNED ✓" : "UNALIGNED ✗");
    if (!a1_ok) exit_code = 1;

    // Write a pattern to verify no overlap
    std::memset(p1, 0xAA, 37);

    // Payload 2: 100 bytes (not a multiple of 16)
    uint8_t* p2 = arena_alloc(100);
    uintptr_t addr2 = reinterpret_cast<uintptr_t>(p2);
    bool a2_ok = (addr2 % 16 == 0);
    std::printf("  Alloc 2: %u bytes → address %p → %% 16 = %u → %s\n",
                100u, static_cast<void*>(p2),
                static_cast<unsigned>(addr2 % 16),
                a2_ok ? "ALIGNED ✓" : "UNALIGNED ✗");
    if (!a2_ok) exit_code = 1;

    std::memset(p2, 0xBB, 100);

    // Payload 3: 1 byte (worst case — must still align next alloc to 16)
    uint8_t* p3 = arena_alloc(1);
    uintptr_t addr3 = reinterpret_cast<uintptr_t>(p3);
    bool a3_ok = (addr3 % 16 == 0);
    std::printf("  Alloc 3: %u bytes → address %p → %% 16 = %u → %s\n",
                1u, static_cast<void*>(p3),
                static_cast<unsigned>(addr3 % 16),
                a3_ok ? "ALIGNED ✓" : "UNALIGNED ✗");
    if (!a3_ok) exit_code = 1;

    *p3 = 0xCC;

    // Verify monotonic ordering (no overlap)
    bool no_overlap = (addr2 > addr1) && (addr3 > addr2);
    std::printf("  Monotonic ordering: addr1=%p < addr2=%p < addr3=%p → %s\n",
                static_cast<void*>(p1), static_cast<void*>(p2), static_cast<void*>(p3),
                no_overlap ? "NO OVERLAP ✓" : "OVERLAP DETECTED ✗");
    if (!no_overlap) exit_code = 1;

    // Verify integrity of earlier allocations (no overwrite)
    bool integrity_ok = (p1[0] == 0xAA) && (p1[36] == 0xAA) &&
                        (p2[0] == 0xBB) && (p2[99] == 0xBB) &&
                        (*p3 == 0xCC);
    std::printf("  Memory integrity: %s\n", integrity_ok ? "INTACT ✓" : "CORRUPTED ✗");
    if (!integrity_ok) exit_code = 1;

    std::printf("  Total arena used: %u bytes (including %u bytes alignment padding)\n\n",
                arena_used(),
                arena_used() - (37 + 100 + 1));

    // ─── TEST 3: SIMD128 Extended Multiplication ──────────────────────
    std::printf("[TEST 3] SIMD128 i64x2.extmul_low_i32x4_u\n");

    // Allocate SIMD-aligned buffers from the arena (NOT from stack)
    uint32_t* simd_a = reinterpret_cast<uint32_t*>(arena_alloc(16));
    uint32_t* simd_b = reinterpret_cast<uint32_t*>(arena_alloc(16));
    uint64_t* simd_result = reinterpret_cast<uint64_t*>(arena_alloc(16));

    if (!simd_a || !simd_b || !simd_result) {
        std::printf("  [FATAL] Arena allocation failed for SIMD buffers\n");
        return 1;
    }

    // Verify SIMD buffer alignment
    std::printf("  SIMD buffer A:      %p → %% 16 = %u\n",
                static_cast<void*>(simd_a),
                static_cast<unsigned>(reinterpret_cast<uintptr_t>(simd_a) % 16));
    std::printf("  SIMD buffer B:      %p → %% 16 = %u\n",
                static_cast<void*>(simd_b),
                static_cast<unsigned>(reinterpret_cast<uintptr_t>(simd_b) % 16));
    std::printf("  SIMD buffer result: %p → %% 16 = %u\n",
                static_cast<void*>(simd_result),
                static_cast<unsigned>(reinterpret_cast<uintptr_t>(simd_result) % 16));

    // ─── Test Vector 1: Overflow Boundary ────────────────────────────
    // a[0] = 0xFFFFFFFF (max u32), b[0] = 2
    // Expected: (u64)0xFFFFFFFF * (u64)2 = 8589934590 = 0x1FFFFFFFE
    // If truncated to u32: 0xFFFFFFFE = 4294967294 (WRONG)
    simd_a[0] = 0xFFFFFFFF;
    simd_a[1] = 0x12345678;
    simd_a[2] = 0;
    simd_a[3] = 0;

    simd_b[0] = 2;
    simd_b[1] = 3;
    simd_b[2] = 0;
    simd_b[3] = 0;

    std::printf("\n  Test vector:\n");
    std::printf("    a = [0x%08X, 0x%08X, 0, 0]\n", simd_a[0], simd_a[1]);
    std::printf("    b = [%u, %u, 0, 0]\n", simd_b[0], simd_b[1]);

    // Execute SIMD
    test_simd_extmul(simd_result, simd_a, simd_b);

    // Expected results
    uint64_t expected_0 = (uint64_t)0xFFFFFFFFu * 2u;  // = 8589934590
    uint64_t expected_1 = (uint64_t)0x12345678u * 3u;   // = 920657784

    std::printf("\n  Results:\n");
    std::printf("    SIMD extmul result[0] = %llu (expected %llu) → %s\n",
                static_cast<unsigned long long>(simd_result[0]),
                static_cast<unsigned long long>(expected_0),
                simd_result[0] == expected_0 ? "CORRECT ✓" : "WRONG ✗");

    std::printf("    SIMD extmul result[1] = %llu (expected %llu) → %s\n",
                static_cast<unsigned long long>(simd_result[1]),
                static_cast<unsigned long long>(expected_1),
                simd_result[1] == expected_1 ? "CORRECT ✓" : "WRONG ✗");

    if (simd_result[0] != expected_0 || simd_result[1] != expected_1) {
        exit_code = 1;
    }

    // ─── Test Vector 2: Zero Multiplication ──────────────────────────
    // Verifies that SIMD doesn't produce garbage on zero inputs
    simd_a[0] = 0;
    simd_a[1] = 0;
    simd_b[0] = 0xFFFFFFFF;
    simd_b[1] = 0xFFFFFFFF;

    test_simd_extmul(simd_result, simd_a, simd_b);

    bool zero_ok = (simd_result[0] == 0) && (simd_result[1] == 0);
    std::printf("    Zero multiplication:   result = [%llu, %llu] → %s\n",
                static_cast<unsigned long long>(simd_result[0]),
                static_cast<unsigned long long>(simd_result[1]),
                zero_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!zero_ok) exit_code = 1;

    // ─── Test Vector 3: Identity Multiplication ──────────────────────
    // a × 1 = a (widened to 64-bit)
    simd_a[0] = 0xDEADBEEF;
    simd_a[1] = 0xCAFEBABE;
    simd_b[0] = 1;
    simd_b[1] = 1;

    test_simd_extmul(simd_result, simd_a, simd_b);

    bool identity_ok = (simd_result[0] == 0xDEADBEEFu) && (simd_result[1] == 0xCAFEBABEu);
    std::printf("    Identity multiplication: result = [0x%llX, 0x%llX] → %s\n",
                static_cast<unsigned long long>(simd_result[0]),
                static_cast<unsigned long long>(simd_result[1]),
                identity_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!identity_ok) exit_code = 1;

    // ─── TEST 4: 256-bit Montgomery CIOS Reduction ───────────────────
    std::printf("\n[TEST 4] 256-bit Montgomery CIOS — 9×29-bit Reduced-Radix\n");

    // Initialize field constants (P_LIMBS, P_PRIME_0)
    zk_engine_field_init_full();

    // 4a: Verify prime decomposition round-trip
    bool p_roundtrip = verify_decomposition(P_BYTES);
    std::printf("  Prime decomposition round-trip: %s\n",
                p_roundtrip ? "LOSSLESS ✓" : "LOSSY ✗");
    if (!p_roundtrip) exit_code = 1;

    // Print the prime in limb form
    fp_print("P_LIMBS", get_p_limbs());
    std::printf("  P_PRIME_0 = 0x%08X\n", get_p_prime_0());

    // 4b: Verify P_PRIME_0 correctness
    // Check: (p[0] * p'_0) mod 2^29 should equal 2^29 - 1 (i.e., -1 mod 2^29)
    // Because p'_0 = -p^(-1) mod 2^29, so p[0] * p'_0 = -1 mod 2^29
    {
        uint64_t check = ((uint64_t)get_p_limbs()->limb[0] * (uint64_t)get_p_prime_0()) & LIMB_MASK;
        bool pp0_ok = (check == LIMB_MASK);
        std::printf("  P_PRIME_0 verification: p[0]*p'_0 mod 2^29 = 0x%08X (expected 0x%08X) → %s\n",
                    (uint32_t)check, LIMB_MASK, pp0_ok ? "CORRECT ✓" : "WRONG ✗");
        if (!pp0_ok) exit_code = 1;
    }

    // 4c: Test vector — Montgomery multiplication of 1 * 1
    // In Montgomery form: 1_mont = R mod p
    // MonPro(1_mont, 1_mont) should equal 1_mont (since R * R * R^(-1) = R mod p)
    // But simpler: MonPro(a, 1) = a * R^(-1) mod p
    // So MonPro(R mod p, 1) = 1
    //
    // We test: to_montgomery(3) → 3R, to_montgomery(7) → 7R
    //          MonPro(3R, 7R) = 3*7*R = 21R mod p
    //          from_montgomery(21R) = 21

    // Construct a = 3
    Fp9x29 a_raw;
    std::memset(&a_raw, 0, sizeof(a_raw));
    a_raw.limb[0] = 3;

    // Construct b = 7
    Fp9x29 b_raw;
    std::memset(&b_raw, 0, sizeof(b_raw));
    b_raw.limb[0] = 7;

    // Convert to Montgomery form
    Fp9x29 a_mont, b_mont;
    to_montgomery(&a_mont, &a_raw);
    to_montgomery(&b_mont, &b_raw);

    fp_print("a=3 (mont)", &a_mont);
    fp_print("b=7 (mont)", &b_mont);

    // Multiply in Montgomery domain
    Fp9x29 product_mont;
    vectorized_montgomery_cios_schoolbook(&product_mont, &a_mont, &b_mont);

    fp_print("a*b (mont)", &product_mont);

    // Convert back from Montgomery form
    Fp9x29 product_normal;
    from_montgomery(&product_normal, &product_mont);

    fp_print("a*b (normal)", &product_normal);

    // Expected: 3 * 7 = 21
    Fp9x29 expected_21;
    std::memset(&expected_21, 0, sizeof(expected_21));
    expected_21.limb[0] = 21;

    bool mul_ok = fp_equal(&product_normal, &expected_21);
    std::printf("  Montgomery multiplication 3*7 = 21: %s\n",
                mul_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!mul_ok) {
        std::printf("  Expected limb[0] = 21, got limb[0] = %u\n",
                    product_normal.limb[0]);
        exit_code = 1;
    }

    // 4d: Test with larger values near the field boundary
    // a = p - 1, b = 2
    // Expected: (p-1) * 2 mod p = 2p - 2 mod p = p - 2
    Fp9x29 pm1;
    pm1 = *get_p_limbs();
    // Subtract 1 from limb[0]
    pm1.limb[0] -= 1;

    Fp9x29 two;
    std::memset(&two, 0, sizeof(two));
    two.limb[0] = 2;

    Fp9x29 pm1_mont, two_mont;
    to_montgomery(&pm1_mont, &pm1);
    to_montgomery(&two_mont, &two);

    Fp9x29 boundary_product_mont;
    vectorized_montgomery_cios_schoolbook(&boundary_product_mont, &pm1_mont, &two_mont);

    Fp9x29 boundary_product;
    from_montgomery(&boundary_product, &boundary_product_mont);

    // Expected: p - 2
    Fp9x29 expected_pm2;
    expected_pm2 = *get_p_limbs();
    expected_pm2.limb[0] -= 2;

    bool boundary_ok = fp_equal(&boundary_product, &expected_pm2);
    std::printf("  Boundary multiplication (p-1)*2 mod p = p-2: %s\n",
                boundary_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!boundary_ok) {
        fp_print("Expected (p-2)", &expected_pm2);
        fp_print("Got", &boundary_product);
        exit_code = 1;
    }

    // 4e: Test multiplicative identity: a * 1 = a
    Fp9x29 one_raw;
    std::memset(&one_raw, 0, sizeof(one_raw));
    one_raw.limb[0] = 1;

    Fp9x29 test_val;
    std::memset(&test_val, 0, sizeof(test_val));
    test_val.limb[0] = 42;
    test_val.limb[1] = 1337;

    Fp9x29 test_mont, one_mont;
    to_montgomery(&test_mont, &test_val);
    to_montgomery(&one_mont, &one_raw);

    Fp9x29 identity_product_mont;
    vectorized_montgomery_cios_schoolbook(&identity_product_mont, &test_mont, &one_mont);

    Fp9x29 identity_result;
    from_montgomery(&identity_result, &identity_product_mont);

    bool identity_mul_ok = fp_equal(&identity_result, &test_val);
    std::printf("  Multiplicative identity: a*1 = a: %s\n",
                identity_mul_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!identity_mul_ok) {
        fp_print("Expected", &test_val);
        fp_print("Got", &identity_result);
        exit_code = 1;
    }

    // 4f: Test squaring: a² via MonPro(a, a)
    Fp9x29 five;
    std::memset(&five, 0, sizeof(five));
    five.limb[0] = 5;

    Fp9x29 five_mont;
    to_montgomery(&five_mont, &five);

    Fp9x29 square_mont;
    vectorized_montgomery_cios_schoolbook(&square_mont, &five_mont, &five_mont);

    Fp9x29 square_result;
    from_montgomery(&square_result, &square_mont);

    Fp9x29 expected_25;
    std::memset(&expected_25, 0, sizeof(expected_25));
    expected_25.limb[0] = 25;

    bool square_ok = fp_equal(&square_result, &expected_25);
    std::printf("  Squaring: 5² = 25: %s\n",
                square_ok ? "CORRECT ✓" : "WRONG ✗");
    if (!square_ok) {
        fp_print("Expected 25", &expected_25);
        fp_print("Got", &square_result);
        exit_code = 1;
    }

    // ─── Final Summary ───────────────────────────────────────────────
    std::printf("\n═══════════════════════════════════════════════════════════\n");
    std::printf("  Total arena consumed: %u bytes\n", arena_used());
    std::printf("  Total alignment padding: %u bytes\n",
                arena_used() - (37 + 100 + 1 + 16 + 16 + 16) /* Note: TEST 4 uses no arena */);

    if (exit_code == 0) {
        std::printf("\n  ✅ ALL VERIFICATION CHECKS PASSED\n");
    } else {
        std::printf("\n  ❌ VERIFICATION FAILED — See errors above\n");
    }
    std::printf("═══════════════════════════════════════════════════════════\n");

    return exit_code;
}
