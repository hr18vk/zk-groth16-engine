// ===========================================================================
// proof_assembly.cpp — ZK Engine Phase 3.4: Groth16 Proof Assembly
// ===========================================================================
// ARCHITECTURE: This file is #include'd by prover_engine.cpp as an inline
// STU extension. It has direct access to all prior definitions:
//
//   From vector_prover.cpp:
//     Fp9x29, NLIMBS (9), RADIX (29), LIMB_MASK (0x1FFFFFFF)
//     P_BYTES[32], P_LIMBS, P_PRIME_0
//     bytes_to_limbs(), limbs_to_bytes(), to_montgomery(), from_montgomery()
//     vectorized_montgomery_cios_schoolbook(), zk_engine_field_init()
//     fp_equal(), fp_print()
//
//   From prover_engine.cpp:
//     FieldElement (alignas(32), sizeof=64), JacobianPoint (alignas(64), sizeof=256)
//     fp_add(), fp_sub(), fp_mul(), fp_sqr()
//     jacobian_set_identity(), jacobian_is_identity()
//     jacobian_double(), jacobian_mixed_add()
//     ntt_bailey_4step()
//     g_proof_buffer[4096], g_zkey_buffer[64MB]
//
// DO NOT #include "vector_prover.cpp" — it is already included by prover_engine.cpp.
// ===========================================================================

#include <wasm_simd128.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: Fp2 Extension Field  ( F_p[u] / (u² + 1) )
// ═══════════════════════════════════════════════════════════════════════════

struct alignas(64) Fp2Element {
    Fp9x29 c0;         // Real component:      36 bytes at offset 0
    Fp9x29 c1;         // Imaginary component:  36 bytes at offset 36
    uint8_t _pad[56];  // Pad to 128 bytes (next multiple of alignof=64)
};
static_assert(sizeof(Fp2Element) == 128, "Fp2Element must be exactly 128 bytes");
static_assert(alignof(Fp2Element) == 64, "Fp2Element must be 64-byte aligned");

struct alignas(64) JacobianPointG2 {
    Fp2Element X;       // 128 bytes at offset 0
    Fp2Element Y;       // 128 bytes at offset 128
    Fp2Element Z;       // 128 bytes at offset 256
};
static_assert(sizeof(JacobianPointG2) == 384, "JacobianPointG2 must be exactly 384 bytes");
static_assert(alignof(JacobianPointG2) == 64, "JacobianPointG2 must be 64-byte aligned");

// Zero element (valid in both normal and Montgomery form: 0·R = 0)
static const Fp9x29 FP_ZERO = {{0, 0, 0, 0, 0, 0, 0, 0, 0}};

// ─── Fp2 Addition ─────────────────────────────────────────────────────────
static void fp2_add(Fp2Element* r, const Fp2Element* a, const Fp2Element* b) {
    fp_add(&r->c0, &a->c0, &b->c0);
    fp_add(&r->c1, &a->c1, &b->c1);
}

// ─── Fp2 Subtraction ─────────────────────────────────────────────────────
static void fp2_sub(Fp2Element* r, const Fp2Element* a, const Fp2Element* b) {
    fp_sub(&r->c0, &a->c0, &b->c0);
    fp_sub(&r->c1, &a->c1, &b->c1);
}

// ─── Fp2 Negation ─────────────────────────────────────────────────────────
static void fp2_neg(Fp2Element* r, const Fp2Element* a) {
    fp_sub(&r->c0, &FP_ZERO, &a->c0);
    fp_sub(&r->c1, &FP_ZERO, &a->c1);
}

// ─── Fp2 Karatsuba Multiplication ─────────────────────────────────────────
// (a0 + a1·u)(b0 + b1·u) = (a0·b0 − a1·b1) + ((a0+a1)(b0+b1) − a0·b0 − a1·b1)·u
// Cost: 3 fp_mul + 2 fp_add + 3 fp_sub = 3 CIOS SIMD128 multiplies
// Each fp_mul delegates to vectorized_montgomery_cios_schoolbook() which
// uses wasm_u64x2_extmul_low_u32x4 for 32-bit limb widening multiplication.
static void fp2_mul(Fp2Element* r, const Fp2Element* a, const Fp2Element* b) {
    Fp9x29 v0, v1, a_sum, b_sum, cross;

    fp_mul(&v0, &a->c0, &b->c0);       // v0 = a0·b0
    fp_mul(&v1, &a->c1, &b->c1);       // v1 = a1·b1

    fp_add(&a_sum, &a->c0, &a->c1);    // a_sum = a0 + a1
    fp_add(&b_sum, &b->c0, &b->c1);    // b_sum = b0 + b1

    fp_mul(&cross, &a_sum, &b_sum);     // cross = (a0+a1)(b0+b1)

    fp_sub(&r->c1, &cross, &v0);        // c1 = cross − v0
    fp_sub(&r->c1, &r->c1, &v1);        // c1 = cross − v0 − v1  (= a0·b1 + a1·b0)

    fp_sub(&r->c0, &v0, &v1);           // c0 = v0 − v1  (= a0·b0 − a1·b1)
}

// ─── Fp2 Optimized Squaring ───────────────────────────────────────────────
// (a0 + a1·u)² = (a0+a1)(a0−a1) + 2·a0·a1·u
// Cost: 2 fp_mul + 1 fp_add + 1 fp_sub (saves 1 mul vs generic multiply)
static void fp2_sqr(Fp2Element* r, const Fp2Element* a) {
    Fp9x29 sum, diff, cross;

    fp_add(&sum, &a->c0, &a->c1);      // sum  = a0 + a1
    fp_sub(&diff, &a->c0, &a->c1);     // diff = a0 − a1
    fp_mul(&r->c0, &sum, &diff);        // c0 = (a0+a1)(a0−a1) = a0² − a1²

    fp_mul(&cross, &a->c0, &a->c1);    // cross = a0·a1
    fp_add(&r->c1, &cross, &cross);     // c1 = 2·a0·a1
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: Fermat Inversion in Fp
// ═══════════════════════════════════════════════════════════════════════════

// p − 2 in big-endian bytes, computed once from P_BYTES at runtime.
// Adapts automatically to whichever prime vector_prover.cpp uses.
static uint8_t PA_P_MINUS_2[32];
static bool pa_init_done = false;

static void pa_compute_p_minus_2() {
    memcpy(PA_P_MINUS_2, P_BYTES, 32);
    int borrow = 2;
    for (int i = 31; i >= 0 && borrow > 0; i--) {
        int val = static_cast<int>(PA_P_MINUS_2[i]) - borrow;
        if (val < 0) {
            PA_P_MINUS_2[i] = static_cast<uint8_t>(val + 256);
            borrow = 1;
        } else {
            PA_P_MINUS_2[i] = static_cast<uint8_t>(val);
            borrow = 0;
        }
    }
}

static void pa_ensure_init() {
    if (pa_init_done) return;
    pa_compute_p_minus_2();
    pa_init_done = true;
}

// Returns R mod p (Montgomery form of 1)
static Fp9x29 fp_montgomery_one() {
    Fp9x29 one;
    memset(&one, 0, sizeof(Fp9x29));
    one.limb[0] = 1;
    Fp9x29 result;
    to_montgomery(&result, &one);
    return result;
}

// ─── Fermat Inversion: a^(p-2) mod p ─────────────────────────────────────
// Structurally constant-time square-and-multiply over 256 bits.
// Branching depends strictly on the public curve prime P_MINUS_2, ensuring
// the execution trajectory is identical for every operand `a`, protecting
// sensitive witness-derived Jacobian Z coordinates from timing attacks.
// Cost: 256 squarings + ~128 multiplies x 81 CIOS iterations each = ~31K ops.
static void fp_inv(Fp9x29* result, const Fp9x29* a) {
    pa_ensure_init();

    Fp9x29 acc = fp_montgomery_one();  // acc = R (Montgomery 1)

    for (uint32_t i = 0; i < 32; i++) {
        for (int bit = 7; bit >= 0; bit--) {
            fp_sqr(&acc, &acc);
            if ((PA_P_MINUS_2[i] >> bit) & 1) {
                fp_mul(&acc, &acc, a);
            }
        }
    }
    *result = acc;
}

// ─── Repeated Squaring: base^(2^log_n) ───────────────────────────────────
static void fp_exp_pow2(Fp9x29* result, const Fp9x29* base, uint32_t log_n) {
    *result = *base;
    for (uint32_t i = 0; i < log_n; i++) {
        fp_sqr(result, result);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: Fp2 Inversion via Norm
// ═══════════════════════════════════════════════════════════════════════════
// inv(c0 + c1·u) = (c0 − c1·u) / (c0² + c1²)
// Requires only ONE Fermat inversion in Fp (for the norm).

static void fp2_inv(Fp2Element* r, const Fp2Element* a) {
    Fp9x29 c0_sq, c1_sq, norm, inv_norm;

    fp_sqr(&c0_sq, &a->c0);               // c0²
    fp_sqr(&c1_sq, &a->c1);               // c1²
    fp_add(&norm, &c0_sq, &c1_sq);         // N = c0² + c1² (in Fp)

    fp_inv(&inv_norm, &norm);               // N^(p-2) — single Fermat inversion

    fp_mul(&r->c0, &a->c0, &inv_norm);     // c0 / N
    fp_mul(&r->c1, &a->c1, &inv_norm);
    fp_sub(&r->c1, &FP_ZERO, &r->c1);      // −c1 / N
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: Little-Endian Limb Serialization
// ═══════════════════════════════════════════════════════════════════════════
// Converts 9x29-bit Montgomery-reduced limbs to 32 little-endian bytes.
// Paired with serialize_proof_be() which SIMD-swizzles to big-endian.

static void limbs_to_bytes_le(uint8_t* out_le, const Fp9x29& fp) {
    uint64_t acc = 0;
    uint32_t bits = 0;
    uint32_t pos = 0;

    for (uint32_t i = 0; i < NLIMBS && pos < 32; i++) {
        acc |= (static_cast<uint64_t>(fp.limb[i] & LIMB_MASK)) << bits;
        bits += RADIX;
        while (bits >= 8 && pos < 32) {
            out_le[pos++] = static_cast<uint8_t>(acc & 0xFF);
            acc >>= 8;
            bits -= 8;
        }
    }
    while (pos < 32) {
        out_le[pos++] = static_cast<uint8_t>(acc & 0xFF);
        acc >>= 8;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: In-Place Quotient Polynomial H(x)
// ═══════════════════════════════════════════════════════════════════════════
// On the coset {g*w^0, g*w^1, ..., g*w^(n-1)}:
//   Z_H(g*w^i) = (g*w^i)^n - 1 = g^n - 1  (constant, since w^n = 1)
//   T_i = (A_i * B_i - C_i) * (g^n - 1)^(-1)
//
// DESTRUCTIVE: Overwrites a_eval[i].fp with T_i. Zero additional allocation.

static void compute_quotient_inplace(
    FieldElement*        a_eval,     // IN: A on coset, OUT: H on coset
    const FieldElement*  b_eval,     // IN: B evaluated on coset
    const FieldElement*  c_eval,     // IN: C evaluated on coset
    const Fp9x29*        inv_z_h,    // Precomputed (g^n - 1)^(-1) in Montgomery form
    uint32_t             domain_size
) {
    for (uint32_t i = 0; i < domain_size; i++) {
        Fp9x29 ab, ab_minus_c;
        fp_mul(&ab, &a_eval[i].fp, &b_eval[i].fp);          // A_i * B_i
        fp_sub(&ab_minus_c, &ab, &c_eval[i].fp);             // A_i * B_i - C_i
        fp_mul(&a_eval[i].fp, &ab_minus_c, inv_z_h);         // x inv(Z_H) -> overwrite A[i]
    }
}

// Precompute vanishing polynomial inverse on the coset
static void compute_inv_z_h(
    Fp9x29*       inv_z_h,
    const Fp9x29* coset_gen_mont,   // Coset generator g in Montgomery form
    uint32_t      log_domain_size
) {
    Fp9x29 g_to_n;
    fp_exp_pow2(&g_to_n, coset_gen_mont, log_domain_size);  // g^(2^log_n) = g^n

    Fp9x29 mont_one = fp_montgomery_one();
    Fp9x29 z_h;
    fp_sub(&z_h, &g_to_n, &mont_one);                       // Z_H = g^n - 1

    fp_inv(inv_z_h, &z_h);                                   // (g^n - 1)^(-1)
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: Bifurcated Batch Inversion (Jacobian -> Affine)
// ═══════════════════════════════════════════════════════════════════════════

// ─── G1: Batch invert Z_A, Z_C via Montgomery's trick ────────────────────
// 1 product + 1 Fermat inversion + 2 extractions = batch of 2 for 1 inv.
// Writes 64 bytes for pi_A at offset [0..63] and 64 bytes for pi_C at [192..255].
// Uses limbs_to_bytes_le (little-endian); caller applies serialize_proof_be.
static int32_t batch_invert_g1_pair(
    const JacobianPoint* pa_jac,   // pi_A in Jacobian coordinates
    const JacobianPoint* pc_jac,   // pi_C in Jacobian coordinates
    uint8_t*             proof_buf // Output: 256-byte proof buffer
) {
    bool pa_is_inf = jacobian_is_identity(pa_jac);
    bool pc_is_inf = jacobian_is_identity(pc_jac);

    // ─── Handle identity points (Z == 0 -> affine (0,0)) ─────────────
    if (pa_is_inf) memset(proof_buf, 0, 64);
    if (pc_is_inf) memset(proof_buf + 192, 0, 64);

    if (pa_is_inf && pc_is_inf) return 0;

    // ─── Solo inversion fast-path ────────────────────────────────────
    if (pa_is_inf || pc_is_inf) {
        const JacobianPoint* pt = pa_is_inf ? pc_jac : pa_jac;
        uint32_t offset = pa_is_inf ? 192 : 0;

        Fp9x29 inv_z;
        fp_inv(&inv_z, &pt->Z.fp);

        Fp9x29 inv_z_sq, inv_z_cub;
        fp_sqr(&inv_z_sq, &inv_z);
        fp_mul(&inv_z_cub, &inv_z_sq, &inv_z);

        Fp9x29 x_aff, y_aff, x_norm, y_norm;
        fp_mul(&x_aff, &pt->X.fp, &inv_z_sq);          // X / Z^2
        fp_mul(&y_aff, &pt->Y.fp, &inv_z_cub);         // Y / Z^3
        from_montgomery(&x_norm, &x_aff);
        from_montgomery(&y_norm, &y_aff);
        limbs_to_bytes_le(proof_buf + offset, x_norm);
        limbs_to_bytes_le(proof_buf + offset + 32, y_norm);
        return 0;
    }

    // ─── Montgomery's Trick: batch invert Z_A x Z_C ─────────────────
    Fp9x29 product;
    fp_mul(&product, &pa_jac->Z.fp, &pc_jac->Z.fp);    // Z_A * Z_C

    Fp9x29 inv_product;
    fp_inv(&inv_product, &product);                      // (Z_A * Z_C)^(-1)

    // Extract individual inverses
    Fp9x29 inv_z_a, inv_z_c;
    fp_mul(&inv_z_c, &inv_product, &pa_jac->Z.fp);      // (Z_A*Z_C)^(-1) * Z_A = Z_C^(-1)
    fp_mul(&inv_z_a, &inv_product, &pc_jac->Z.fp);      // (Z_A*Z_C)^(-1) * Z_C = Z_A^(-1)

    // ─── pi_A -> affine (offset 0) ────────────────────────────────
    {
        Fp9x29 inv_sq, inv_cub;
        fp_sqr(&inv_sq, &inv_z_a);
        fp_mul(&inv_cub, &inv_sq, &inv_z_a);

        Fp9x29 x_aff, y_aff, x_norm, y_norm;
        fp_mul(&x_aff, &pa_jac->X.fp, &inv_sq);
        fp_mul(&y_aff, &pa_jac->Y.fp, &inv_cub);
        from_montgomery(&x_norm, &x_aff);
        from_montgomery(&y_norm, &y_aff);
        limbs_to_bytes_le(proof_buf, x_norm);            // [0..31]:  pi_A.x
        limbs_to_bytes_le(proof_buf + 32, y_norm);       // [32..63]: pi_A.y
    }

    // ─── pi_C -> affine (offset 192) ──────────────────────────────
    {
        Fp9x29 inv_sq, inv_cub;
        fp_sqr(&inv_sq, &inv_z_c);
        fp_mul(&inv_cub, &inv_sq, &inv_z_c);

        Fp9x29 x_aff, y_aff, x_norm, y_norm;
        fp_mul(&x_aff, &pc_jac->X.fp, &inv_sq);
        fp_mul(&y_aff, &pc_jac->Y.fp, &inv_cub);
        from_montgomery(&x_norm, &x_aff);
        from_montgomery(&y_norm, &y_aff);
        limbs_to_bytes_le(proof_buf + 192, x_norm);      // [192..223]: pi_C.x
        limbs_to_bytes_le(proof_buf + 224, y_norm);       // [224..255]: pi_C.y
    }

    return 0;
}

// ─── G2: Invert Z_B in F_p^2 via norm ────────────────────────────────────
// norm(z) = c0^2 + c1^2 in Fp  ->  1 Fermat inversion in Fp
// inv(z) = conj(z) / norm(z) = (c0/N, -c1/N)
// Writes 128 bytes for pi_B at offset [64..191].
static int32_t invert_g2_to_affine(
    const JacobianPointG2* pb_jac,  // pi_B in Jacobian (Fp2 coordinates)
    uint8_t*               proof_buf // Output: 256-byte proof buffer
) {
    // Check for G2 identity: Z.c0 == 0 && Z.c1 == 0
    bool z_is_zero = true;
    for (uint32_t i = 0; i < NLIMBS; i++) {
        if (pb_jac->Z.c0.limb[i] != 0 || pb_jac->Z.c1.limb[i] != 0) {
            z_is_zero = false;
            break;
        }
    }
    if (z_is_zero) {
        memset(proof_buf + 64, 0, 128);      // pi_B = (0,0,0,0)
        return 0;
    }

    // Fp2 inversion via norm
    Fp2Element inv_z;
    fp2_inv(&inv_z, &pb_jac->Z);

    // inv_Z^2, inv_Z^3 in Fp2
    Fp2Element inv_z_sq, inv_z_cub;
    fp2_sqr(&inv_z_sq, &inv_z);
    fp2_mul(&inv_z_cub, &inv_z_sq, &inv_z);

    // X_affine = X * inv_Z^2
    Fp2Element x_aff;
    fp2_mul(&x_aff, &pb_jac->X, &inv_z_sq);

    // Y_affine = Y * inv_Z^3
    Fp2Element y_aff;
    fp2_mul(&y_aff, &pb_jac->Y, &inv_z_cub);

    // Serialize: from_montgomery -> limbs_to_bytes_le -> proof buffer
    Fp9x29 tmp;
    from_montgomery(&tmp, &x_aff.c0);
    limbs_to_bytes_le(proof_buf + 64, tmp);               // [64..95]:   pi_B.x.c0
    from_montgomery(&tmp, &x_aff.c1);
    limbs_to_bytes_le(proof_buf + 96, tmp);               // [96..127]:  pi_B.x.c1
    from_montgomery(&tmp, &y_aff.c0);
    limbs_to_bytes_le(proof_buf + 128, tmp);              // [128..159]: pi_B.y.c0
    from_montgomery(&tmp, &y_aff.c1);
    limbs_to_bytes_le(proof_buf + 160, tmp);              // [160..191]: pi_B.y.c1

    return 0;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: Big-Endian SIMD Swizzle Serialization
// ═══════════════════════════════════════════════════════════════════════════
// Reverses byte order of each 32-byte field element using wasm_i8x16_swizzle.
// 8 field elements x 32 bytes = 256 bytes. In-place. Zero allocation.
// Cost: 16 loads + 16 swizzles + 16 stores = 48 SIMD operations.

static void serialize_proof_be(uint8_t* proof_buf) {
    static const uint8_t REV_PATTERN[16] __attribute__((aligned(16))) = {
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
    };
    const v128_t rev_mask = wasm_v128_load(REV_PATTERN);

    // Process 8 field elements, each 32 bytes
    for (uint32_t i = 0; i < 256; i += 32) {
        v128_t lo = wasm_v128_load(proof_buf + i);         // bytes [0..15]
        v128_t hi = wasm_v128_load(proof_buf + i + 16);    // bytes [16..31]

        v128_t rev_lo = wasm_i8x16_swizzle(lo, rev_mask);  // reverse lo
        v128_t rev_hi = wasm_i8x16_swizzle(hi, rev_mask);  // reverse hi

        // Swap halves: reversed-hi -> [0..15], reversed-lo -> [16..31]
        wasm_v128_store(proof_buf + i, rev_hi);
        wasm_v128_store(proof_buf + i + 16, rev_lo);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: Full Proof Assembly Pipeline
// ═══════════════════════════════════════════════════════════════════════════
// Orchestrates: quotient -> iNTT -> batch inversion -> serialization.
// Writes the final 256 big-endian proof bytes to g_proof_buffer.

static int32_t assemble_groth16_proof(
    FieldElement*          a_poly,           // Domain-size array, OVERWRITTEN with H(x)
    const FieldElement*    b_poly,           // Domain-size array (read-only)
    const FieldElement*    c_poly,           // Domain-size array (read-only)
    const FieldElement*    ntt_roots,        // Roots of unity for NTT
    uint32_t               log_domain_size,
    const Fp9x29*          coset_gen_mont,   // Coset generator g in Montgomery form
    JacobianPoint*         pi_a_jac,         // MSM result for pi_A  (G1, Jacobian)
    JacobianPointG2*       pi_b_jac,         // MSM result for pi_B  (G2, Jacobian)
    JacobianPoint*         pi_c_jac          // MSM result for pi_C  (G1, Jacobian)
) {
    pa_ensure_init();

    // ─── Stage 1: Compute H(x) in evaluation form (in-place) ────────
    Fp9x29 inv_z_h;
    compute_inv_z_h(&inv_z_h, coset_gen_mont, log_domain_size);
    compute_quotient_inplace(a_poly, b_poly, c_poly, &inv_z_h, 1u << log_domain_size);

    // ─── Stage 2: iNTT -> H(x) coefficients ─────────────────────────
    ntt_bailey_4step(a_poly, log_domain_size, ntt_roots, true /* inverse */);

    // ─── Stage 3: Jacobian -> Affine serialization ───────────────────
    // Layout: [pi_A: 64B][pi_B: 128B][pi_C: 64B] = 256 bytes
    memset(g_proof_buffer, 0, 256);

    int32_t rc = batch_invert_g1_pair(pi_a_jac, pi_c_jac, g_proof_buffer);
    if (rc != 0) return rc;

    rc = invert_g2_to_affine(pi_b_jac, g_proof_buffer);
    if (rc != 0) return rc;

    // ─── Stage 4: Big-endian swizzle for Ethereum / snarkjs ─────────
    serialize_proof_be(g_proof_buffer);

    return 0;
}


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 9: Self-Test Suite (8 tests)
// ═══════════════════════════════════════════════════════════════════════════
// Returns (total_tests << 16) | passed_count.
// Caller checks: (result >> 16) == (result & 0xFFFF) -> all passed.

static int32_t run_proof_assembly_selftest() {
    pa_ensure_init();
    zk_engine_field_init();

    uint32_t pass_count = 0;

    // ─── Test 1: Fp2 multiplication identity ─────────────────────────
    // (2 + 3u) x (1 + 0u) = (2 + 3u)
    {
        Fp2Element a, b, result;
        memset(&a, 0, sizeof(a)); memset(&b, 0, sizeof(b));

        Fp9x29 two, three, one_val;
        memset(&two, 0, sizeof(Fp9x29));   two.limb[0] = 2;
        memset(&three, 0, sizeof(Fp9x29)); three.limb[0] = 3;
        memset(&one_val, 0, sizeof(Fp9x29)); one_val.limb[0] = 1;

        to_montgomery(&a.c0, &two);
        to_montgomery(&a.c1, &three);
        to_montgomery(&b.c0, &one_val);
        b.c1 = FP_ZERO;

        fp2_mul(&result, &a, &b);
        if (fp_equal(&result.c0, &a.c0) && fp_equal(&result.c1, &a.c1)) pass_count++;
    }

    // ─── Test 2: Fp2 squaring consistency ────────────────────────────
    // a^2 via fp2_sqr must equal a x a via fp2_mul
    {
        Fp2Element a, sq_result, mul_result;
        memset(&a, 0, sizeof(a));

        Fp9x29 five, seven;
        memset(&five, 0, sizeof(Fp9x29)); five.limb[0] = 5;
        memset(&seven, 0, sizeof(Fp9x29)); seven.limb[0] = 7;

        to_montgomery(&a.c0, &five);
        to_montgomery(&a.c1, &seven);

        fp2_sqr(&sq_result, &a);
        fp2_mul(&mul_result, &a, &a);
        if (fp_equal(&sq_result.c0, &mul_result.c0) &&
            fp_equal(&sq_result.c1, &mul_result.c1)) pass_count++;
    }

    // ─── Test 3: fp_inv correctness ──────────────────────────────────
    // a x a^(-1) = 1 (Montgomery form)
    {
        Fp9x29 a, inv_a, product;
        memset(&a, 0, sizeof(Fp9x29)); a.limb[0] = 7;
        to_montgomery(&a, &a);

        fp_inv(&inv_a, &a);
        fp_mul(&product, &a, &inv_a);

        Fp9x29 mont_one = fp_montgomery_one();
        if (fp_equal(&product, &mont_one)) pass_count++;
    }

    // ─── Test 4: fp2_inv correctness ─────────────────────────────────
    // a x a^(-1) = (1, 0) in Fp2
    {
        Fp2Element a, inv_a, product;
        memset(&a, 0, sizeof(a));

        Fp9x29 eleven, thirteen;
        memset(&eleven, 0, sizeof(Fp9x29)); eleven.limb[0] = 11;
        memset(&thirteen, 0, sizeof(Fp9x29)); thirteen.limb[0] = 13;

        to_montgomery(&a.c0, &eleven);
        to_montgomery(&a.c1, &thirteen);

        fp2_inv(&inv_a, &a);
        fp2_mul(&product, &a, &inv_a);

        Fp9x29 mont_one = fp_montgomery_one();
        if (fp_equal(&product.c0, &mont_one) && fp_equal(&product.c1, &FP_ZERO))
            pass_count++;
    }

    // ─── Test 5: Big-endian swizzle reversal ─────────────────────────
    // Sequential bytes -> swizzle -> each 32-byte field reversed
    {
        uint8_t buf[256];
        for (uint32_t i = 0; i < 256; i++) buf[i] = static_cast<uint8_t>(i & 0xFF);

        serialize_proof_be(buf);

        bool ok = true;
        for (uint32_t field = 0; field < 8 && ok; field++) {
            for (uint32_t j = 0; j < 32 && ok; j++) {
                uint8_t expected = static_cast<uint8_t>((field * 32 + 31 - j) & 0xFF);
                if (buf[field * 32 + j] != expected) ok = false;
            }
        }
        if (ok) pass_count++;
    }

    // ─── Test 6: Double swizzle = identity ───────────────────────────
    {
        uint8_t buf[256], original[256];
        for (uint32_t i = 0; i < 256; i++) {
            buf[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
            original[i] = buf[i];
        }
        serialize_proof_be(buf);
        serialize_proof_be(buf);
        if (memcmp(buf, original, 256) == 0) pass_count++;
    }

    // ─── Test 7: Batch G1 inversion round-trip ───────────────────────
    // Create Jacobian (X=2R, Y=3R, Z=5R), convert to affine, verify:
    //   x_aff x Z^2 == X  and  y_aff x Z^3 == Y   (in Montgomery form)
    {
        JacobianPoint pa, pc;
        memset(&pa, 0, sizeof(pa)); memset(&pc, 0, sizeof(pc));

        Fp9x29 v2, v3, v5, v11, v13, v17;
        memset(&v2, 0, sizeof(Fp9x29));  v2.limb[0] = 2;
        memset(&v3, 0, sizeof(Fp9x29));  v3.limb[0] = 3;
        memset(&v5, 0, sizeof(Fp9x29));  v5.limb[0] = 5;
        memset(&v11, 0, sizeof(Fp9x29)); v11.limb[0] = 11;
        memset(&v13, 0, sizeof(Fp9x29)); v13.limb[0] = 13;
        memset(&v17, 0, sizeof(Fp9x29)); v17.limb[0] = 17;

        to_montgomery(&pa.X.fp, &v2);  to_montgomery(&pa.Y.fp, &v3);
        to_montgomery(&pa.Z.fp, &v5);
        to_montgomery(&pc.X.fp, &v11); to_montgomery(&pc.Y.fp, &v13);
        to_montgomery(&pc.Z.fp, &v17);

        uint8_t proof_buf[256];
        memset(proof_buf, 0, 256);
        batch_invert_g1_pair(&pa, &pc, proof_buf);
        serialize_proof_be(proof_buf);  // LE -> BE so bytes_to_limbs can read

        // Read back and verify pi_A
        Fp9x29 xa_read = bytes_to_limbs(proof_buf);
        Fp9x29 ya_read = bytes_to_limbs(proof_buf + 32);
        Fp9x29 xa_mont, ya_mont;
        to_montgomery(&xa_mont, &xa_read);
        to_montgomery(&ya_mont, &ya_read);

        Fp9x29 z_sq, z_cub, check;
        fp_sqr(&z_sq, &pa.Z.fp);
        fp_mul(&check, &xa_mont, &z_sq);
        bool ok = fp_equal(&check, &pa.X.fp);

        fp_mul(&z_cub, &z_sq, &pa.Z.fp);
        fp_mul(&check, &ya_mont, &z_cub);
        ok = ok && fp_equal(&check, &pa.Y.fp);

        // Read back and verify pi_C
        Fp9x29 xc_read = bytes_to_limbs(proof_buf + 192);
        Fp9x29 yc_read = bytes_to_limbs(proof_buf + 224);
        Fp9x29 xc_mont, yc_mont;
        to_montgomery(&xc_mont, &xc_read);
        to_montgomery(&yc_mont, &yc_read);

        fp_sqr(&z_sq, &pc.Z.fp);
        fp_mul(&check, &xc_mont, &z_sq);
        ok = ok && fp_equal(&check, &pc.X.fp);

        fp_mul(&z_cub, &z_sq, &pc.Z.fp);
        fp_mul(&check, &yc_mont, &z_cub);
        ok = ok && fp_equal(&check, &pc.Y.fp);

        if (ok) pass_count++;
    }

    // ─── Test 8: Quotient computation (A*B - C = 0 => T = 0) ────────
    {
        FieldElement a_elem, b_elem, c_elem;
        memset(&a_elem, 0, sizeof(a_elem));
        memset(&b_elem, 0, sizeof(b_elem));
        memset(&c_elem, 0, sizeof(c_elem));

        Fp9x29 six_val, one_val;
        memset(&six_val, 0, sizeof(Fp9x29)); six_val.limb[0] = 6;
        memset(&one_val, 0, sizeof(Fp9x29)); one_val.limb[0] = 1;

        to_montgomery(&a_elem.fp, &six_val);
        to_montgomery(&b_elem.fp, &one_val);
        c_elem.fp = a_elem.fp;  // C = A, so A*B - C = A*1 - A = 0

        // inv_z_h is arbitrary (result is 0 x anything = 0)
        Fp9x29 arbitrary;
        memset(&arbitrary, 0, sizeof(Fp9x29)); arbitrary.limb[0] = 42;
        Fp9x29 inv_z_h;
        to_montgomery(&inv_z_h, &arbitrary);

        compute_quotient_inplace(&a_elem, &b_elem, &c_elem, &inv_z_h, 1);

        bool ok = true;
        for (uint32_t j = 0; j < NLIMBS; j++) {
            if (a_elem.fp.limb[j] != 0) { ok = false; break; }
        }
        if (ok) pass_count++;
    }

    return static_cast<int32_t>((8u << 16) | pass_count);
}
