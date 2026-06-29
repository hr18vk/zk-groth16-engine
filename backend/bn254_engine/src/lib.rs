//! ═══════════════════════════════════════════════════════════════════════════
//! Zero-Knowledge Engine
//! BN254 Groth16 Verification Engine (Rust/PyO3 Native Extension)
//! ═══════════════════════════════════════════════════════════════════════════
//!
//! This module exposes three functions to Python:
//!
//!   1. `validate_point_g1(x: bytes, y: bytes) -> bool`
//!      Deserializes x, y as Fq elements, checks the affine point is on the
//!      BN254 G₁ curve (Y² = X³ + 3) AND in the prime-order r subgroup.
//!
//!   2. `initialize_vkey(vkey_bytes: bytes, num_public_inputs: int) -> None`
//!      Deserializes the verification key, computes the PreparedVerifyingKey
//!      (Fp12 cyclotomic precomputations for e(α,β), -γ_G2, -δ_G2), and
//!      caches it in a global RwLock. MUST be called once at server boot.
//!      Subsequent calls are idempotent (overwrites the cached PVK).
//!
//!   3. `verify_groth16(proof_bytes: bytes, public_inputs: list[bytes]) -> bool`
//!      Full Groth16 verification using the globally cached PVK. Deserializes
//!      the proof, accumulates the public-input linear combination, evaluates
//!      the pairing equation. The GIL is RELEASED for the entire computation.
//!      Raises ValueError if initialize_vkey() has not been called.
//!
//! ARCHITECTURAL LAWS:
//!   - py.allow_threads() wraps ALL arithmetic heavier than O(1).
//!   - All coordinate inputs are BIG-ENDIAN 32-byte field elements.
//!   - Subgroup checks use scalar multiplication by r (not cofactor tricks).
//!   - Invalid points produce Python ValueError, never panic.
//!   - The PreparedVerifyingKey is computed ONCE at boot (not per-request).
//!     This avoids ~1-2ms of redundant Fp12 cyclotomic precomputation per call.
//! ═══════════════════════════════════════════════════════════════════════════

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;

use ark_bn254::{Bn254, Fq, Fq2, Fr, G1Affine, G2Affine, G1Projective};
use ark_ec::{AffineRepr, CurveGroup, AdditiveGroup};
use ark_ff::{BigInteger256, PrimeField, BigInteger};
use ark_groth16::{Groth16, Proof, VerifyingKey, PreparedVerifyingKey};
use ark_snark::SNARK;
use ark_std::Zero;
use num_bigint::BigUint;
use std::sync::RwLock;

// Global lock for the PreparedVerifyingKey to prevent cyclotomic CPU burn on every request.
static GLOBAL_PVK: RwLock<Option<PreparedVerifyingKey<Bn254>>> = RwLock::new(None);

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: Field Element Deserialization (Big-Endian → ark-ff)
// ═══════════════════════════════════════════════════════════════════════════

/// Converts a 32-byte big-endian slice to an ark-ff Fq element.
/// Returns Err if the value >= p (the BN254 base field prime).
fn bytes_to_fq(data: &[u8]) -> Result<Fq, String> {
    if data.len() != 32 {
        return Err(format!("Fq: expected 32 bytes, got {}", data.len()));
    }
    // ark-ff BigInteger256 expects little-endian u64 limbs
    let val = BigUint::from_bytes_be(data);
    let limbs = val.to_u64_digits();
    let mut arr = [0u64; 4];
    for (i, &limb) in limbs.iter().enumerate().take(4) {
        arr[i] = limb;
    }
    let big = BigInteger256::new(arr);
    Fq::from_bigint(big).ok_or_else(|| "Fq: value >= field modulus p".to_string())
}

/// Converts a 32-byte big-endian slice to an ark-ff Fr element (scalar field).
fn bytes_to_fr(data: &[u8]) -> Result<Fr, String> {
    if data.len() != 32 {
        return Err(format!("Fr: expected 32 bytes, got {}", data.len()));
    }
    let val = BigUint::from_bytes_be(data);
    let limbs = val.to_u64_digits();
    let mut arr = [0u64; 4];
    for (i, &limb) in limbs.iter().enumerate().take(4) {
        arr[i] = limb;
    }
    let big = BigInteger256::new(arr);
    Fr::from_bigint(big).ok_or_else(|| "Fr: value >= scalar field order r".to_string())
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: Point Deserialization with Subgroup Checks
// ═══════════════════════════════════════════════════════════════════════════

/// Deserializes a G₁ affine point from two 32-byte big-endian coordinates.
/// Performs:
///   1. Field membership check (x, y < p)
///   2. Curve equation check: y² ≡ x³ + 3 (mod p)
///   3. Subgroup check: [r]P == O (point at infinity)
///
/// The subgroup check prevents twisted-curve / small-subgroup attacks where
/// an attacker provides a point on a curve with the same equation but wrong
/// embedding degree, or a point in a low-order subgroup of the trace-zero
/// subgroup.
fn deserialize_g1(x_bytes: &[u8], y_bytes: &[u8]) -> Result<G1Affine, String> {
    let x = bytes_to_fq(x_bytes)?;
    let y = bytes_to_fq(y_bytes)?;

    // Construct the affine point — ark-ec checks y² = x³ + b automatically
    let point = G1Affine::new_unchecked(x, y);

    // Explicit curve equation check
    if !point.is_on_curve() {
        return Err("G1: point not on curve (y² ≠ x³ + 3 mod p)".to_string());
    }

    // Subgroup check: [r]P must be the identity
    if !point.is_in_correct_subgroup_assuming_on_curve() {
        return Err("G1: point not in prime-order subgroup ([r]P ≠ O)".to_string());
    }

    Ok(point)
}

/// Deserializes a G₂ affine point from four 32-byte big-endian coordinates.
/// G₂ lives over Fq2 = Fq[u] / (u² + 1).
///
/// Wire format from ZkProofExporter (see frontend/src/lib/ZkProofExporter.ts):
///   pi_B.x = (c0, c1) where Fq2 element = c0 + c1·u
///   pi_B.y = (c0, c1) where Fq2 element = c0 + c1·u
///
/// CRITICAL: The ZkProofExporter stores G2 x-coordinate as [x.c0, x.c1] at
/// byte offsets [64..95, 96..127] and y-coordinate as [y.c0, y.c1] at
/// [128..159, 160..191]. This is the RAW layout — NOT the snarkjs-swizzled
/// [c1, c0] order used in toGroth16Json(). We read the raw 256-byte buffer.
fn deserialize_g2(
    x_c0_bytes: &[u8],
    x_c1_bytes: &[u8],
    y_c0_bytes: &[u8],
    y_c1_bytes: &[u8],
) -> Result<G2Affine, String> {
    let x_c0 = bytes_to_fq(x_c0_bytes)?;
    let x_c1 = bytes_to_fq(x_c1_bytes)?;
    let y_c0 = bytes_to_fq(y_c0_bytes)?;
    let y_c1 = bytes_to_fq(y_c1_bytes)?;

    let x = Fq2::new(x_c0, x_c1);
    let y = Fq2::new(y_c0, y_c1);

    let point = G2Affine::new_unchecked(x, y);

    if !point.is_on_curve() {
        return Err("G2: point not on twisted curve".to_string());
    }

    if !point.is_in_correct_subgroup_assuming_on_curve() {
        return Err("G2: point not in prime-order subgroup ([r]P ≠ O)".to_string());
    }

    Ok(point)
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: Groth16 Verification Core
// ═══════════════════════════════════════════════════════════════════════════

/// Deserializes the verification key from a packed binary format.
///
/// VKey binary layout (big-endian, each coordinate = 32 bytes):
///   [  0.. 63]  alpha_g1      (G1: x, y)
///   [ 64..191]  beta_g2       (G2: x.c0, x.c1, y.c0, y.c1)
///   [192..319]  gamma_g2      (G2: x.c0, x.c1, y.c0, y.c1)
///   [320..447]  delta_g2      (G2: x.c0, x.c1, y.c0, y.c1)
///   [448..]     IC points     (G1: x, y) × (num_public_inputs + 1)
fn deserialize_vkey(data: &[u8], num_public_inputs: usize) -> Result<VerifyingKey<Bn254>, String> {
    let expected_len = 448 + (num_public_inputs + 1) * 64;
    if data.len() < expected_len {
        return Err(format!(
            "VKey: expected >= {} bytes for {} public inputs, got {}",
            expected_len, num_public_inputs, data.len()
        ));
    }

    let alpha_g1 = deserialize_g1(&data[0..32], &data[32..64])?;
    let beta_g2 = deserialize_g2(
        &data[64..96], &data[96..128], &data[128..160], &data[160..192],
    )?;
    let gamma_g2 = deserialize_g2(
        &data[192..224], &data[224..256], &data[256..288], &data[288..320],
    )?;
    let delta_g2 = deserialize_g2(
        &data[320..352], &data[352..384], &data[384..416], &data[416..448],
    )?;

    let mut gamma_abc_g1 = Vec::with_capacity(num_public_inputs + 1);
    for i in 0..=num_public_inputs {
        let offset = 448 + i * 64;
        let ic_point = deserialize_g1(&data[offset..offset + 32], &data[offset + 32..offset + 64])?;
        gamma_abc_g1.push(ic_point);
    }

    Ok(VerifyingKey {
        alpha_g1,
        beta_g2,
        gamma_g2,
        delta_g2,
        gamma_abc_g1,
    })
}

/// Core Groth16 verification with unrolled pairing.
///
/// Computes:
///   L = IC[0] + Σ (public_inputs[i] × IC[i+1])
///
/// Then checks the pairing equation:
///   e(π_A, π_B) == e(α, β) · e(L, γ) · e(π_C, δ)
///
/// Which is equivalent to checking:
///   e(π_A, π_B) · e(-L, γ) · e(-π_C, δ) == e(α, β)
fn verify_proof_inner(
    proof: &Proof<Bn254>,
    pvk: &PreparedVerifyingKey<Bn254>,
    public_inputs: &[Fr],
) -> Result<bool, String> {
    if public_inputs.len() + 1 != pvk.vk.gamma_abc_g1.len() {
        return Err("Mismatched number of public inputs".to_string());
    }

    // 1. Accumulate public signals using strict w=5 MSM
    let msm_result = strict_w5_msm(&pvk.vk.gamma_abc_g1[1..], public_inputs);

    // 2. Add to IC_0 (base point)
    let g_ic = pvk.vk.gamma_abc_g1[0].into_group() + msm_result;

    // 3. Evaluate multi-pairing
    use ark_ec::pairing::Pairing;
    let a1: <Bn254 as Pairing>::G1Prepared = proof.a.into();
    let b1: <Bn254 as Pairing>::G2Prepared = proof.b.into();
    let a2: <Bn254 as Pairing>::G1Prepared = (g_ic.into_affine()).into();
    let b2 = pvk.gamma_g2_neg_pc.clone();
    let a3: <Bn254 as Pairing>::G1Prepared = (proof.c).into();
    let b3 = pvk.delta_g2_neg_pc.clone();

    let pairing = Bn254::multi_pairing([a1, a2, a3], [b1, b2, b3]);

    Ok(pairing.0 == pvk.alpha_g1_beta_g2)
}

/// Manually unrolls a w=5 windowed Pippenger MSM to protect the L1 cache.
/// Each 254-bit scalar is divided into 5-bit windows, accumulating into 31 buckets.
fn strict_w5_msm(bases: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    const W: usize = 5;
    const NUM_BUCKETS: usize = (1 << W) - 1;
    let mut res = G1Projective::zero();
    
    let num_bits = Fr::MODULUS_BIT_SIZE as usize; // 254
    let num_windows = (num_bits + W - 1) / W; 
    
    for window in (0..num_windows).rev() {
        if window != num_windows - 1 {
            for _ in 0..W {
                res.double_in_place();
            }
        }
        
        let mut buckets = vec![G1Projective::zero(); NUM_BUCKETS];
        
        for (base, scalar) in bases.iter().zip(scalars.iter()) {
            let bigint = scalar.into_bigint();
            let mut window_val = 0usize;
            for b in 0..W {
                let bit_idx = window * W + b;
                if bit_idx < num_bits {
                    if bigint.get_bit(bit_idx) {
                        window_val |= 1 << b;
                    }
                }
            }
            if window_val > 0 {
                buckets[window_val - 1] += base;
            }
        }
        
        let mut running_sum = G1Projective::zero();
        for bucket in buckets.iter().rev() {
            running_sum += bucket;
            res += running_sum;
        }
    }
    res
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: PyO3 Python Bindings
// ═══════════════════════════════════════════════════════════════════════════

/// Validates that a G1 point (given as two 32-byte big-endian coordinates)
/// is on the BN254 curve and in the correct prime-order subgroup.
///
/// Returns True if valid, raises ValueError if invalid.
#[pyfunction]
fn validate_point_g1(py: Python<'_>, x: &Bound<'_, PyBytes>, y: &Bound<'_, PyBytes>) -> PyResult<bool> {
    let x_data = x.as_bytes();
    let y_data = y.as_bytes();

    py.allow_threads(|| {
        deserialize_g1(x_data, y_data)
            .map(|_| true)
            .map_err(|e| PyValueError::new_err(e))
    })
}

/// Initializes the global verification key in host RAM.
/// Deserializes the bytes and prepares the PVK, locking it in `GLOBAL_PVK`.
/// This prevents cyclotomic Fp12 operations from burning the CPU per-request.
#[pyfunction]
fn initialize_vkey(py: Python<'_>, vkey_bytes: &Bound<'_, PyBytes>, num_public_inputs: usize) -> PyResult<()> {
    let vkey_data = vkey_bytes.as_bytes().to_vec();
    py.allow_threads(move || {
        let vk = deserialize_vkey(&vkey_data, num_public_inputs)
            .map_err(|e| PyValueError::new_err(format!("VKey: {}", e)))?;
        let pvk = Groth16::<Bn254>::process_vk(&vk)
            .map_err(|e| PyValueError::new_err(format!("VKey processing: {}", e)))?;
        
        let mut lock = GLOBAL_PVK.write().unwrap();
        *lock = Some(pvk);
        Ok(())
    })
}

/// Full Groth16 proof verification using the globally locked PVK.
///
/// Arguments:
///   proof_bytes: 256 bytes — the raw proof (π_A, π_B, π_C) in big-endian.
///   public_inputs: List of 32-byte big-endian scalar field elements.
///
/// Returns True if the proof is valid, False if the pairing check fails.
/// Raises ValueError for deserialization or structural errors.
///
/// THE GIL IS RELEASED for the entire function body via py.allow_threads().
#[pyfunction]
fn verify_groth16(
    py: Python<'_>,
    proof_bytes: &Bound<'_, PyBytes>,
    public_inputs: Vec<Bound<'_, PyBytes>>,
) -> PyResult<bool> {
    let proof_data = proof_bytes.as_bytes().to_vec();
    let inputs_data: Vec<Vec<u8>> = public_inputs
        .iter()
        .map(|b| b.as_bytes().to_vec())
        .collect();

    py.allow_threads(move || {
        // ── Acquire global PVK read lock ───────────────────────────────
        let lock = GLOBAL_PVK.read().unwrap();
        let pvk = lock.as_ref().ok_or_else(|| PyValueError::new_err("Verification engine not initialized with vkey"))?;

        // ── Step 1: Deserialize proof (256 bytes) ──────────────────────
        if proof_data.len() != 256 {
            return Err(PyValueError::new_err(format!("Proof must be exactly 256 bytes")));
        }

        let pi_a = deserialize_g1(&proof_data[0..32], &proof_data[32..64])
            .map_err(|e| PyValueError::new_err(format!("π_A: {}", e)))?;
        let pi_b = deserialize_g2(
            &proof_data[64..96], &proof_data[96..128],
            &proof_data[128..160], &proof_data[160..192],
        ).map_err(|e| PyValueError::new_err(format!("π_B: {}", e)))?;
        let pi_c = deserialize_g1(&proof_data[192..224], &proof_data[224..256])
            .map_err(|e| PyValueError::new_err(format!("π_C: {}", e)))?;

        let proof = Proof { a: pi_a, b: pi_b, c: pi_c };

        // ── Step 2: Deserialize public inputs ──────────────────────────
        let mut fr_inputs = Vec::with_capacity(inputs_data.len());
        for (i, input) in inputs_data.iter().enumerate() {
            let fr = bytes_to_fr(input)
                .map_err(|e| PyValueError::new_err(format!("public_input[{}]: {}", i, e)))?;
            fr_inputs.push(fr);
        }

        // ── Step 3: Verify the pairing equation ────────────────────────
        verify_proof_inner(&proof, pvk, &fr_inputs)
            .map_err(|e| PyValueError::new_err(e))
    })
}

/// Python module definition.
#[pymodule]
fn bn254_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_point_g1, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_vkey, m)?)?;
    m.add_function(wrap_pyfunction!(verify_groth16, m)?)?;
    Ok(())
}
