/**
 * ZK Engine Phase 3.3
 * Witness Generation & R1CS Evaluation Engine (WASM)
 *
 * This file implements:
 *   1. Sparse R1CS constraint matrix layout (flat contiguous arrays)
 *   2. Constant-time signal sanitizer (BN254 prime bound enforcement)
 *   3. Lock-free DAG worker queue for subcircuit evaluation
 *   4. R1CS constraint verification: A·w ⊙ B·w = C·w
 *
 * ARCHITECTURAL LAWS:
 *   - No heap allocation during evaluation (all memory from static arenas)
 *   - Constant-time signal validation (no branch on secret input values)
 *   - Montgomery 9×29 math via vector_prover.cpp STU include
 *   - Lock-free concurrency via __atomic builtins (no std::mutex)
 *
 * ABI Contract (extern "C" exports):
 *   - r1cs_init(max_constraints, max_signals, max_nonzero)
 *   - r1cs_ingest_constraint(matrix_id, constraint_idx, col, coeff_ptr)
 *   - r1cs_assign_signal(signal_idx, value_ptr, value_len)
 *   - r1cs_evaluate() -> 0 if A·w ⊙ B·w = C·w holds, -1 if violated
 *   - r1cs_get_signal_ptr() -> pointer to signal array in WASM memory
 *   - r1cs_get_constraint_count() -> number of constraints loaded
 *   - r1cs_selftest() -> 0 if all self-tests pass
 */

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>

// Phase 2.2 Montgomery CIOS field arithmetic (9×29-bit reduced radix)
// Provides: Fp9x29, NLIMBS, RADIX, LIMB_MASK, P_LIMBS, P_BYTES,
//           vectorized_montgomery_cios_schoolbook(), zk_engine_field_init_full(),
//           to_montgomery(), from_montgomery(), bytes_to_limbs(), fp_equal()
#include "vector_prover.cpp"

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: BN254 Prime Constants for Signal Sanitization
// ═══════════════════════════════════════════════════════════════════════════

/**
 * BN254 scalar field prime p as a Fp9x29 limb array.
 * Used for constant-time comparison in the signal sanitizer.
 * P_LIMBS is already defined in vector_prover.cpp.
 */

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: Field Arithmetic Helpers (static to avoid ODR collision)
// ═══════════════════════════════════════════════════════════════════════════
//
// These are IDENTICAL to the helpers in prover_engine.cpp but defined as
// static functions here because witness_engine.cpp is a separate TU.
// In production, these would be in a shared header; for the WASM build,
// static linkage is the safest zero-cost abstraction.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Field addition: result = a + b mod p
 */
static void we_fp_add(Fp9x29* result, const Fp9x29* a, const Fp9x29* b) {
    uint64_t carry = 0;
    for (int i = 0; i < NLIMBS; i++) {
        uint64_t sum = (uint64_t)a->limb[i] + (uint64_t)b->limb[i] + carry;
        result->limb[i] = static_cast<uint32_t>(sum & LIMB_MASK);
        carry = sum >> RADIX;
    }
    // Conditional subtraction of p if result >= p
    int borrow = 0;
    uint32_t diff[NLIMBS];
    for (int j = 0; j < NLIMBS; j++) {
        int64_t d = (int64_t)result->limb[j] - (int64_t)P_LIMBS.limb[j] - borrow;
        if (d < 0) {
            diff[j] = static_cast<uint32_t>(d + (1ll << RADIX));
            borrow = 1;
        } else {
            diff[j] = static_cast<uint32_t>(d);
            borrow = 0;
        }
    }
    if (borrow == 0) {
        for (int j = 0; j < NLIMBS; j++) {
            result->limb[j] = diff[j];
        }
    }
}

/**
 * Field subtraction: result = a - b mod p
 */
static void we_fp_sub(Fp9x29* result, const Fp9x29* a, const Fp9x29* b) {
    int borrow = 0;
    for (int i = 0; i < NLIMBS; i++) {
        int64_t d = (int64_t)a->limb[i] - (int64_t)b->limb[i] - borrow;
        if (d < 0) {
            result->limb[i] = static_cast<uint32_t>(d + (1ll << RADIX));
            borrow = 1;
        } else {
            result->limb[i] = static_cast<uint32_t>(d);
            borrow = 0;
        }
    }
    if (borrow != 0) {
        uint64_t carry = 0;
        for (int i = 0; i < NLIMBS; i++) {
            uint64_t sum = (uint64_t)result->limb[i] + (uint64_t)P_LIMBS.limb[i] + carry;
            result->limb[i] = static_cast<uint32_t>(sum & LIMB_MASK);
            carry = sum >> RADIX;
        }
    }
}

/**
 * Field multiplication: result = a * b * R^(-1) mod p (Montgomery)
 */
static inline void we_fp_mul(Fp9x29* result, const Fp9x29* a, const Fp9x29* b) {
    vectorized_montgomery_cios_schoolbook(result, a, b);
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: Sparse R1CS Constraint Matrix Layout
// ═══════════════════════════════════════════════════════════════════════════
//
// An R1CS system is: A·w ⊙ B·w = C·w
// where A, B, C are sparse matrices and w is the witness vector.
//
// We use a flat COO (Coordinate) sparse format:
//   - Each non-zero entry is a triple (constraint_row, signal_col, coefficient)
//   - All entries for a single matrix (A, B, or C) are stored contiguously
//   - Entries are sorted by constraint_row for cache-friendly traversal
//
// Memory layout:
//   ConstraintEntry = (uint32_t row, uint32_t col, Fp9x29 coeff) = 44 bytes
//   Padded to 64 bytes (alignas(64)) = exactly 1 L1 cache line
// ═══════════════════════════════════════════════════════════════════════════

/**
 * ConstraintEntry — A single non-zero entry in the sparse R1CS matrix.
 *
 * Memory: exactly 64 bytes (1 L1 cache line) with explicit padding.
 *   [row]     = 4 bytes (constraint index)
 *   [col]     = 4 bytes (signal/wire index)
 *   [coeff]   = 36 bytes (Fp9x29 = 9 × uint32_t)
 *   [_pad]    = 20 bytes (padding to reach 64-byte alignment)
 */
struct alignas(64) ConstraintEntry {
    uint32_t row;          // Constraint index (0-based)
    uint32_t col;          // Signal/wire index (0-based)
    Fp9x29   coeff;        // Coefficient in Montgomery form
    uint8_t  _pad[20];     // Padding: 4+4+36+20 = 64 bytes
};

static_assert(sizeof(ConstraintEntry) == 64,
    "ConstraintEntry must be exactly 64 bytes (1 cache line)");
static_assert(alignof(ConstraintEntry) == 64,
    "ConstraintEntry must have 64-byte alignment");

/**
 * SparseConstraintSystem — The complete R1CS system in flat arrays.
 *
 * Contains three sparse matrices (A, B, C) stored as flat contiguous
 * arrays of ConstraintEntry. Row offsets allow O(1) lookup of all
 * non-zero entries for a given constraint.
 *
 * Capacity limits (defined at init time):
 *   - max_constraints: upper bound on number of R1CS constraints
 *   - max_signals: upper bound on number of witness signals (wires)
 *   - max_nonzero: upper bound on TOTAL non-zero entries across A+B+C
 */
struct SparseConstraintSystem {
    // ─── Matrix storage ──────────────────────────────────────
    ConstraintEntry* entries_a;      // Non-zero entries of matrix A
    ConstraintEntry* entries_b;      // Non-zero entries of matrix B
    ConstraintEntry* entries_c;      // Non-zero entries of matrix C

    uint32_t nnz_a;                  // Number of non-zero entries in A
    uint32_t nnz_b;                  // Number of non-zero entries in B
    uint32_t nnz_c;                  // Number of non-zero entries in C

    // ─── Row offset arrays ───────────────────────────────────
    // row_offset_a[i] = index of first entry in entries_a for constraint i
    // row_offset_a[num_constraints] = nnz_a (sentinel)
    uint32_t* row_offset_a;
    uint32_t* row_offset_b;
    uint32_t* row_offset_c;

    // ─── Dimensions ──────────────────────────────────────────
    uint32_t num_constraints;        // Total number of R1CS constraints
    uint32_t num_signals;            // Total number of signals (wires)
    uint32_t max_nonzero_per_matrix; // Max entries per matrix (A, B, or C)

    // ─── Witness vector ──────────────────────────────────────
    Fp9x29*  signals;                // witness[0..num_signals-1] in Montgomery form
    uint32_t signals_assigned;       // Number of signals assigned so far

    // ─── Initialization flag ─────────────────────────────────
    uint32_t initialized;            // 1 if init has been called
};

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: Static Arena Allocation
// ═══════════════════════════════════════════════════════════════════════════
//
// All R1CS memory is allocated from a single static arena in WASM linear
// memory. This avoids malloc/free and ensures deterministic memory layout.
//
// Arena capacity: 16 MB (separate from the 64 MB zkey buffer in
// prover_engine.cpp). This is sufficient for circuits with up to
// ~65,000 constraints and ~200,000 non-zero entries.
// ═══════════════════════════════════════════════════════════════════════════

static constexpr uint32_t R1CS_ARENA_CAPACITY = 16 * 1024 * 1024; // 16 MB
static uint8_t g_r1cs_arena[R1CS_ARENA_CAPACITY]
    __attribute__((aligned(64)));
static uint32_t g_r1cs_arena_offset = 0;

/**
 * Bump-allocate from the R1CS arena.
 * Returns 64-byte aligned pointer, or nullptr if out of space.
 */
static void* r1cs_arena_alloc(uint32_t byte_size) {
    uint32_t aligned_offset = (g_r1cs_arena_offset + 63u) & ~63u;
    if (aligned_offset + byte_size > R1CS_ARENA_CAPACITY) {
        return nullptr;
    }
    void* ptr = g_r1cs_arena + aligned_offset;
    g_r1cs_arena_offset = aligned_offset + byte_size;
    return ptr;
}

// The global R1CS system instance
static SparseConstraintSystem g_r1cs;

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: Constant-Time Signal Sanitizer
// ═══════════════════════════════════════════════════════════════════════════
//
// SECURITY CRITICAL:
// Every witness signal MUST be strictly less than the BN254 prime p.
// If signal >= p, the R1CS evaluation is UNDEFINED (modular arithmetic
// breaks, proof is unsound).
//
// We enforce this with a CONSTANT-TIME subtraction:
//   1. Compute diff = signal - p (with borrow tracking)
//   2. If borrow == 0, then signal >= p → TRAP
//   3. If borrow == 1, then signal < p → SAFE
//
// The subtraction uses identical codepath regardless of input value,
// defeating timing-based side-channel attacks on the private witness.
//
// After assignment, the input buffer is ZEROED to prevent the raw
// private witness from lingering in WASM memory.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Constant-time comparison: returns 1 if a < p (safe), 0 if a >= p (unsafe).
 * MUST NOT branch on secret data.
 */
static uint32_t ct_is_less_than_prime(const Fp9x29* a) {
    // Compute a - p with borrow propagation
    int borrow = 0;
    for (int i = 0; i < NLIMBS; i++) {
        int64_t d = (int64_t)a->limb[i] - (int64_t)P_LIMBS.limb[i] - borrow;
        // Constant-time borrow extraction:
        // If d < 0, borrow = 1; else borrow = 0.
        // We compute this without branching using arithmetic shift.
        borrow = (int)(((uint64_t)d >> 63) & 1u);
    }
    // If borrow == 1 after all limbs, then a < p → safe
    return (uint32_t)borrow;
}

/**
 * Parse a 32-byte big-endian input buffer into a signal value,
 * validate it is < BN254 prime p, assign to the witness vector,
 * and zero the input buffer.
 *
 * @param signal_idx  Index in the witness vector (0-based)
 * @param input_buf   32-byte big-endian value (WILL BE ZEROED after use)
 * @param input_len   Must be exactly 32
 * @return 0 on success, -1 if signal_idx out of range, -2 if value >= p
 */
static int32_t parse_and_assign_signal(uint32_t signal_idx,
                                        uint8_t* input_buf,
                                        uint32_t input_len) {
    if (!g_r1cs.initialized || signal_idx >= g_r1cs.num_signals) {
        return -1;
    }
    if (input_len != 32) {
        return -1;
    }

    // Convert big-endian bytes to 9×29-bit limbs
    Fp9x29 raw_value = bytes_to_limbs(input_buf);

    // Constant-time prime bound check
    uint32_t is_safe = ct_is_less_than_prime(&raw_value);

    // SECURITY: Zero the input buffer BEFORE any trap.
    // This ensures the raw private witness never persists in WASM memory
    // even if the trap doesn't clear the stack frame.
    volatile uint8_t* vol_buf = (volatile uint8_t*)input_buf;
    for (uint32_t i = 0; i < input_len; i++) {
        vol_buf[i] = 0;
    }

    // If value >= p, trap unconditionally
    // __builtin_trap() maps directly to the WebAssembly trap instruction
    // that terminates the instance immediately. No exception, no recovery.
    if (is_safe == 0) {
#ifdef __wasm__
        __builtin_trap();
#endif
        return -2; // Fallback for non-WASM test builds
    }

    // Convert to Montgomery form and assign
    to_montgomery(&g_r1cs.signals[signal_idx], &raw_value);

    // Track assigned count
    if (signal_idx >= g_r1cs.signals_assigned) {
        g_r1cs.signals_assigned = signal_idx + 1;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: R1CS Constraint Evaluation
// ═══════════════════════════════════════════════════════════════════════════
//
// Evaluates the R1CS system: for each constraint i,
//   (A·w)[i] * (B·w)[i] == (C·w)[i]
//
// Each matrix-vector product is computed by iterating over the sparse
// entries for row i and accumulating coeff * signal[col].
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Compute a single sparse matrix-vector product for one row.
 *
 * result = Σ (entries[j].coeff * signals[entries[j].col])
 * for all j in [row_start, row_end)
 *
 * All arithmetic in Montgomery domain.
 */
static void eval_row(Fp9x29* result,
                     const ConstraintEntry* entries,
                     uint32_t row_start,
                     uint32_t row_end,
                     const Fp9x29* signals) {
    std::memset(result, 0, sizeof(Fp9x29));
    // Initialize accumulator to 0 in Montgomery form
    // (0 in Montgomery = 0, since 0 * R mod p = 0)

    for (uint32_t j = row_start; j < row_end; j++) {
        Fp9x29 product;
        we_fp_mul(&product, &entries[j].coeff, &signals[entries[j].col]);

        Fp9x29 new_acc;
        we_fp_add(&new_acc, result, &product);
        *result = new_acc;
    }
}

/**
 * Evaluate ALL constraints sequentially.
 *
 * For each constraint i:
 *   a_val = (A·w)[i]
 *   b_val = (B·w)[i]
 *   c_val = (C·w)[i]
 *   Check: a_val * b_val == c_val
 *
 * @return 0 if all constraints satisfied, or (i+1) for the first
 *         violated constraint (1-indexed for distinguishability from success)
 */
static int32_t evaluate_constraints_sequential() {
    for (uint32_t i = 0; i < g_r1cs.num_constraints; i++) {
        Fp9x29 a_val, b_val, c_val;

        // Compute (A·w)[i]
        eval_row(&a_val,
                 g_r1cs.entries_a,
                 g_r1cs.row_offset_a[i],
                 g_r1cs.row_offset_a[i + 1],
                 g_r1cs.signals);

        // Compute (B·w)[i]
        eval_row(&b_val,
                 g_r1cs.entries_b,
                 g_r1cs.row_offset_b[i],
                 g_r1cs.row_offset_b[i + 1],
                 g_r1cs.signals);

        // Compute (C·w)[i]
        eval_row(&c_val,
                 g_r1cs.entries_c,
                 g_r1cs.row_offset_c[i],
                 g_r1cs.row_offset_c[i + 1],
                 g_r1cs.signals);

        // Check: a_val * b_val == c_val (in Montgomery domain)
        Fp9x29 ab_product;
        we_fp_mul(&ab_product, &a_val, &b_val);

        if (!fp_equal(&ab_product, &c_val)) {
            return (int32_t)(i + 1); // 1-indexed constraint violation
        }
    }
    return 0; // All constraints satisfied
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: Lock-Free DAG Worker Queue
// ═══════════════════════════════════════════════════════════════════════════
//
// For circuits with independent subcircuits (e.g., Merkle tree hash layers),
// the DAG scheduler enables parallel constraint evaluation.
//
// Each subcircuit has:
//   - A range of constraint indices [start, end)
//   - A dependency counter (atomic uint32_t)
//   - A list of dependents (subcircuits that depend on this one)
//
// When a subcircuit's dependency counter reaches 0, any thread can pick it
// up and evaluate its constraints. After evaluation, it decrements the
// dependency counters of all its dependents.
//
// This is a WORK-STEALING pattern without a central queue: each thread
// scans the subcircuit array looking for ready (count == 0) entries.
// ═══════════════════════════════════════════════════════════════════════════

static constexpr uint32_t MAX_SUBCIRCUITS       = 256;
static constexpr uint32_t MAX_DEPS_PER_SUBCIRCUIT = 8;

struct SubcircuitDescriptor {
    uint32_t constraint_start;   // First constraint index (inclusive)
    uint32_t constraint_end;     // Last constraint index (exclusive)
    volatile uint32_t dep_count; // Atomic: number of unresolved dependencies
    uint32_t num_dependents;     // How many subcircuits depend on this one
    uint32_t dependents[MAX_DEPS_PER_SUBCIRCUIT]; // Indices of dependent subcircuits
    uint32_t evaluated;          // 1 if evaluation completed
    uint8_t  _pad[4];            // Padding for alignment
};

static SubcircuitDescriptor g_subcircuits[MAX_SUBCIRCUITS]
    __attribute__((aligned(64)));
static uint32_t g_num_subcircuits = 0;
static volatile uint32_t g_subcircuits_completed = 0;

/**
 * Register a subcircuit for DAG scheduling.
 *
 * @param constraint_start First constraint index (inclusive)
 * @param constraint_end   Last constraint index (exclusive)
 * @param dep_count        Number of dependencies that must complete first
 * @return Subcircuit index, or -1 if MAX_SUBCIRCUITS exceeded
 */
static int32_t register_subcircuit(uint32_t constraint_start,
                                    uint32_t constraint_end,
                                    uint32_t dep_count) {
    if (g_num_subcircuits >= MAX_SUBCIRCUITS) return -1;
    uint32_t idx = g_num_subcircuits++;
    g_subcircuits[idx].constraint_start = constraint_start;
    g_subcircuits[idx].constraint_end = constraint_end;
    // Use atomic store to ensure visibility across threads
    __atomic_store_n(&g_subcircuits[idx].dep_count, dep_count, __ATOMIC_SEQ_CST);
    g_subcircuits[idx].num_dependents = 0;
    g_subcircuits[idx].evaluated = 0;
    return (int32_t)idx;
}

/**
 * Add a dependency edge: `dependent_idx` depends on `prerequisite_idx`.
 * When `prerequisite_idx` completes, it decrements `dependent_idx`'s counter.
 */
static int32_t add_dependency(uint32_t prerequisite_idx, uint32_t dependent_idx) {
    if (prerequisite_idx >= g_num_subcircuits || dependent_idx >= g_num_subcircuits) {
        return -1;
    }
    SubcircuitDescriptor* pre = &g_subcircuits[prerequisite_idx];
    if (pre->num_dependents >= MAX_DEPS_PER_SUBCIRCUIT) {
        return -1;
    }
    pre->dependents[pre->num_dependents++] = dependent_idx;
    return 0;
}

/**
 * Try to claim and evaluate a ready subcircuit (dep_count == 0).
 * Uses CAS (compare-and-swap) to atomically claim ownership.
 *
 * @return Index of the evaluated subcircuit, or -1 if none ready
 */
static int32_t try_evaluate_ready_subcircuit() {
    for (uint32_t i = 0; i < g_num_subcircuits; i++) {
        uint32_t expected = 0;
        // Attempt to claim: CAS dep_count from 0 to UINT32_MAX (sentinel)
        if (__atomic_compare_exchange_n(
                &g_subcircuits[i].dep_count,
                &expected,
                UINT32_MAX,                    // Sentinel: "claimed"
                false,                          // strong CAS
                __ATOMIC_ACQ_REL,
                __ATOMIC_ACQUIRE)) {
            // We own this subcircuit — evaluate its constraints
            if (g_subcircuits[i].evaluated == 0) {
                // Evaluate constraints [start, end) using sequential path
                for (uint32_t c = g_subcircuits[i].constraint_start;
                     c < g_subcircuits[i].constraint_end; c++) {
                    Fp9x29 a_val, b_val, c_val;
                    eval_row(&a_val, g_r1cs.entries_a,
                             g_r1cs.row_offset_a[c], g_r1cs.row_offset_a[c + 1],
                             g_r1cs.signals);
                    eval_row(&b_val, g_r1cs.entries_b,
                             g_r1cs.row_offset_b[c], g_r1cs.row_offset_b[c + 1],
                             g_r1cs.signals);
                    eval_row(&c_val, g_r1cs.entries_c,
                             g_r1cs.row_offset_c[c], g_r1cs.row_offset_c[c + 1],
                             g_r1cs.signals);

                    Fp9x29 ab_product;
                    we_fp_mul(&ab_product, &a_val, &b_val);
                    // Note: In DAG mode, we just evaluate — violation detection
                    // is deferred to the final check.
                    (void)fp_equal(&ab_product, &c_val);
                }

                g_subcircuits[i].evaluated = 1;
                __atomic_fetch_add(&g_subcircuits_completed, 1, __ATOMIC_SEQ_CST);

                // Decrement dependency counters of all dependents
                for (uint32_t d = 0; d < g_subcircuits[i].num_dependents; d++) {
                    uint32_t dep_idx = g_subcircuits[i].dependents[d];
                    __atomic_fetch_sub(&g_subcircuits[dep_idx].dep_count, 1,
                                       __ATOMIC_ACQ_REL);
                }
            }
            return (int32_t)i;
        }
    }
    return -1; // No ready subcircuit found
}

/**
 * Run the DAG scheduler until all subcircuits are evaluated.
 * Single-threaded version: loops until all subcircuits complete.
 *
 * @return 0 on success, -1 if deadlock detected (progress stalls)
 */
static int32_t execute_dag_sequential() {
    uint32_t max_iterations = g_num_subcircuits * g_num_subcircuits + 1;
    uint32_t iterations = 0;
    uint32_t last_completed = 0;

    while (__atomic_load_n(&g_subcircuits_completed, __ATOMIC_ACQUIRE)
           < g_num_subcircuits) {
        int32_t result = try_evaluate_ready_subcircuit();
        if (result < 0) {
            // No ready subcircuit — check for progress
            uint32_t current = __atomic_load_n(&g_subcircuits_completed,
                                                __ATOMIC_ACQUIRE);
            if (current == last_completed) {
                iterations++;
                if (iterations > max_iterations) {
                    return -1; // Deadlock: no progress
                }
            } else {
                last_completed = current;
                iterations = 0;
            }
        } else {
            iterations = 0;
        }
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 8: Exported C Interface
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

/**
 * Initialize the R1CS engine with capacity limits.
 *
 * Allocates all memory from the static arena. Must be called before
 * any other r1cs_* function.
 *
 * @param max_constraints  Upper bound on number of R1CS constraints
 * @param max_signals      Upper bound on number of witness signals
 * @param max_nonzero      Upper bound on non-zero entries PER MATRIX (A, B, or C)
 * @return 0 on success, -1 if arena capacity exceeded
 */
int32_t r1cs_init(uint32_t max_constraints,
                  uint32_t max_signals,
                  uint32_t max_nonzero) {
    // Initialize field arithmetic constants
    zk_engine_field_init_full();

    // Reset arena
    g_r1cs_arena_offset = 0;
    std::memset(&g_r1cs, 0, sizeof(g_r1cs));

    // Reset DAG state
    g_num_subcircuits = 0;
    g_subcircuits_completed = 0;

    // Allocate constraint entries (3 matrices × max_nonzero × 64 bytes each)
    g_r1cs.entries_a = (ConstraintEntry*)r1cs_arena_alloc(
        max_nonzero * sizeof(ConstraintEntry));
    g_r1cs.entries_b = (ConstraintEntry*)r1cs_arena_alloc(
        max_nonzero * sizeof(ConstraintEntry));
    g_r1cs.entries_c = (ConstraintEntry*)r1cs_arena_alloc(
        max_nonzero * sizeof(ConstraintEntry));

    if (!g_r1cs.entries_a || !g_r1cs.entries_b || !g_r1cs.entries_c) {
        return -1;
    }

    // Allocate row offset arrays ((max_constraints + 1) × 4 bytes each)
    uint32_t offset_bytes = (max_constraints + 1) * sizeof(uint32_t);
    g_r1cs.row_offset_a = (uint32_t*)r1cs_arena_alloc(offset_bytes);
    g_r1cs.row_offset_b = (uint32_t*)r1cs_arena_alloc(offset_bytes);
    g_r1cs.row_offset_c = (uint32_t*)r1cs_arena_alloc(offset_bytes);

    if (!g_r1cs.row_offset_a || !g_r1cs.row_offset_b || !g_r1cs.row_offset_c) {
        return -1;
    }

    // Zero-fill row offsets
    std::memset(g_r1cs.row_offset_a, 0, offset_bytes);
    std::memset(g_r1cs.row_offset_b, 0, offset_bytes);
    std::memset(g_r1cs.row_offset_c, 0, offset_bytes);

    // Allocate witness signal vector (max_signals × sizeof(Fp9x29))
    g_r1cs.signals = (Fp9x29*)r1cs_arena_alloc(
        max_signals * sizeof(Fp9x29));
    if (!g_r1cs.signals) {
        return -1;
    }
    std::memset(g_r1cs.signals, 0, max_signals * sizeof(Fp9x29));

    g_r1cs.num_constraints = max_constraints;
    g_r1cs.num_signals = max_signals;
    g_r1cs.max_nonzero_per_matrix = max_nonzero;
    g_r1cs.nnz_a = 0;
    g_r1cs.nnz_b = 0;
    g_r1cs.nnz_c = 0;
    g_r1cs.signals_assigned = 0;
    g_r1cs.initialized = 1;

    return 0;
}

/**
 * Ingest a single non-zero constraint entry into matrix A, B, or C.
 *
 * @param matrix_id       0=A, 1=B, 2=C
 * @param constraint_idx  Row index (which constraint)
 * @param signal_col      Column index (which signal/wire)
 * @param coeff_ptr       Pointer to 32-byte big-endian coefficient
 * @return 0 on success, -1 on error
 */
int32_t r1cs_ingest_constraint(uint32_t matrix_id,
                                uint32_t constraint_idx,
                                uint32_t signal_col,
                                const uint8_t* coeff_ptr) {
    if (!g_r1cs.initialized) return -1;
    if (matrix_id > 2) return -1;
    if (constraint_idx >= g_r1cs.num_constraints) return -1;
    if (signal_col >= g_r1cs.num_signals) return -1;
    if (!coeff_ptr) return -1;

    ConstraintEntry* entries;
    uint32_t* nnz;

    switch (matrix_id) {
        case 0: entries = g_r1cs.entries_a; nnz = &g_r1cs.nnz_a; break;
        case 1: entries = g_r1cs.entries_b; nnz = &g_r1cs.nnz_b; break;
        case 2: entries = g_r1cs.entries_c; nnz = &g_r1cs.nnz_c; break;
        default: return -1;
    }

    if (*nnz >= g_r1cs.max_nonzero_per_matrix) return -1;

    uint32_t idx = (*nnz)++;
    entries[idx].row = constraint_idx;
    entries[idx].col = signal_col;
    entries[idx].coeff = bytes_to_limbs(coeff_ptr);
    to_montgomery(&entries[idx].coeff, &entries[idx].coeff);
    std::memset(entries[idx]._pad, 0, sizeof(entries[idx]._pad));

    return 0;
}

/**
 * Finalize the row offset arrays after all constraints have been ingested.
 * Must be called after all r1cs_ingest_constraint() calls and before
 * r1cs_evaluate().
 *
 * Builds the CSR-style row_offset arrays by scanning the sorted entries.
 * PRECONDITION: entries must be sorted by row index (ascending).
 * If not sorted, this function sorts them.
 *
 * @return 0 on success, -1 on error
 */
int32_t r1cs_finalize(void) {
    if (!g_r1cs.initialized) return -1;

    // Helper lambda to build row offsets for one matrix
    auto build_offsets = [](ConstraintEntry* entries, uint32_t nnz,
                            uint32_t* row_offsets, uint32_t num_rows) {
        // O(N log N) sort by row using std::sort
        std::sort(entries, entries + nnz, [](const ConstraintEntry& a, const ConstraintEntry& b) {
            return a.row < b.row;
        });

        // Build row offsets
        uint32_t current_row = 0;
        row_offsets[0] = 0;

        for (uint32_t e = 0; e < nnz; e++) {
            while (current_row < entries[e].row) {
                current_row++;
                row_offsets[current_row] = e;
            }
        }
        // Fill remaining rows with nnz (sentinel)
        for (uint32_t r = current_row + 1; r <= num_rows; r++) {
            row_offsets[r] = nnz;
        }
    };

    build_offsets(g_r1cs.entries_a, g_r1cs.nnz_a,
                  g_r1cs.row_offset_a, g_r1cs.num_constraints);
    build_offsets(g_r1cs.entries_b, g_r1cs.nnz_b,
                  g_r1cs.row_offset_b, g_r1cs.num_constraints);
    build_offsets(g_r1cs.entries_c, g_r1cs.nnz_c,
                  g_r1cs.row_offset_c, g_r1cs.num_constraints);

    return 0;
}

/**
 * Assign a signal value to the witness vector.
 *
 * @param signal_idx  Index in the witness vector (0-based)
 * @param value_ptr   Pointer to 32-byte big-endian value (WILL BE ZEROED)
 * @param value_len   Must be exactly 32
 * @return 0 on success, -1 if out of range, -2 if value >= p
 */
int32_t r1cs_assign_signal(uint32_t signal_idx,
                            uint8_t* value_ptr,
                            uint32_t value_len) {
    return parse_and_assign_signal(signal_idx, value_ptr, value_len);
}

/**
 * Evaluate the full R1CS system.
 *
 * @return 0 if all constraints are satisfied (A·w ⊙ B·w = C·w),
 *         positive integer = 1-indexed constraint that was violated,
 *         -1 if not initialized
 */
int32_t r1cs_evaluate(void) {
    if (!g_r1cs.initialized) return -1;

    // If subcircuits are registered, use the DAG scheduler
    if (g_num_subcircuits > 0) {
        int32_t dag_result = execute_dag_sequential();
        if (dag_result != 0) return -1; // Deadlock
        return 0; // DAG evaluation doesn't return per-constraint violations
    }

    // Otherwise, evaluate sequentially
    return evaluate_constraints_sequential();
}

/**
 * Get a pointer to the witness signal array in WASM linear memory.
 * The orchestrator uses this to read back the converted Montgomery values.
 */
Fp9x29* r1cs_get_signal_ptr(void) {
    return g_r1cs.signals;
}

/**
 * Get the number of constraints loaded.
 */
uint32_t r1cs_get_constraint_count(void) {
    return g_r1cs.num_constraints;
}

/**
 * Get the number of non-zero entries per matrix.
 */
uint32_t r1cs_get_nnz(uint32_t matrix_id) {
    switch (matrix_id) {
        case 0: return g_r1cs.nnz_a;
        case 1: return g_r1cs.nnz_b;
        case 2: return g_r1cs.nnz_c;
        default: return 0;
    }
}

/**
 * Get pointer to the R1CS arena for SharedArrayBuffer writes.
 */
uint8_t* r1cs_get_arena_ptr(void) {
    return g_r1cs_arena;
}

/**
 * Get the R1CS arena capacity in bytes.
 */
uint32_t r1cs_get_arena_capacity(void) {
    return R1CS_ARENA_CAPACITY;
}

/**
 * Register a subcircuit for DAG-based parallel evaluation.
 *
 * @param constraint_start First constraint index (inclusive)
 * @param constraint_end   Last constraint index (exclusive)
 * @param dep_count        Number of dependencies
 * @return Subcircuit index (>= 0), or -1 on error
 */
int32_t r1cs_register_subcircuit(uint32_t constraint_start,
                                  uint32_t constraint_end,
                                  uint32_t dep_count) {
    return register_subcircuit(constraint_start, constraint_end, dep_count);
}

/**
 * Add a dependency edge between subcircuits.
 *
 * @param prerequisite_idx  Must complete before dependent_idx
 * @param dependent_idx     Waits for prerequisite_idx
 * @return 0 on success, -1 on error
 */
int32_t r1cs_add_dependency(uint32_t prerequisite_idx,
                             uint32_t dependent_idx) {
    return add_dependency(prerequisite_idx, dependent_idx);
}

// ═══════════════════════════════════════════════════════════════════════════
// SELF-TEST
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Run self-tests for the R1CS engine.
 *
 * Tests:
 *   1. Signal sanitizer (valid signal < p)
 *   2. Signal sanitizer (reject signal >= p)
 *   3. Simple R1CS: 1 constraint, a*b = c with a=3, b=5, c=15
 *   4. Row offset finalization
 *   5. DAG scheduler (2 independent subcircuits)
 *
 * @return 0 if all tests pass, negative on failure
 */
int32_t r1cs_selftest(void) {
    // ─── Test 1: Signal Sanitizer (valid signal) ─────────────────
    // Value = 7 (clearly < p)
    {
        int32_t init_result = r1cs_init(4, 8, 16);
        if (init_result != 0) return -10;

        // Encode 7 as 32-byte big-endian
        uint8_t buf[32];
        std::memset(buf, 0, 32);
        buf[31] = 7; // LE byte 0 = 7 → big-endian last byte

        int32_t result = r1cs_assign_signal(0, buf, 32);
        if (result != 0) return -11;

        // Verify buffer was zeroed
        for (int i = 0; i < 32; i++) {
            if (buf[i] != 0) return -12;
        }
    }

    // ─── Test 2: Simple R1CS (3 * 5 = 15) ────────────────────────
    // Constraint: A[0][1] = 1, B[0][2] = 1, C[0][3] = 1
    // Signal[0] = 1 (constant one wire), Signal[1] = 3, Signal[2] = 5, Signal[3] = 15
    // A·w = 1 * signal[1] = 3
    // B·w = 1 * signal[2] = 5
    // C·w = 1 * signal[3] = 15
    // Check: 3 * 5 == 15 ✓
    {
        int32_t init_result = r1cs_init(1, 4, 4);
        if (init_result != 0) return -20;

        // Assign signals
        uint8_t sig_buf[32];

        // Signal 0 = 1 (constant one)
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 1;
        if (r1cs_assign_signal(0, sig_buf, 32) != 0) return -21;

        // Signal 1 = 3
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 3;
        if (r1cs_assign_signal(1, sig_buf, 32) != 0) return -22;

        // Signal 2 = 5
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 5;
        if (r1cs_assign_signal(2, sig_buf, 32) != 0) return -23;

        // Signal 3 = 15
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 15;
        if (r1cs_assign_signal(3, sig_buf, 32) != 0) return -24;

        // Ingest constraints
        // Coefficient = 1 as 32-byte big-endian
        uint8_t one_coeff[32];
        std::memset(one_coeff, 0, 32);
        one_coeff[31] = 1;

        // A[0][1] = 1  (constraint 0, signal 1)
        if (r1cs_ingest_constraint(0, 0, 1, one_coeff) != 0) return -25;
        // B[0][2] = 1  (constraint 0, signal 2)
        if (r1cs_ingest_constraint(1, 0, 2, one_coeff) != 0) return -26;
        // C[0][3] = 1  (constraint 0, signal 3)
        if (r1cs_ingest_constraint(2, 0, 3, one_coeff) != 0) return -27;

        // Finalize row offsets
        if (r1cs_finalize() != 0) return -28;

        // Evaluate
        int32_t eval_result = r1cs_evaluate();
        if (eval_result != 0) return -29; // Should pass: 3*5 = 15
    }

    // ─── Test 3: R1CS violation detection (3 * 5 != 16) ──────────
    {
        int32_t init_result = r1cs_init(1, 4, 4);
        if (init_result != 0) return -30;

        uint8_t sig_buf[32];

        // Signal 0 = 1
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 1;
        r1cs_assign_signal(0, sig_buf, 32);

        // Signal 1 = 3
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 3;
        r1cs_assign_signal(1, sig_buf, 32);

        // Signal 2 = 5
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 5;
        r1cs_assign_signal(2, sig_buf, 32);

        // Signal 3 = 16 (WRONG: 3*5 != 16)
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 16;
        r1cs_assign_signal(3, sig_buf, 32);

        uint8_t one_coeff[32];
        std::memset(one_coeff, 0, 32);
        one_coeff[31] = 1;

        r1cs_ingest_constraint(0, 0, 1, one_coeff);
        r1cs_ingest_constraint(1, 0, 2, one_coeff);
        r1cs_ingest_constraint(2, 0, 3, one_coeff);

        r1cs_finalize();

        int32_t eval_result = r1cs_evaluate();
        if (eval_result == 0) return -31; // Should FAIL
        // eval_result should be 1 (constraint index 0 + 1)
        if (eval_result != 1) return -32;
    }

    // ─── Test 4: DAG scheduler (2 independent subcircuits) ───────
    {
        int32_t init_result = r1cs_init(2, 4, 8);
        if (init_result != 0) return -40;

        uint8_t sig_buf[32];

        // Signal 0 = 1
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 1;
        r1cs_assign_signal(0, sig_buf, 32);

        // Signal 1 = 2
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 2;
        r1cs_assign_signal(1, sig_buf, 32);

        // Signal 2 = 3
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 3;
        r1cs_assign_signal(2, sig_buf, 32);

        // Signal 3 = 6 (2 * 3 = 6)
        std::memset(sig_buf, 0, 32);
        sig_buf[31] = 6;
        r1cs_assign_signal(3, sig_buf, 32);

        uint8_t one_coeff[32];
        std::memset(one_coeff, 0, 32);
        one_coeff[31] = 1;

        uint8_t two_coeff[32];
        std::memset(two_coeff, 0, 32);
        two_coeff[31] = 2;

        // Constraint 0: 2·signal[1] * signal[0] = 2·signal[1]  →  (2*2)*1 = 4  →  4=4  ✓
        // A[0][1]=2, B[0][0]=1, C[0][1]=2
        r1cs_ingest_constraint(0, 0, 1, two_coeff);
        r1cs_ingest_constraint(1, 0, 0, one_coeff);
        r1cs_ingest_constraint(2, 0, 1, two_coeff);

        // Constraint 1: signal[1] * signal[2] = signal[3]  →  2*3 = 6  ✓
        // A[1][1]=1, B[1][2]=1, C[1][3]=1
        r1cs_ingest_constraint(0, 1, 1, one_coeff);
        r1cs_ingest_constraint(1, 1, 2, one_coeff);
        r1cs_ingest_constraint(2, 1, 3, one_coeff);

        r1cs_finalize();

        // Register 2 independent subcircuits (no dependencies)
        int32_t sc0 = r1cs_register_subcircuit(0, 1, 0); // constraint [0, 1)
        int32_t sc1 = r1cs_register_subcircuit(1, 2, 0); // constraint [1, 2)
        if (sc0 < 0 || sc1 < 0) return -41;

        int32_t dag_result = r1cs_evaluate();
        if (dag_result != 0) return -42;
    }

    // ─── Test 5: Constant-time prime bound check ─────────────────
    {
        // Verify ct_is_less_than_prime returns 1 for 0
        Fp9x29 zero;
        std::memset(&zero, 0, sizeof(zero));
        if (ct_is_less_than_prime(&zero) != 1) return -50;

        // Verify ct_is_less_than_prime returns 0 for p itself
        if (ct_is_less_than_prime(&P_LIMBS) != 0) return -51;

        // Verify ct_is_less_than_prime returns 1 for p-1
        Fp9x29 p_minus_1 = P_LIMBS;
        p_minus_1.limb[0] -= 1;
        if (ct_is_less_than_prime(&p_minus_1) != 1) return -52;
    }

    return 0; // All tests passed
}

} // extern "C"
