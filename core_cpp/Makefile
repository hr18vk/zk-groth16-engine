# Zero-Knowledge Engine
# Emscripten Build Chain: C++ -> WASM
#
# Build:   make
# Clean:   make clean
# Debug:   make DEBUG=1
#
# Output:  build/prover_engine.wasm
#          build/prover_engine.js (Emscripten glue code)
#
# REQUIREMENTS: Emscripten SDK must be installed and activated.
#   source /workspaces/zk_engine/emsdk/emsdk_env.sh

CC = emcc
CXX = em++

SRC_DIR = src
BUILD_DIR = build

# Source files
SOURCES = $(SRC_DIR)/prover_engine.cpp

# Output
TARGET = $(BUILD_DIR)/prover_engine.js

# Compiler Flags
# -Oz               Extreme size optimization (critical for mobile download)
# -flto             Link-time optimization (dead code elimination across TUs)
# -s INITIAL_MEMORY  256MB initial WASM linear memory
# -s ALLOW_MEMORY_GROWTH  Allow growth beyond initial (for large circuits)
# -s MAXIMUM_MEMORY  512MB cap to prevent runaway allocation
# -s SHARED_MEMORY=1  Enable SharedArrayBuffer for pthread support
# -s EXPORTED_FUNCTIONS  Only export our C interface (no bloat)
# -s EXPORTED_RUNTIME_METHODS  Export ccall/cwrap for JS interop
# -s MODULARIZE=1   Wrap in a factory function (required for Web Worker loading)
# -s EXPORT_NAME    Name of the factory function
# -s ENVIRONMENT    Target Web Worker environment only
# -s FILESYSTEM=0   Disable Emscripten's virtual filesystem (we use OPFS)
# -s ASSERTIONS=0   Disable runtime assertions in release (saves ~50KB)
# -s MALLOC=emmalloc  Smaller allocator than dlmalloc (saves ~10KB)

CXXFLAGS = -std=c++17 -Oz -flto -msimd128

LDFLAGS = \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s SHARED_MEMORY=1 \
	-s PTHREAD_POOL_SIZE=0 \
	-s EXPORTED_FUNCTIONS='["_get_memory_ptr","_get_buffer_capacity","_get_proof_ptr","_get_proof_status","_get_progress","_reset_engine","_ingest_zkey_chunk","_get_zkey_bytes_loaded","_compute_proof","_init_g1_arena_export","_allocate_g1_arena_export","_get_g1_arena_remaining_export","_proof_assembly_selftest","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]' \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createProverEngine' \
	-s ENVIRONMENT='worker' \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=0 \
	-s MALLOC=emmalloc \
	-s EXPORT_ES6=1 \
	-s WASM=1

ifdef DEBUG
	CXXFLAGS = -std=c++17 -O0 -g3
	LDFLAGS += -s ASSERTIONS=2
endif

# Targets

.PHONY: all clean test-simd test-pthread test-proof-assembly

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SOURCES) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)
	@echo ""
	@echo "BUILD SUCCESSFUL"
	@echo "Output: $(BUILD_DIR)/prover_engine.wasm"
	@echo "Output: $(BUILD_DIR)/prover_engine.js"
	@ls -lh $(BUILD_DIR)/prover_engine.wasm
	@ls -lh $(BUILD_DIR)/prover_engine.js

clean:
	rm -rf $(BUILD_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 2.1 — SIMD128 Verification Harness
# ═══════════════════════════════════════════════════════════════════════════════
# Build + auto-execute the standalone verification harness in Node.js.
# Usage: make test-simd
#
# This target uses DIFFERENT flags than the production prover:
#   - No MODULARIZE (has main(), not a factory)
#   - No SHARED_MEMORY (single-threaded test)
#   - No ENVIRONMENT='worker' (runs in Node.js)
#   - Added -msimd128 (enables WASM SIMD128 instructions)
# ═══════════════════════════════════════════════════════════════════════════════

HARNESS_SRC = $(SRC_DIR)/verification_harness.cpp
HARNESS_OUT = $(BUILD_DIR)/verification_harness.js

HARNESS_CXXFLAGS = -std=c++17 -Oz -msimd128

HARNESS_LDFLAGS = \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=0 \
	-s STANDALONE_WASM=0 \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1

test-simd: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling SIMD128 Verification Harness..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(HARNESS_CXXFLAGS) $(HARNESS_SRC) -o $(HARNESS_OUT) $(HARNESS_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/verification_harness.wasm
	@ls -lh $(BUILD_DIR)/verification_harness.js
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing in Node.js..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} $(HARNESS_OUT)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Verification harness execution complete."
	@echo "═══════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 2.3 — Pthread Orchestration Harness
# ═══════════════════════════════════════════════════════════════════════════════
# Build + auto-execute the multi-threaded verification harness in Node.js.
# Usage: make test-pthread
#
# This target uses DIFFERENT flags than both production and test-simd:
#   - MODULARIZE=1 (factory function, loaded by the JS harness)
#   - SHARED_MEMORY=1 (enables SharedArrayBuffer)
#   - -pthread (enables Emscripten pthreads)
#   - PTHREAD_POOL_SIZE=4 (pre-spawns 4 Web Worker threads)
#   - PROXY_TO_PTHREAD=1 (proxies main() to a worker thread)
#   - STACK_SIZE=5242880 (5MB per-thread stack to handle BN254 math)
#   - ENVIRONMENT='node,worker' (runs in Node.js with worker threads)
#   - -msimd128 (enables WASM SIMD128 instructions for Montgomery CIOS)
#
# CRITICAL NODE.JS FLAGS:
#   --experimental-wasm-threads: Enables WASM threading proposal
#   --experimental-wasm-bulk-memory: Required by Emscripten's pthread impl
# ═══════════════════════════════════════════════════════════════════════════════

PTHREAD_SRC = $(SRC_DIR)/vector_prover.cpp
PTHREAD_OUT = $(BUILD_DIR)/pthread_prover.js
PTHREAD_HARNESS = test/pthread_harness.mjs

PTHREAD_CXXFLAGS = -std=c++17 -Oz -msimd128 -pthread

PTHREAD_LDFLAGS = \
	-pthread \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s SHARED_MEMORY=1 \
	-s PTHREAD_POOL_SIZE=4 \
	-s PROXY_TO_PTHREAD=1 \
	-s STACK_SIZE=5242880 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createPthreadProver' \
	-s ENVIRONMENT='node,worker' \
	-s EXPORTED_FUNCTIONS='["_zk_engine_field_init_full","_get_shared_arena_ptr","_get_shared_arena_capacity","_get_sync_flag_ptr","_get_orchestration_result","_orchestrate_parallel_proof_matrix","_start_orchestration","_get_p_limbs","_get_p_prime_0","_malloc","_free","_main"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","PThread"]' \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1

test-pthread: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling Pthread Orchestration Engine..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(PTHREAD_CXXFLAGS) $(PTHREAD_SRC) -o $(PTHREAD_OUT) $(PTHREAD_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/pthread_prover.wasm
	@ls -lh $(BUILD_DIR)/pthread_prover.js
	@ls -lh $(BUILD_DIR)/pthread_prover.worker.js 2>/dev/null || true
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing in Node.js (with threads)..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} $(PTHREAD_HARNESS)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Pthread orchestration harness execution complete."
	@echo "═══════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 3.1 — MSM & NTT Layout Verification
# ═══════════════════════════════════════════════════════════════════════════════
# Build the prover_engine.cpp (which now includes vector_prover.cpp via STU)
# with SIMD128 enabled to verify that FieldElement, JacobianPoint, and the
# G1 arena allocator compile and link correctly.
#
# Usage: make test-msm
#
# This target compiles in NODE environment without pthreads to verify
# the struct layouts and arena allocation logic independently.
# ═══════════════════════════════════════════════════════════════════════════════

MSM_SRC = $(SRC_DIR)/prover_engine.cpp
MSM_OUT = $(BUILD_DIR)/msm_layout_test.js

MSM_CXXFLAGS = -std=c++17 -Oz -msimd128

MSM_LDFLAGS = \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createMsmTest' \
	-s ENVIRONMENT='node' \
	-s EXPORTED_FUNCTIONS='["_get_memory_ptr","_get_buffer_capacity","_get_proof_ptr","_get_proof_status","_get_progress","_reset_engine","_ingest_zkey_chunk","_get_zkey_bytes_loaded","_compute_proof","_init_g1_arena_export","_allocate_g1_arena_export","_get_g1_arena_remaining_export","_execute_msm_export","_execute_ntt_export","_msm_ntt_selftest","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]'

test-msm: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling MSM & NTT Test (Phase 3.2)..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(MSM_CXXFLAGS) $(MSM_SRC) -o $(MSM_OUT) $(MSM_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/msm_layout_test.wasm
	@ls -lh $(BUILD_DIR)/msm_layout_test.js
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing Phase 3.2 self-test in Node.js..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} test/msm_ntt_harness.mjs
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  MSM & NTT Phase 3.2 verification complete."
	@echo "═══════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 3.2 — Multi-Threaded MSM Verification
# ═══════════════════════════════════════════════════════════════════════════════
# Usage: make test-msm-pthread
# ═══════════════════════════════════════════════════════════════════════════════

MSM_PTHREAD_SRC = $(SRC_DIR)/prover_engine.cpp
MSM_PTHREAD_OUT = $(BUILD_DIR)/msm_pthread_test.js

MSM_PTHREAD_CXXFLAGS = -std=c++17 -Oz -msimd128 -pthread

MSM_PTHREAD_LDFLAGS = \
	-pthread \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s SHARED_MEMORY=1 \
	-s PTHREAD_POOL_SIZE=4 \
	-s PROXY_TO_PTHREAD=1 \
	-s STACK_SIZE=5242880 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createMsmPthreadTest' \
	-s ENVIRONMENT='node,worker' \
	-s EXPORTED_FUNCTIONS='["_get_memory_ptr","_get_buffer_capacity","_get_proof_ptr","_get_proof_status","_get_progress","_reset_engine","_ingest_zkey_chunk","_get_zkey_bytes_loaded","_compute_proof","_init_g1_arena_export","_allocate_g1_arena_export","_get_g1_arena_remaining_export","_execute_msm_export","_execute_ntt_export","_msm_ntt_selftest","_malloc","_free","_main"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8","PThread"]' \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1

test-msm-pthread: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling MSM Pthread Test (Phase 3.2)..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(MSM_PTHREAD_CXXFLAGS) $(MSM_PTHREAD_SRC) -o $(MSM_PTHREAD_OUT) $(MSM_PTHREAD_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/msm_pthread_test.wasm
	@ls -lh $(BUILD_DIR)/msm_pthread_test.js
	@ls -lh $(BUILD_DIR)/msm_pthread_test.worker.js 2>/dev/null || true
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing MSM Pthread Test in Node.js..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} test/msm_ntt_harness.mjs --pthread
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  MSM Pthread Phase 3.2 verification complete."
	@echo "═══════════════════════════════════════════════════════════"


# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 3.3 — R1CS Witness Engine Verification
# ═══════════════════════════════════════════════════════════════════════════════
# Build + auto-execute the witness_engine.cpp self-tests in Node.js.
# Usage: make test-r1cs
#
# This target compiles witness_engine.cpp (which includes vector_prover.cpp
# via STU) with SIMD128 enabled. It exports the R1CS API functions and
# runs the self-test harness.
# ═══════════════════════════════════════════════════════════════════════════════

R1CS_SRC = $(SRC_DIR)/witness_engine.cpp
R1CS_OUT = $(BUILD_DIR)/r1cs_witness_test.js

R1CS_CXXFLAGS = -std=c++17 -Oz -msimd128

R1CS_LDFLAGS = \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createR1csTest' \
	-s ENVIRONMENT='node' \
	-s EXPORTED_FUNCTIONS='["_r1cs_init","_r1cs_ingest_constraint","_r1cs_assign_signal","_r1cs_finalize","_r1cs_evaluate","_r1cs_get_signal_ptr","_r1cs_get_constraint_count","_r1cs_get_nnz","_r1cs_get_arena_ptr","_r1cs_get_arena_capacity","_r1cs_register_subcircuit","_r1cs_add_dependency","_r1cs_selftest","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPU8"]'

test-r1cs: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling R1CS Witness Engine (Phase 3.3)..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(R1CS_CXXFLAGS) $(R1CS_SRC) -o $(R1CS_OUT) $(R1CS_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/r1cs_witness_test.wasm
	@ls -lh $(BUILD_DIR)/r1cs_witness_test.js
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing Phase 3.3 self-test in Node.js..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} test/r1cs_witness_harness.mjs
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  R1CS Witness Engine Phase 3.3 verification complete."
	@echo "═══════════════════════════════════════════════════════════"


# ═══════════════════════════════════════════════════════════════════════════════
# ZK Engine Phase 3.4 — Proof Assembly Self-Test
# ═══════════════════════════════════════════════════════════════════════════════
# Build + auto-execute the proof assembly self-tests in Node.js.
# Usage: make test-proof-assembly
#
# This target compiles prover_engine.cpp (which now includes proof_assembly.cpp
# via inline STU) with SIMD128 enabled. It exports only the self-test and
# field init functions, then runs the JS harness.
# ═══════════════════════════════════════════════════════════════════════════════

PA_SRC = $(SRC_DIR)/prover_engine.cpp
PA_OUT = $(BUILD_DIR)/proof_assembly_test.js

PA_CXXFLAGS = -std=c++17 -Oz -msimd128

PA_LDFLAGS = \
	-s INITIAL_MEMORY=268435456 \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s MAXIMUM_MEMORY=536870912 \
	-s FILESYSTEM=0 \
	-s ASSERTIONS=1 \
	-s WASM=1 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createProofAssemblyTest' \
	-s ENVIRONMENT='node' \
	-s EXPORTED_FUNCTIONS='["_zk_engine_field_init_full","_proof_assembly_selftest","_malloc","_free"]' \
	-s EXPORTED_RUNTIME_METHODS='["ccall","HEAPU8"]' \
	-s EXPORT_ES6=1

test-proof-assembly: | $(BUILD_DIR)
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Compiling Proof Assembly Engine (Phase 3.4)..."
	@echo "═══════════════════════════════════════════════════════════"
	$(CXX) $(PA_CXXFLAGS) $(PA_SRC) -o $(PA_OUT) $(PA_LDFLAGS)
	@echo ""
	@echo "  Compilation successful."
	@ls -lh $(BUILD_DIR)/proof_assembly_test.wasm
	@ls -lh $(BUILD_DIR)/proof_assembly_test.js
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Executing Phase 3.4 self-test in Node.js..."
	@echo "═══════════════════════════════════════════════════════════"
	@echo ""
	$${EMSDK_NODE:-node} test/proof_assembly_harness.mjs
	@echo ""
	@echo "═══════════════════════════════════════════════════════════"
	@echo "  Proof Assembly Phase 3.4 verification complete."
	@echo "═══════════════════════════════════════════════════════════"

