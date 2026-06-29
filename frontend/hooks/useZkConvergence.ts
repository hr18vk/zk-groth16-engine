/**
 * ═══════════════════════════════════════════════════════════════════════════════
 * Zero-Knowledge Engine
 * useZK EngineConvergence — End-to-End ZK Convergence Hook
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * Orchestrates the full client-to-server ZK pipeline:
 *   1. Spawns prover.worker.ts as a module worker
 *   2. Sends FORGE command with circuit WASM URL, .zkey IndexedDB key, and inputs
 *   3. Tracks FSM states: IDLE → FORGING → TRANSMITTING → VERIFIED / FAILED
 *   4. On FORGE_SUCCESS: immediately POSTs the 256-byte hex proof + public
 *      signals to POST /api/verify-proof
 *   5. Terminates the worker in a finally block to prevent memory leaks
 *
 * ARCHITECTURAL LAWS:
 *   - The worker is single-use: instantiated on forge, terminated on completion
 *   - The .zkey buffer NEVER touches the main thread
 *   - Worker termination is guaranteed via finally blocks and useEffect cleanup
 *   - The hook is fully reactive: all state transitions trigger re-renders
 * ═══════════════════════════════════════════════════════════════════════════════
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// ─── Types ──────────────────────────────────────────────────────────────────

/**
 * Strict FSM states — no implicit transitions allowed.
 *
 * IDLE         → User hasn't started / has reset
 * FORGING      → prover.worker.ts is running fullProve
 * TRANSMITTING → Proof generated, POSTing to /api/verify-proof
 * VERIFIED     → Backend confirmed proof is cryptographically valid
 * FAILED       → Any stage failed (forge error, network error, invalid proof)
 */
export type ConvergencePhase =
  | 'IDLE'
  | 'FORGING'
  | 'TRANSMITTING'
  | 'VERIFIED'
  | 'FAILED';

export interface ForgeProgress {
  phase: string;
  message: string;
  progress?: number; // 0-100
}

export interface VerificationResult {
  /** True if the Rust backend confirmed the proof is valid */
  valid: boolean;
  /** Verification time on the server in ms (Rust pairing computation) */
  serverVerificationTimeMs: number;
  /** End-to-end time from forge start to verification response in ms */
  totalPipelineTimeMs: number;
  /** Client-side proof generation time in ms */
  clientForgeTimeMs: number;
  /** The 512-char hex proof string that was verified */
  proofHex: string;
  /** The hex-encoded public signals */
  publicSignalsHex: string[];
}

export interface ConvergenceError {
  code: string;
  message: string;
  details?: string;
  /** Which stage failed */
  stage: 'FORGE' | 'TRANSMIT' | 'VERIFY';
}

export interface UseZK EngineConvergenceState {
  phase: ConvergencePhase;
  progress: ForgeProgress | null;
  result: VerificationResult | null;
  error: ConvergenceError | null;
}

export interface UseZK EngineConvergenceActions {
  /**
   * Initiates the full convergence pipeline:
   *   1. Spawn worker → fullProve → serialize proof
   *   2. POST to /api/verify-proof
   *   3. Return verification result
   *
   * @param circuitWasmUrl  URL to the circuit .wasm file
   * @param zkeyKey         localforage key for the .zkey ArrayBuffer
   * @param inputs          Circuit input signals
   */
  forgeAndVerify: (
    circuitWasmUrl: string,
    zkeyKey: string,
    inputs: Record<string, string | number | bigint>,
  ) => void;

  /** Resets all state to IDLE */
  reset: () => void;
}

// ─── API Configuration ──────────────────────────────────────────────────────

/**
 * Resolves the API base URL.
 * In development, the Vite proxy (vite.config.ts line 11-14) forwards /api
 * to http://127.0.0.1:8000, so we use '' (relative path).
 * In production, VITE_API_URL points to the Cloudflare Gateway.
 */
const API_BASE = import.meta.env.VITE_API_URL || '';

// ─── Hook ───────────────────────────────────────────────────────────────────

export function useZK EngineConvergence(): UseZK EngineConvergenceState & UseZK EngineConvergenceActions {
  const [phase, setPhase] = useState<ConvergencePhase>('IDLE');
  const [progress, setProgress] = useState<ForgeProgress | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [error, setError] = useState<ConvergenceError | null>(null);

  // Ref to the active worker — used for cleanup
  const workerRef = useRef<Worker | null>(null);

  // Ref to track if the component is still mounted
  const mountedRef = useRef(true);

  // Cleanup on unmount: terminate any active worker
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, []);

  /**
   * Terminates the worker safely. Called in finally blocks.
   * Idempotent — safe to call multiple times.
   */
  const terminateWorker = useCallback(() => {
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }
  }, []);

  /**
   * POST the proof to the backend verification endpoint.
   * Returns the parsed VerifyProofResponse.
   */
  const transmitProof = useCallback(
    async (proofHex: string, publicSignalsHex: string[]): Promise<{
      valid: boolean;
      verification_time_ms: number;
      error: string | null;
    }> => {
      const response = await fetch(`${API_BASE}/api/verify-proof`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          proof: proofHex,
          public_inputs: publicSignalsHex,
        }),
      });

      if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(
          `HTTP ${response.status}: ${text || response.statusText}`
        );
      }

      return response.json();
    },
    [],
  );

  /**
   * The main convergence pipeline.
   */
  const forgeAndVerify = useCallback(
    (
      circuitWasmUrl: string,
      zkeyKey: string,
      inputs: Record<string, string | number | bigint>,
    ) => {
      // Prevent concurrent forgeries
      if (phase === 'FORGING' || phase === 'TRANSMITTING') return;

      // Reset state
      setPhase('FORGING');
      setProgress(null);
      setResult(null);
      setError(null);

      const pipelineStart = performance.now();

      // ── Spawn worker ──────────────────────────────────────────────
      const worker = new Worker(
        new URL('../workers/prover.worker.ts', import.meta.url),
        { type: 'module' },
      );
      workerRef.current = worker;

      // ── Worker message handler ─────────────────────────────────────
      worker.onmessage = async (event: MessageEvent) => {
        if (!mountedRef.current) {
          terminateWorker();
          return;
        }

        const msg = event.data;

        switch (msg.type) {
          case 'FORGE_STATUS': {
            setProgress({
              phase: msg.payload.phase,
              message: msg.payload.message,
              progress: msg.payload.progress,
            });
            break;
          }

          case 'FORGE_SUCCESS': {
            const { proofHex, publicSignalsHex, computeTimeMs } = msg.payload;

            // ── Immediately terminate the worker ────────────────────
            // The .zkey buffer dies with the worker's realm.
            terminateWorker();

            // ── Transition to TRANSMITTING ──────────────────────────
            if (!mountedRef.current) return;
            setPhase('TRANSMITTING');
            setProgress({
              phase: 'TRANSMITTING',
              message: 'Sending proof to verification backend...',
            });

            try {
              const verifyResult = await transmitProof(proofHex, publicSignalsHex);

              if (!mountedRef.current) return;

              const totalPipelineTimeMs = Math.round(performance.now() - pipelineStart);

              if (verifyResult.valid) {
                setPhase('VERIFIED');
                setResult({
                  valid: true,
                  serverVerificationTimeMs: verifyResult.verification_time_ms,
                  totalPipelineTimeMs,
                  clientForgeTimeMs: computeTimeMs,
                  proofHex,
                  publicSignalsHex,
                });
              } else {
                setPhase('FAILED');
                setError({
                  code: 'PROOF_INVALID',
                  message: verifyResult.error || 'Proof rejected by the verification backend.',
                  stage: 'VERIFY',
                });
              }
            } catch (transmitErr) {
              if (!mountedRef.current) return;

              setPhase('FAILED');
              setError({
                code: 'TRANSMIT_FAILED',
                message: transmitErr instanceof Error
                  ? transmitErr.message
                  : 'Network error during proof transmission.',
                details: transmitErr instanceof Error ? transmitErr.stack : undefined,
                stage: 'TRANSMIT',
              });
            }
            break;
          }

          case 'FORGE_ERROR': {
            terminateWorker();

            if (!mountedRef.current) return;

            setPhase('FAILED');
            setError({
              code: msg.payload.code,
              message: msg.payload.message,
              details: msg.payload.details,
              stage: 'FORGE',
            });
            break;
          }
        }
      };

      // ── Worker crash handler ────────────────────────────────────────
      worker.onerror = (event: ErrorEvent) => {
        terminateWorker();

        if (!mountedRef.current) return;

        setPhase('FAILED');
        setError({
          code: 'WORKER_CRASH',
          message: 'Prover worker crashed unexpectedly.',
          details: event.message,
          stage: 'FORGE',
        });
      };

      // ── Send FORGE command ──────────────────────────────────────────
      worker.postMessage({
        type: 'FORGE',
        payload: { circuitWasmUrl, zkeyKey, inputs },
      });
    },
    [phase, terminateWorker, transmitProof],
  );

  const reset = useCallback(() => {
    terminateWorker();
    setPhase('IDLE');
    setProgress(null);
    setResult(null);
    setError(null);
  }, [terminateWorker]);

  return {
    phase,
    progress,
    result,
    error,
    forgeAndVerify,
    reset,
  };
}
