/**
 * Zero-Knowledge Engine
 * useZkProver React Hook
 *
 * Manages the ZK Orchestrator Web Worker lifecycle from the main thread.
 * Provides reactive state for UI binding.
 *
 * ARCHITECTURAL LAW: This hook NEVER performs any cryptographic computation.
 * It only sends messages to the Worker and reads responses.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

// ─── Types ─────────────────────────────────────────────────────────────────

type Phase =
  | 'idle'
  | 'initializing'
  | 'downloading'
  | 'cached'
  | 'computing'
  | 'done'
  | 'error';

interface ProofResult {
  publicInputsHash: string;
  nullifierHash: string;
  proofCommitmentMatrix: {
    pi_a: [string, string, string];
    pi_b: [[string, string], [string, string], [string, string]];
    pi_c: [string, string, string];
  };
  computeTimeMs: number;
}

interface ZkProverState {
  phase: Phase;
  statusMessage: string;
  downloadProgress: number;
  computeProgress: number;
  proof: ProofResult | null;
  error: { code: string; message: string; details?: string } | null;
  isReady: boolean;
}

interface ZkProverActions {
  initialize: () => void;
  generateProof: (publicInputs: number[], zkeyUrl: string, zkeyHash: string) => void;
  reset: () => void;
}

// ─── Hook ──────────────────────────────────────────────────────────────────

export function useZkProver(): ZkProverState & ZkProverActions {
  const workerRef = useRef<Worker | null>(null);
  const [phase, setPhase] = useState<Phase>('idle');
  const [statusMessage, setStatusMessage] = useState('');
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [computeProgress, setComputeProgress] = useState(0);
  const [proof, setProof] = useState<ProofResult | null>(null);
  const [error, setError] = useState<ZkProverState['error']>(null);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/zkOrchestrator.worker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data;

      switch (msg.type) {
        case 'status': {
          const status = msg.payload;
          setPhase(status.phase);
          setStatusMessage(status.message);
          if (status.phase === 'initializing' && status.message.includes('ready')) {
            setIsReady(true);
          }
          break;
        }
        case 'progress': {
          const progress = msg.payload;
          if (progress.phase === 'download') {
            setDownloadProgress(Math.max(0, progress.percent));
          } else if (progress.phase === 'compute') {
            setComputeProgress(Math.max(0, progress.percent));
          }
          break;
        }
        case 'proof': {
          setProof(msg.payload);
          setPhase('done');
          break;
        }
        case 'error': {
          setError(msg.payload);
          setPhase('error');
          break;
        }
      }
    };

    worker.onerror = (event: ErrorEvent) => {
      setError({
        code: 'WORKER_CRASH',
        message: 'ZK Worker crashed unexpectedly.',
        details: event.message,
      });
      setPhase('error');
    };

    workerRef.current = worker;

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  const initialize = useCallback(() => {
    if (!workerRef.current) return;
    setPhase('initializing');
    setError(null);
    workerRef.current.postMessage({ type: 'init' });
  }, []);

  const generateProof = useCallback(
    (publicInputs: number[], zkeyUrl: string, zkeyHash: string) => {
      if (!workerRef.current) return;
      setPhase('initializing');
      setError(null);
      setProof(null);
      setDownloadProgress(0);
      setComputeProgress(0);
      workerRef.current.postMessage({
        type: 'prove',
        payload: { publicInputs, zkeyUrl, zkeyHash },
      });
    },
    []
  );

  const reset = useCallback(() => {
    setPhase('idle');
    setStatusMessage('');
    setDownloadProgress(0);
    setComputeProgress(0);
    setProof(null);
    setError(null);
  }, []);

  return {
    phase,
    statusMessage,
    downloadProgress,
    computeProgress,
    proof,
    error,
    isReady,
    initialize,
    generateProof,
    reset,
  };
}
