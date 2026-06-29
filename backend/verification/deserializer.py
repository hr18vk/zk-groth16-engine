"""
═══════════════════════════════════════════════════════════════════════════════
Zero-Knowledge Engine
Groth16 Proof Deserializer & Cryptographic Validator
═══════════════════════════════════════════════════════════════════════════════

Unpacks the 256-byte big-endian proof payload emitted by the frontend
ZkProofExporter (frontend/src/lib/ZkProofExporter.ts) and validates
every elliptic curve point for curve membership and subgroup containment.

Wire Format (256 bytes, big-endian):
  Offset  | Size | Field
  --------|------|-----------------------------------
  [0..31] | 32   | π_A.x      (G1, Fq)
  [32..63]| 32   | π_A.y      (G1, Fq)
  [64..95]| 32   | π_B.x.c0   (G2, Fq component)
  [96..127]| 32  | π_B.x.c1   (G2, Fq component)
  [128..159]| 32 | π_B.y.c0   (G2, Fq component)
  [160..191]| 32 | π_B.y.c1   (G2, Fq component)
  [192..223]| 32 | π_C.x      (G1, Fq)
  [224..255]| 32 | π_C.y      (G1, Fq)

ARCHITECTURAL LAWS:
  - NO floating-point arithmetic. All values are exact integer field elements.
  - All Pydantic models use `bytes` fields with strict length validation.
  - Cryptographic validation is delegated to the Rust extension
    (bn254_engine) which releases the GIL during computation.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from typing import Final

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger("zk_engine.verification.deserializer")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BN254 Curve Constants
# ═══════════════════════════════════════════════════════════════════════════════

# BN254 base field prime p
# p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
BN254_P: Final[int] = 0x30644E72E131A029B85045B68181585D97816A916871CA8D3C208C16D87CFD47

# BN254 scalar field order r  (same value as in vector_prover.cpp P_BYTES)
# r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
BN254_R: Final[int] = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001

# Proof payload size in bytes
PROOF_SIZE: Final[int] = 256

# Coordinate size in bytes (32 bytes = 256 bits for BN254 field elements)
COORD_SIZE: Final[int] = 32


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Pydantic Type-Safe Models
# ═══════════════════════════════════════════════════════════════════════════════

class G1Point(BaseModel):
    """
    A point on the BN254 G₁ curve in affine coordinates.
    Both coordinates are 32-byte big-endian field elements.
    """
    x: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)
    y: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)

    @field_validator("x", "y")
    @classmethod
    def validate_field_element(cls, v: bytes) -> bytes:
        """Validate that the coordinate is a valid Fq element (< p)."""
        val = int.from_bytes(v, byteorder="big")
        if val >= BN254_P:
            raise ValueError(
                f"Coordinate value {val:#066x} >= BN254 base field prime p"
            )
        return v

    model_config = {"frozen": True}


class G2Point(BaseModel):
    """
    A point on the BN254 G₂ twisted curve in affine coordinates.
    G₂ lives over Fq2 = Fq[u] / (u² + 1).
    Each Fq2 element has two Fq components (c0, c1).
    """
    x_c0: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)
    x_c1: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)
    y_c0: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)
    y_c1: bytes = Field(..., min_length=COORD_SIZE, max_length=COORD_SIZE)

    @field_validator("x_c0", "x_c1", "y_c0", "y_c1")
    @classmethod
    def validate_field_element(cls, v: bytes) -> bytes:
        """Validate that the coordinate is a valid Fq element (< p)."""
        val = int.from_bytes(v, byteorder="big")
        if val >= BN254_P:
            raise ValueError(
                f"Coordinate value {val:#066x} >= BN254 base field prime p"
            )
        return v

    model_config = {"frozen": True}


class DeserializedProof(BaseModel):
    """
    A fully deserialized and field-validated Groth16 proof.
    All coordinates have been checked for field membership (< p).
    Curve and subgroup checks are performed separately by the Rust engine.
    """
    pi_a: G1Point
    pi_b: G2Point
    pi_c: G1Point
    raw_bytes: bytes = Field(
        ..., min_length=PROOF_SIZE, max_length=PROOF_SIZE,
        description="Original 256-byte proof payload (retained for Rust FFI)"
    )

    model_config = {"frozen": True}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Deserialization Engine
# ═══════════════════════════════════════════════════════════════════════════════

def deserialize_proof(payload: bytes) -> DeserializedProof:
    """
    Deserialize a 256-byte big-endian Groth16 proof payload.

    Performs:
      1. Length validation (exactly 256 bytes)
      2. Field membership checks for all 8 coordinates (< p)
      3. Structural packing into type-safe Pydantic models

    Does NOT perform curve equation or subgroup checks — those are
    delegated to the Rust extension (bn254_engine.verify_groth16)
    which handles them in constant time with GIL released.

    Args:
        payload: Exactly 256 bytes of big-endian proof data.

    Returns:
        DeserializedProof with pi_a (G1), pi_b (G2), pi_c (G1).

    Raises:
        ValueError: If payload length is wrong or any coordinate >= p.
    """
    if len(payload) != PROOF_SIZE:
        raise ValueError(
            f"Proof payload must be exactly {PROOF_SIZE} bytes, got {len(payload)}"
        )

    logger.debug("Deserializing 256-byte Groth16 proof payload")

    # ── Extract G₁ point π_A (offset 0..63) ───────────────────────────────
    pi_a = G1Point(
        x=payload[0:32],
        y=payload[32:64],
    )

    # ── Extract G₂ point π_B (offset 64..191) ─────────────────────────────
    pi_b = G2Point(
        x_c0=payload[64:96],
        x_c1=payload[96:128],
        y_c0=payload[128:160],
        y_c1=payload[160:192],
    )

    # ── Extract G₁ point π_C (offset 192..255) ────────────────────────────
    pi_c = G1Point(
        x=payload[192:224],
        y=payload[224:256],
    )

    proof = DeserializedProof(
        pi_a=pi_a,
        pi_b=pi_b,
        pi_c=pi_c,
        raw_bytes=payload,
    )

    logger.info(
        "Proof deserialized: π_A.x=%s... π_B.x.c0=%s... π_C.x=%s...",
        payload[0:4].hex(),
        payload[64:68].hex(),
        payload[192:196].hex(),
    )

    return proof


def validate_public_input(signal: bytes) -> None:
    """
    Validate a single public input (32-byte big-endian scalar field element).

    The value must be < r (the BN254 scalar field order).

    Args:
        signal: 32 bytes, big-endian.

    Raises:
        ValueError: If length != 32 or value >= r.
    """
    if len(signal) != COORD_SIZE:
        raise ValueError(
            f"Public input must be exactly {COORD_SIZE} bytes, got {len(signal)}"
        )
    val = int.from_bytes(signal, byteorder="big")
    if val >= BN254_R:
        raise ValueError(
            f"Public input {val:#066x} >= BN254 scalar field order r"
        )
