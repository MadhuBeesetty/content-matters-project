"""Prover9 theorem prover interface.

Produces syntactically valid Prover9 input and parses real success/failure
markers. Prover9 input requires assumptions and goals in separate blocks:

    formulas(assumptions).
    <formula>.
    end_of_list.
    formulas(goals).
    <formula>.
    end_of_list.

A successful run prints a "============================== PROOF =====..."
section and exits with "Exiting with N proof(s)". A failed search prints
"SEARCH FAILED" / exits "without proof". The previous version of this file
emitted `formula({theorem}).` (invalid syntax) and treated the mere presence
of the substring "proof" as success — Prover9 prints that word in its banner
even on failure, so every call reported success.
"""

import re
import subprocess
from typing import Optional, Tuple

PROOF_SECTION = "============================== PROOF"


class Prover9Verifier:
    """Interface to the Prover9 theorem prover."""

    def __init__(self, prover9_path: Optional[str] = None):
        self.prover9_path = prover9_path or "prover9"

    def is_installed(self) -> bool:
        try:
            result = subprocess.run(
                [self.prover9_path, "-x"],  # no valid input -> usage error is fine
                input=b"", capture_output=True, timeout=5,
            )
            # Any response (even a usage error) means the binary exists
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def build_input(axioms_fol: list[str], goal_fol: str) -> str:
        """Build a valid Prover9 input file from FOL strings."""
        assumptions = "\n".join(f"{a.rstrip('.')}." for a in axioms_fol)
        return (
            "formulas(assumptions).\n"
            f"{assumptions}\n"
            "end_of_list.\n"
            "formulas(goals).\n"
            f"{goal_fol.rstrip('.')}.\n"
            "end_of_list.\n"
        )

    def prove(self, axioms_fol: list[str], goal_fol: str,
              timeout: int = 10) -> Tuple[bool, str]:
        """Ask Prover9 whether `goal_fol` follows from `axioms_fol`.

        Returns:
            (proved, raw_output). `proved` is True only when Prover9 actually
            emitted a proof section.
        """
        input_text = self.build_input(axioms_fol, goal_fol)
        try:
            result = subprocess.run(
                [self.prover9_path],
                input=input_text.encode(),
                capture_output=True,
                timeout=timeout,
            )
            output = result.stdout.decode(errors="replace")
            proved = (
                PROOF_SECTION in output
                and "SEARCH FAILED" not in output
                and not re.search(r"Exiting with .*fail", output)
            )
            return proved, output
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except FileNotFoundError:
            return False, "Prover9 not installed — see scripts/02_setup_prover9.sh"

    def verify_proof(self, axioms: str, theorem: str, proof: str,
                     timeout: int = 10) -> Tuple[bool, str]:
        """Backward-compatible wrapper.

        NOTE: like Logic-LM's pipeline, this verifies the *conclusion* by
        re-proving the theorem from the axioms; it does not yet check the
        individual steps of the candidate `proof` trace. Step-level checking
        is future work.
        """
        return self.prove([axioms], theorem, timeout=timeout)
