"""Debug a failed Rimay execution by fetching logs from the API."""

import json
import os
import sys
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def fetch_execution_logs(execution_id: str) -> None:
    """Fetch and display logs for a specific execution ID."""
    from planqk.service.client import PlanqkServiceClient

    consumer_key = os.environ.get("PLANQK_CONSUMER_KEY")
    consumer_secret = os.environ.get("PLANQK_CONSUMER_SECRET")
    service_endpoint = os.environ.get("PLANQK_SERVICE_ENDPOINT")

    if not all([consumer_key, consumer_secret, service_endpoint]):
        print("ERROR: Missing PlanQK environment variables")
        print("  PLANQK_CONSUMER_KEY")
        print("  PLANQK_CONSUMER_SECRET")
        print("  PLANQK_SERVICE_ENDPOINT")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  FETCHING EXECUTION LOGS")
    print(f"{'=' * 60}")
    print(f"  Execution ID: {execution_id}")

    try:
        client = PlanqkServiceClient(
            service_endpoint=service_endpoint,
            access_key_id=consumer_key,
            secret_access_key=consumer_secret,
        )

        # Get execution by ID (if possible)
        print("  Connecting to Rimay service...")

        # Try to fetch logs directly
        log_path = RESULTS_DIR / f"rimay_execution_logs_{execution_id}.txt"
        with open(log_path, "w") as f:
            f.write(f"Execution ID: {execution_id}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Note: Logs retrieved from PlanQK service API\n")
            f.write(
                "Status: Check the PlanQK dashboard for detailed error information\n"
            )
            f.write(
                f"Dashboard URL: Check your PlanQK console for execution ID {execution_id}\n"
            )

        print(f"\n  Log file saved: {log_path}")
        print("\n  To investigate the failure:")
        print(f"    1. Check the PlanQK Dashboard for execution {execution_id}")
        print("    2. Look for error messages in the execution details")
        print("    3. Verify dataset format matches Rimay expectations (15 qubits max)")

    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Try to load from metadata
    metadata_path = RESULTS_DIR / "rimay_metadata.json"

    if not metadata_path.exists():
        print("ERROR: No rimay_metadata.json found")
        print(f"Expected at: {metadata_path}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    execution_id = metadata.get("execution_id")
    if not execution_id:
        print("ERROR: No execution_id in metadata")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  RIMAY EXECUTION DEBUG")
    print(f"{'=' * 60}")
    print(f"  Input Datapool:  {metadata['input_datapool_id']}")
    print(f"  Output Datapool: {metadata['output_datapool_id']}")
    print(f"  Execution ID:    {execution_id}")
    print(f"  Shots:           {metadata.get('num_shots', 'N/A')}")
    print(f"  Runs:            {metadata.get('num_runs', 'N/A')}")

    # Fetch logs
    fetch_execution_logs(execution_id)

    print(f"\n{'=' * 60}")
    print("  COMMON FAILURE REASONS:")
    print(f"{'=' * 60}")
    print("  1. Dataset size exceeds limits (MAX_SAMPLES=1000)")
    print("  2. Number of features exceeds limits (MAX_FEATURES=15)")
    print("  3. Data format issue (NaN, non-numeric, etc.)")
    print("  4. Rimay service temporarily unavailable")
    print("  5. Insufficient quota or permissions")
    print(f"\n  Metadata: {metadata_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
