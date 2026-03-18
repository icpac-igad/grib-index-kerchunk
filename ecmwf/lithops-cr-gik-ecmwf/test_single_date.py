#!/usr/bin/env python3
"""
Test ECMWF Worker - Single Date Processing
===========================================

Quick test of the Cloud Run worker with a single date and limited members.

Usage:
    python test_single_date.py
"""

import requests
import time
import json
from datetime import datetime

# Service URL
WORKER_URL = "https://ecmwf-parquet-worker-yiyrp6yumq-uc.a.run.app"

def test_single_date(date_str="20260206", run="00", max_members=3):
    """
    Test processing a single date with limited members.

    Args:
        date_str: Date in YYYYMMDD format
        run: Model run hour (00 or 12)
        max_members: Number of ensemble members to process (for quick test)
    """
    print("="*70)
    print("ECMWF PARQUET WORKER - SINGLE DATE TEST")
    print("="*70)
    print(f"\nWorker URL: {WORKER_URL}")
    print(f"Date: {date_str}")
    print(f"Run: {run}Z")
    print(f"Max Members: {max_members} (quick test)")
    print("\n" + "="*70)

    # Test health first
    print("\n[1/3] Testing /health endpoint...")
    try:
        health_response = requests.get(f"{WORKER_URL}/health", timeout=10)
        health_response.raise_for_status()
        print(f"✓ Health check passed: {health_response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

    # Test root endpoint
    print("\n[2/3] Testing / endpoint...")
    try:
        root_response = requests.get(f"{WORKER_URL}/", timeout=10)
        root_response.raise_for_status()
        config = root_response.json()
        print(f"✓ Service info retrieved:")
        print(f"  Name: {config['name']}")
        print(f"  Version: {config['version']}")
        print(f"  GCS Bucket: {config['configuration']['gcs_bucket']}")
        print(f"  Parallel Workers: {config['configuration']['parallel_workers']}")
    except Exception as e:
        print(f"✗ Root endpoint failed: {e}")
        return False

    # Process date
    print(f"\n[3/3] Processing date {date_str} with {max_members} members...")
    print(f"NOTE: This will take approximately 2-3 minutes for {max_members} members")
    print("      (Full 51 members takes ~10-15 minutes)")

    payload = {
        "date": date_str,
        "run": run,
        "max_members": max_members
    }

    print(f"\nSending POST request to {WORKER_URL}/process")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    start_time = time.time()

    try:
        response = requests.post(
            f"{WORKER_URL}/process",
            json=payload,
            timeout=3600  # 1 hour timeout
        )

        elapsed = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        result = response.json()

        print("\n" + "="*70)
        print("PROCESSING RESULT")
        print("="*70)

        if result.get('success'):
            print(f"\n✓ SUCCESS!")
            print(f"\nDetails:")
            print(f"  Date: {result.get('date')}")
            print(f"  Run: {result.get('run')}Z")
            print(f"  Output Directory: {result.get('output_dir')}")
            print(f"  GCS Path: {result.get('gcs_path')}")
            print(f"  Files Uploaded: {result.get('files_count')}")
            print(f"  Processing Time: {result.get('total_time_seconds')}s ({result.get('total_time_seconds')/60:.1f} min)")
            print(f"  Message: {result.get('message')}")

            # Show GCS verification command
            gcs_path = result.get('gcs_path')
            if gcs_path:
                print(f"\nVerify GCS upload:")
                print(f"  gsutil ls {gcs_path}/")
                print(f"  Expected files: {result.get('files_count')} parquet files")

            return True
        else:
            print(f"\n✗ FAILED!")
            print(f"\nError: {result.get('error')}")

            if result.get('traceback'):
                print(f"\nTraceback:")
                print(result['traceback'])

            return False

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n✗ Request timeout after {elapsed:.1f}s")
        return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    # Use recent date for testing
    today = datetime.now()
    # Go back a few days to ensure data is available
    test_date = (today.replace(day=today.day-3)).strftime("%Y%m%d")

    print(f"\nTest date: {test_date}")
    print("(Using date from 3 days ago to ensure ECMWF data availability)\n")

    success = test_single_date(date_str=test_date, run="00", max_members=3)

    if success:
        print("\n" + "="*70)
        print("✅ TEST PASSED!")
        print("="*70)
        print("\nNext steps:")
        print("1. Check GCS bucket for uploaded parquets")
        print("2. Test with more members: max_members=10 or full 51")
        print("3. Use Lithops for batch processing multiple dates")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print("\nCheck Cloud Run logs:")
        print(f"  gcloud logging read \"resource.type=cloud_run_revision AND resource.labels.service_name=ecmwf-parquet-worker\" --project=e4drr-crafd --limit=50")
        sys.exit(1)
