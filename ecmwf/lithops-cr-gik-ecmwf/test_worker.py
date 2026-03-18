#!/usr/bin/env python3
"""
Test Cloud Run Worker Endpoints
================================

Test the ECMWF parquet worker service endpoints directly (without Lithops).

Usage:
    # Test health endpoint
    python test_worker.py --health

    # Test processing with a date
    python test_worker.py --date 20260206

    # Custom worker URL
    python test_worker.py --date 20260206 --url https://custom-url.run.app

Author: ICPAC GIK Team
"""

import os
import sys
import argparse
import time
import requests
from datetime import datetime

# Default worker URL
DEFAULT_WORKER_URL = os.environ.get(
    'WORKER_URL',
    'https://ecmwf-parquet-worker-462481537368.us-central1.run.app'
)


def test_health(worker_url: str):
    """Test health endpoint."""
    print("\n" + "=" * 70)
    print("Testing /health endpoint")
    print("=" * 70)

    url = f"{worker_url}/health"
    print(f"URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("\n✓ Health check passed")
            return True
        else:
            print("\n✗ Health check failed")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_root(worker_url: str):
    """Test root endpoint."""
    print("\n" + "=" * 70)
    print("Testing / endpoint")
    print("=" * 70)

    url = f"{worker_url}/"
    print(f"URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"Service: {data.get('name')}")
            print(f"Version: {data.get('version')}")
            print(f"Description: {data.get('description')}")
            print(f"\nConfiguration:")
            for key, value in data.get('configuration', {}).items():
                print(f"  {key}: {value}")
            print(f"\nEndpoints:")
            for path, desc in data.get('endpoints', {}).items():
                print(f"  {path}: {desc}")

            print("\n✓ Root endpoint passed")
            return True
        else:
            print(f"\nResponse: {response.text}")
            print("\n✗ Root endpoint failed")
            return False

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_process(worker_url: str, date_str: str, run: str = "00", max_members: int = None):
    """Test processing endpoint."""
    print("\n" + "=" * 70)
    print("Testing /process endpoint")
    print("=" * 70)

    # Validate date format
    try:
        datetime.strptime(date_str, '%Y%m%d')
    except ValueError:
        print(f"✗ Invalid date format: {date_str}. Expected YYYYMMDD")
        return False

    url = f"{worker_url}/process"
    print(f"URL: {url}")

    payload = {
        "date": date_str,
        "run": run
    }

    if max_members:
        payload["max_members"] = max_members

    print(f"Payload: {payload}")
    print("\nSending request (this may take 10-15 minutes)...")

    start_time = time.time()

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=3600  # 1 hour
        )

        elapsed = time.time() - start_time

        print(f"\nStatus Code: {response.status_code}")
        print(f"Elapsed Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        result = response.json()

        print("\nResponse:")
        print(f"  Success: {result.get('success')}")
        print(f"  Date: {result.get('date')}")
        print(f"  Run: {result.get('run')}")

        if result.get('success'):
            print(f"  Output Dir: {result.get('output_dir')}")
            print(f"  GCS Path: {result.get('gcs_path')}")
            print(f"  Files Count: {result.get('files_count')}")
            print(f"  Processing Time: {result.get('total_time_seconds')}s")
            print(f"  Message: {result.get('message')}")

            print("\n✓ Processing completed successfully")
            return True
        else:
            print(f"  Error: {result.get('error')}")

            if result.get('traceback'):
                print(f"\nTraceback:")
                print(result['traceback'])

            print("\n✗ Processing failed")
            return False

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"\n✗ Timeout after {elapsed:.1f}s")
        return False

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Error after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test ECMWF parquet worker endpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--url', type=str, default=DEFAULT_WORKER_URL,
                        help=f'Worker URL (default: {DEFAULT_WORKER_URL})')
    parser.add_argument('--health', action='store_true',
                        help='Test health endpoint only')
    parser.add_argument('--root', action='store_true',
                        help='Test root endpoint only')
    parser.add_argument('--date', type=str,
                        help='Test processing endpoint with date (YYYYMMDD)')
    parser.add_argument('--run', type=str, default='00', choices=['00', '12'],
                        help='Model run hour (default: 00)')
    parser.add_argument('--max-members', type=int,
                        help='Limit ensemble members for testing')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')

    args = parser.parse_args()

    worker_url = args.url.rstrip('/')

    print("=" * 70)
    print("ECMWF PARQUET WORKER TEST SUITE")
    print("=" * 70)
    print(f"Worker URL: {worker_url}")
    print("=" * 70)

    results = []

    # Run tests
    if args.health or args.all:
        results.append(("Health", test_health(worker_url)))

    if args.root or args.all:
        results.append(("Root", test_root(worker_url)))

    if args.date:
        results.append(("Process", test_process(
            worker_url, args.date, args.run, args.max_members
        )))
    elif args.all:
        print("\nSkipping /process test (use --date to test processing)")

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        passed = sum(1 for _, result in results if result)
        total = len(results)

        for name, result in results:
            status = "✓" if result else "✗"
            print(f"{status} {name}")

        print(f"\nTotal: {passed}/{total} passed")

        if passed == total:
            print("\n✅ All tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    else:
        print("\nNo tests specified. Use --health, --root, --date, or --all")
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
