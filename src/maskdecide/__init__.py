# SPDX-License-Identifier: Apache-2.0
"""MaskDecide: local typed decisions powered by Fast-dLLM."""


def main() -> None:
    """Start a single-worker API server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()
    uvicorn.run("maskdecide.api:app", host=args.host, port=args.port, workers=1)
