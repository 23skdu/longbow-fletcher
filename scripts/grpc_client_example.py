#!/usr/bin/env python3
"""
Example Python client for longbow-fletcher embedding server.

Usage:
    # Single text via command line
    python grpc_client_example.py --server http://localhost:8080 "Hello, world!"

    # Multiple texts
    python grpc_client_example.py --server http://localhost:8080 "Text one" "Text two"

    # Via stdin pipe
    echo "Hello from pipe" | python grpc_client_example.py --server http://localhost:8080

    # File input via pipe
    cat texts.txt | python grpc_client_example.py --server http://localhost:8080

Requirements:
    pip install cbor2 httpx

Note: Fletcher uses HTTP/2 + CBOR, not actual gRPC. This client implements that protocol.
"""

import argparse
import sys
import json

try:
    import cbor2
except ImportError:
    print("Error: cbor2 not installed. Run: pip install cbor2", file=sys.stderr)
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("Error: httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)


def encode_texts(server_url: str, texts: list[str], timeout: float = 30.0) -> list:
    """
    Send texts to Fletcher server for embedding.
    
    Args:
        server_url: Base URL of Fletcher server (e.g., http://localhost:8080)
        texts: List of text strings to embed
        timeout: Request timeout in seconds
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    url = f"{server_url.rstrip('/')}/encode"
    
    # CBOR encode the request
    request_body = cbor2.dumps({"texts": texts})
    
    headers = {
        "Content-Type": "application/cbor",
        "Accept": "application/cbor",
    }
    
    # Use HTTP/2 for better performance
    with httpx.Client(http2=True, timeout=timeout) as client:
        response = client.post(url, content=request_body, headers=headers)
        response.raise_for_status()
        
        # CBOR decode the response
        result = cbor2.loads(response.content)
        return result.get("embeddings", [])


def main():
    parser = argparse.ArgumentParser(
        description="Send text to Fletcher embedding server (HTTP/2 + CBOR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--server", "-s",
        default="http://localhost:8080",
        help="Fletcher server URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output embeddings as JSON (default: summary only)"
    )
    parser.add_argument(
        "--dims", "-d",
        action="store_true",
        help="Print embedding dimensions only"
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="Text(s) to embed (reads from stdin if not provided)"
    )
    
    args = parser.parse_args()
    
    # Collect texts from args or stdin
    texts = args.texts
    if not texts:
        # Read from stdin (supports piping)
        if sys.stdin.isatty():
            print("No texts provided. Use --help for usage.", file=sys.stderr)
            sys.exit(1)
        stdin_content = sys.stdin.read().strip()
        if stdin_content:
            # Split by newlines for multiple texts
            texts = [line.strip() for line in stdin_content.split('\n') if line.strip()]
    
    if not texts:
        print("No texts to process.", file=sys.stderr)
        sys.exit(1)
    
    try:
        embeddings = encode_texts(args.server, texts, args.timeout)
        
        if args.dims:
            # Print dimensions only
            for i, emb in enumerate(embeddings):
                print(f"Text {i+1}: {len(emb)} dimensions")
        elif args.json:
            # Output full JSON
            output = {
                "texts": texts,
                "embeddings": embeddings,
                "dimensions": len(embeddings[0]) if embeddings else 0
            }
            print(json.dumps(output, indent=2))
        else:
            # Summary output
            print(f"Processed {len(texts)} text(s)")
            print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                preview = text[:50] + "..." if len(text) > 50 else text
                print(f"  [{i+1}] \"{preview}\" -> [{emb[0]:.4f}, {emb[1]:.4f}, ...] ({len(emb)}d)")
                
    except httpx.ConnectError:
        print(f"Error: Could not connect to {args.server}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"Error: HTTP {e.response.status_code} - {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
