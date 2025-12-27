#!/usr/bin/env python3
"""
Arrow Flight client for longbow vector search operations.

This client communicates with a longbow server (NOT fletcher) to perform:
- Vector similarity search (VectorSearch DoAction)
- Dataset operations (list, info, upload, download)
- Distributed search with x-longbow-key routing

Usage:
    # Vector search with random query vector
    python arrow_flight_client.py search --dataset mydata --dim 128 --k 10

    # Search with explicit vector (JSON array)
    python arrow_flight_client.py search --dataset mydata --vector "[0.1, 0.2, ...]" --k 5

    # Distributed search with routing key
    python arrow_flight_client.py search --dataset mydata --dim 128 --k 10 --routing-key "shard-1"

    # Global distributed search across all nodes
    python arrow_flight_client.py search --dataset mydata --dim 128 --k 10 --global

    # Hybrid search (vector + text)
    python arrow_flight_client.py search --dataset mydata --dim 128 --k 10 --text "query text"

    # List all datasets
    python arrow_flight_client.py list

    # Get dataset info
    python arrow_flight_client.py info --dataset mydata

Requirements:
    pip install pyarrow numpy
"""

import argparse
import json
import sys
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.flight as flight
except ImportError:
    print("Error: pyarrow not installed. Run: pip install pyarrow", file=sys.stderr)
    sys.exit(1)


def get_options(args) -> flight.FlightCallOptions:
    """Generate FlightCallOptions with routing metadata."""
    headers = []
    
    # x-longbow-key for explicit routing
    if hasattr(args, 'routing_key') and args.routing_key:
        headers.append((b"x-longbow-key", args.routing_key.encode("utf-8")))
    
    # x-longbow-global for distributed search
    if hasattr(args, 'global_search') and args.global_search:
        headers.append((b"x-longbow-global", b"true"))
    
    if headers:
        return flight.FlightCallOptions(headers=headers)
    return flight.FlightCallOptions()


def command_search(args, client: flight.FlightClient):
    """Perform vector similarity search using DoAction(VectorSearch)."""
    # Generate or parse query vector
    if args.vector:
        query_vector = json.loads(args.vector)
    else:
        query_vector = np.random.rand(args.dim).astype(np.float32).tolist()
    
    # Read from stdin if piped
    text_query = args.text
    if not text_query and not sys.stdin.isatty():
        stdin_data = sys.stdin.read().strip()
        if stdin_data:
            text_query = stdin_data
    
    request = {
        "dataset": args.dataset,
        "vector": query_vector,
        "k": args.k,
    }
    
    # Hybrid search
    if text_query:
        request["text_query"] = text_query
        request["alpha"] = args.alpha
        print(f"Hybrid search: text='{text_query[:50]}...' alpha={args.alpha}", file=sys.stderr)
    
    payload = json.dumps(request).encode("utf-8")
    options = get_options(args)
    
    search_type = "GLOBAL" if args.global_search else "LOCAL"
    if args.routing_key:
        search_type += f" (routed to {args.routing_key})"
    
    print(f"[{search_type}] Searching '{args.dataset}' (k={args.k}, dim={len(query_vector)})...", file=sys.stderr)
    
    try:
        action = flight.Action("VectorSearch", payload)
        results = client.do_action(action, options=options)
        
        for res in results:
            body = json.loads(res.body.to_pybytes())
            
            if args.json:
                print(json.dumps(body, indent=2))
            else:
                ids = body.get("ids", [])
                scores = body.get("scores", [])
                print(f"Found {len(ids)} results:")
                for i, (vid, score) in enumerate(zip(ids, scores)):
                    print(f"  [{i+1}] ID={vid} Score={score:.6f}")
                    
    except flight.FlightError as e:
        print(f"Search failed: {e}", file=sys.stderr)
        sys.exit(1)


def command_list(args, client: flight.FlightClient):
    """List available datasets."""
    print("Listing datasets...", file=sys.stderr)
    try:
        flights = client.list_flights()
        datasets = []
        for info in flights:
            path = info.descriptor.path[0].decode('utf-8') if info.descriptor.path else "Unknown"
            datasets.append({
                "name": path,
                "records": info.total_records,
                "bytes": info.total_bytes
            })
            print(f"  - {path} ({info.total_records:,} records, {info.total_bytes:,} bytes)")
        
        if args.json:
            print(json.dumps(datasets, indent=2))
        elif not datasets:
            print("No datasets found.")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def command_info(args, client: flight.FlightClient):
    """Get dataset info."""
    print(f"Getting info for '{args.dataset}'...", file=sys.stderr)
    try:
        descriptor = flight.FlightDescriptor.for_path(args.dataset)
        options = get_options(args)
        info = client.get_flight_info(descriptor, options=options)
        
        result = {
            "dataset": args.dataset,
            "records": info.total_records,
            "bytes": info.total_bytes,
            "schema": str(info.schema)
        }
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Dataset: {args.dataset}")
            print(f"Records: {info.total_records:,}")
            print(f"Bytes: {info.total_bytes:,}")
            print(f"Schema: {info.schema}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def command_status(args, client: flight.FlightClient):
    """Get cluster status."""
    print("Getting cluster status...", file=sys.stderr)
    try:
        action = flight.Action("cluster-status", b"")
        options = get_options(args)
        results = client.do_action(action, options=options)
        
        for res in results:
            status = json.loads(res.body.to_pybytes())
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                count = status.get("count", 0)
                print(f"Cluster: {count} active members")
                for m in status.get("members", []):
                    print(f"  - {m.get('ID')} ({m.get('Addr')}) [{m.get('Status', 'Unknown')}]")
                    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Arrow Flight client for longbow vector search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Global connection options
    parser.add_argument(
        "--data-uri", "-d",
        default="grpc://localhost:3000",
        help="Longbow data server URI (default: grpc://localhost:3000)"
    )
    parser.add_argument(
        "--meta-uri", "-m",
        default="grpc://localhost:3001",
        help="Longbow meta server URI (default: grpc://localhost:3001)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--routing-key", "-r",
        help="Explicit routing key (x-longbow-key header)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # SEARCH command
    search_parser = subparsers.add_parser("search", help="Vector similarity search")
    search_parser.add_argument("--dataset", required=True, help="Dataset name")
    search_parser.add_argument("--dim", type=int, default=128, help="Vector dimension (for random query)")
    search_parser.add_argument("--k", type=int, default=10, help="Top K results")
    search_parser.add_argument("--vector", help="Query vector as JSON array")
    search_parser.add_argument("--text", help="Text query for hybrid search")
    search_parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid alpha (0=sparse, 1=dense)")
    search_parser.add_argument("--global", dest="global_search", action="store_true", 
                               help="Global distributed search (x-longbow-global)")
    
    # LIST command
    subparsers.add_parser("list", help="List all datasets")
    
    # INFO command
    info_parser = subparsers.add_parser("info", help="Get dataset info")
    info_parser.add_argument("--dataset", required=True, help="Dataset name")
    
    # STATUS command
    subparsers.add_parser("status", help="Get cluster status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Use meta client for search/status, data client for list/info
    try:
        if args.command in ("search", "status"):
            client = flight.FlightClient(args.meta_uri)
        else:
            client = flight.FlightClient(args.data_uri)
        
        commands = {
            "search": command_search,
            "list": command_list,
            "info": command_info,
            "status": command_status,
        }
        
        func = commands.get(args.command)
        if func:
            func(args, client)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
