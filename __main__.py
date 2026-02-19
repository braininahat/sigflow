"""CLI entry point: uv run python -m sigflow <command>"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def _import_builtin_nodes():
    """Import all built-in nodes so their decorators register them."""
    import sigflow.nodes.webcam_source  # noqa: F401
    import sigflow.nodes.cv2_display  # noqa: F401
    import sigflow.nodes.crop  # noqa: F401
    import sigflow.nodes.audio_source  # noqa: F401
    import sigflow.nodes.spectrogram  # noqa: F401
    import sigflow.nodes.canvas_display  # noqa: F401
    import sigflow.nodes.dlc_inference  # noqa: F401
    import sigflow.nodes.keypoints_overlay  # noqa: F401


def cmd_run(args):
    """Run a pipeline from a YAML/JSON file."""
    from sigflow.graph import Graph
    from sigflow.runtime import Pipeline

    _import_builtin_nodes()

    path = Path(args.graph)
    log.info("loading graph from %s", path)
    if path.suffix in (".yaml", ".yml"):
        graph = Graph.load_yaml(path)
    else:
        graph = Graph.load_json(path)

    log.info("graph: %d nodes, %d connections", len(graph.nodes), len(graph.connections))
    pipeline = Pipeline.from_graph(graph, max_workers=args.workers)

    # Handle Ctrl+C gracefully
    running = True
    def on_sigint(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, on_sigint)

    pipeline.start()

    try:
        import time
        from sigflow.nodes.cv2_display import drain_display_queue
        while running:
            if not drain_display_queue():
                time.sleep(0.01)
    finally:
        pipeline.stop()
        import cv2
        cv2.destroyAllWindows()


def cmd_list_nodes(args):
    """List all registered node types."""
    from sigflow.registry import all_nodes

    _import_builtin_nodes()

    nodes = all_nodes()
    if not nodes:
        print("No nodes registered.")
        return

    by_category: dict[str, list] = {}
    for spec in nodes.values():
        by_category.setdefault(spec.category, []).append(spec)

    for category, specs in sorted(by_category.items()):
        print(f"\n[{category}]")
        for spec in sorted(specs, key=lambda s: s.name):
            inputs = ", ".join(f"{p.name}:{p.type.__name__}" for p in spec.inputs)
            outputs = ", ".join(f"{p.name}:{p.type.__name__}" for p in spec.outputs)
            kind_tag = spec.kind.upper()
            print(f"  {spec.name} ({kind_tag})")
            if inputs:
                print(f"    inputs:  {inputs}")
            if outputs:
                print(f"    outputs: {outputs}")


def main():
    parser = argparse.ArgumentParser(prog="sigflow", description="sigflow DAG pipeline framework")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="increase verbosity (-v info, -vv debug)")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a pipeline from a YAML/JSON graph file")
    run_parser.add_argument("graph", help="Path to pipeline graph file (.yaml or .json)")
    run_parser.add_argument("--workers", type=int, default=4, help="Max worker threads for process nodes")

    sub.add_parser("list-nodes", help="List all registered node types")

    args = parser.parse_args()

    level = logging.WARNING
    if args.verbose >= 2:
        level = logging.DEBUG
    elif args.verbose >= 1:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list-nodes":
        cmd_list_nodes(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
