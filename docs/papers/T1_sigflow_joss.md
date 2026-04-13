---
title: 'sigflow: A Typed Declarative DAG Runtime for Real-Time Clinical Workstations'
tags:
  - Python
  - PySide6
  - Qt
  - clinical informatics
  - signal processing
  - multimodal pipeline
authors:
  - name: Varun Shijo
    orcid: 0000-0001-8266-9047
    corresponding: true
    affiliation: 1
  - name: Wenyao Xu
    affiliation: 1
  - name: Jun Xia
    affiliation: 2
  - name: Ling-Yu Guo
    affiliation: 3
affiliations:
  - name: Department of Computer Science and Engineering, University at Buffalo
    index: 1
  - name: Department of Biomedical Engineering, University at Buffalo
    index: 2
  - name: Department of Communicative Disorders and Sciences, University at Buffalo
    index: 3
date: 2026-04-11
bibliography: paper.bib
---

# Summary

`sigflow` is a Python library for building and executing typed directed-acyclic-graph (DAG)
pipelines over real-time, heterogeneous signal streams. It targets clinical workstations that
simultaneously ingest ultrasound frames, camera video, microphone audio, joint keypoints, and
discrete phoneme events. Each processing node is a plain Python function decorated with
`@source_node`, `@process_node`, or `@sink_node`; ports carry typed annotations drawn from a class
hierarchy rooted at `PortType`, so incompatible connections fail at graph construction rather than
at runtime. The execution engine stamps every sample with a Lab Streaming Layer (LSL) timestamp at
the source thread and correlates multi-input nodes by source ancestry rather than wall time.
Pipelines serialize to YAML for version-controlled deployment. `sigflow` is the graph runtime
underlying UltraSpeech [@varunshijo2026ultraspeech], a clinical speech analysis workstation that
runs 31 nodes across 5 simultaneous modalities, with sub-millisecond LSL timestamp accuracy
maintained end-to-end across each session.

# Statement of need

Real-time multimodal acquisition for clinical research connects hardware streams with incompatible
frame rates, event structures, and threading models. Existing tools each solve part of this
problem. `pylsl` / `liblsl` [@kothe2015] provides high-accuracy LSL streaming but imposes no graph
topology: the application must route, buffer, and synchronize samples manually. Single-domain
libraries (sounddevice for audio, OpenCV for video) are not time-synchronized to each other.
Reactive frameworks like RxPY [@rxpy2016] provide operators over observable sequences but have no
notion of typed ports, node categories, or session recording. Distributed brokers (Apache Kafka
[@kafka2011], ZeroMQ [@hintjens2013]) offer transport at scale but introduce infrastructure
overhead inappropriate for a self-contained clinical workstation.

The core problem has two components. First, *type safety*: a `CameraFrame` and an `UltrasoundFrame`
are both 2D images, but connecting them to the same downstream stage is physically incorrect.
String-tagged systems miss this; Python's class hierarchy, with a single `issubclass` compatibility
check, catches it at pipeline construction. Second, *multi-stream synchronization*: a node that
fuses ultrasound frames with lip keypoints must wait until both arrive from the same acquisition
epoch. Naive timestamp matching accumulates drift across sources; sigflow traces each input port
back to its ancestor source nodes and applies generation-based matching when all ancestors are
shared, falling back to latest-per-port when they diverge independently.

Before sigflow, the UltraSpeech pipeline was threaded ad-hoc: each device had a private callback
thread, synchronization was enforced by explicit locks on per-modality queues, and the topology
was implicit in call order rather than declared. Migrating to sigflow reduced the pipeline layer
by 2.6× in lines of code, removed all explicit lock usage from application code, and made the
pipeline topology inspectable and serializable for the first time. Researchers new to the project
can read the YAML graph definition before touching Python code.

# Design and features

**Typed port hierarchy.** `PortType` is the root class. `TimeSeries` branches into `TimeSeries1D`
(audio, IMU, EEG/EMG) and `TimeSeries2D` (any frame-shaped modality). Concrete leaf types
(`UltrasoundFrame`, `CameraFrame`, `IRFrame`, `TongueKeypoints`, `LipKeypoints`, `FaceLandmarks`,
`PhonemeEvent`, `MarkerEvent`) carry semantic meaning. A node accepting `TimeSeries2D` accepts all
frame types; a node accepting `UltrasoundFrame` rejects a `CameraFrame` at connection time.
Compatibility is a single `issubclass` call on two Python classes.

**Node decorators.** `@source_node`, `@process_node`, and `@sink_node` wrap a plain function into a
`NodeSpec` and auto-register it to a global registry. An optional `@func.init` / `@func.cleanup`
pair manages per-node state across a session. Parameters declared in the spec are type-coerced on
startup and hot-updatable at runtime via `Pipeline.update_node_config`, so a clinician can adjust a
filter threshold mid-session without restarting the pipeline.

**Thread model and Qt integration.** Source nodes run on dedicated daemon threads with priority
elevated via `os.nice(-5)` and push samples onto a bounded output queue drained by a dispatch
thread. Process and sink nodes execute in a shared `ThreadPoolExecutor`. The pipeline never calls Qt
APIs directly; callers connect `Pipeline.on_sample` to a Qt signal for UI updates. This separation
allows sigflow to run inside a PySide6 application without blocking the main event loop.

**YAML serialization and plugin scanning.** A `Graph` is a flat list of `NodeDef` instances and
`Connection` tuples, serializable to YAML or JSON in a single call. `Pipeline.from_graph`
instantiates the full pipeline from a saved graph. `scan_plugins` imports all `.py` files from a
directory and auto-registers any decorated nodes, so application-specific node libraries extend
sigflow without modifying the library itself.

**Session recording.** `SessionRecorder` writes all dispatched samples to XDF [@kothe2014xdf]
(time-series, keypoints, events) and per-node MP4 (video streams), with actual frame rate inferred
from LSL timestamp deltas rather than assumed from a nominal rate. Sessions replay deterministically
from the XDF and YAML graph.

# Example use

The UltraSpeech clinical workstation [@varunshijo2026ultraspeech] deploys sigflow with 31 nodes
spanning 5 modalities: ultrasound frames via Android scrcpy capture, a USB lip camera, a
microphone, DeepLabCut keypoint streams, and phoneme events from an ONNX acoustic model. The full
pipeline topology serializes to 120 lines of YAML. A representative source node, drawn directly
from the sigflow built-in library, reads:

```python
@source_node(
    name="webcam",
    outputs=[Port("frame", CameraFrame)],
    params=[Param("device", "int", 0)],
)
def webcam(*, state, config, clock):
    if "cap" not in state:
        state["cap"] = cv2.VideoCapture(config["device"], cv2.CAP_V4L2)
    ret, frame = state["cap"].read()
    if ret:
        return {"frame": Sample(
            source_id=config["source_id"],
            lsl_timestamp=clock.lsl_now(),
            session_time_ms=clock.session_time_ms(),
            data=frame,
            port_type=CameraFrame,
        )}
```

Every sample carries an LSL timestamp acquired at the source thread, preserving sub-millisecond
accuracy throughout the pipeline regardless of downstream processing latency. The 234 tests
covering the UltraSpeech pipeline layer exercise sigflow's public API directly, with no mocking of
the graph or scheduler. `MasterClock` accepts an injectable time function so the test suite runs
deterministically without a live LSL installation.

# Acknowledgements

The UltraSpeech clinical workstation was developed with support from Auspex Medix.

# References

<!-- paper.bib entries to be finalized before submission. Placeholder keys used above:
  @varunshijo2026ultraspeech — V. Shijo et al., "UltraSpeech: Real-Time Multimodal Speech
      Analysis for Clinical SLP Workstations," in review, 2026.
  @kothe2015 — C. Kothe and T. Mullen, "Lab Streaming Layer (LSL)," 2015.
  @kothe2014xdf — C. Kothe, "Extensible Data Format (XDF)," 2014.
  @kafka2011 — A. Kreps et al., "Kafka: A Distributed Messaging System for Log Processing,"
      NetDB Workshop, 2011.
  @hintjens2013 — P. Hintjens, "ZeroMQ: Messaging for Many Applications," O'Reilly, 2013.
  @rxpy2016 — ReactiveX Contributors, "RxPY: ReactiveX for Python," 2016.
-->
