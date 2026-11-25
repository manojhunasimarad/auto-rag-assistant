# Bridging ROS2 Vehicle Signals to the Cloud with gRPC

## The two-way problem

A common architecture need: signals arriving in a ROS2 domain (for example,
accelerator pedal angle decoded from an incoming IEEE1722/AVTP stream —
see `ieee1722_avtp_communication.md`) need to (1) be processed/modified by
ROS2 nodes and (2) be sent to a cloud backend for logging, analytics, or a
remote dashboard — and then, in the other direction, the cloud or another
domain may need to push a modified or derived value *back* into the
vehicle's ROS2 graph. That's a two-way pipeline, not a one-way telemetry
upload, and the two directions have different failure modes.

## Why gRPC for the cloud leg

Between the Linux domain and the cloud, gRPC is a reasonable choice over
raw REST/JSON for a few concrete reasons relevant to this kind of link:
Protocol Buffers give a compact binary wire format (useful when the uplink
is metered/bandwidth constrained), a strongly-typed schema (the `.proto`
contract) that both client and server code-generate from, and native
support for streaming RPCs — which matters because pedal-angle-class
signals are naturally a stream, not a single request/response.

A minimal architecture:

- **ROS2 side**: a bridge node subscribes to the relevant ROS2 topics,
  converts each message to the corresponding protobuf message type, and
  is a gRPC *client* streaming those messages up to the cloud service
  using a client-streaming or bidirectional-streaming RPC.
- **Cloud side**: a gRPC server (deployed on cloud infrastructure, fronted
  by a load balancer/ingress) receives the stream, does whatever
  processing/storage is needed, and — for the two-way case — can push
  messages back down the same bidirectional stream.
- **Back into ROS2**: the bridge node receives the pushed-back message on
  the open stream and republishes it onto a ROS2 topic, so downstream
  ROS2 nodes see it exactly like any other topic message — they don't need
  to know it originated from the cloud.

## Why this needs to be a *bridge* node, not baked into every consumer

Keeping gRPC entirely inside one bridge node (rather than having every
ROS2 node that cares about cloud data speak gRPC directly) keeps the
transport concern separate from application logic: ROS2 nodes only ever
deal in ROS2 messages/topics, which makes them independently testable
with recorded rosbag data, and it means the cloud transport can be
swapped (e.g. gRPC to MQTT) without touching application nodes.

## Reliability considerations

- **Reconnection**: a streaming RPC over an intermittent vehicle uplink
  will drop; the bridge node needs its own reconnect-with-backoff logic
  and, if it can't buffer indefinitely, a clear policy for what to do with
  data produced while disconnected (drop oldest, downsample, or persist
  to local disk and replay).
- **Backpressure**: if the cloud side falls behind, an unbounded queue on
  the bridge node will eventually exhaust memory; the bridge needs a
  bounded queue with an explicit drop/downsample policy rather than
  discovering this under load.
- **Message ordering vs. freshness**: for a low-latency control-relevant
  signal, strict in-order delivery is sometimes less important than
  freshness — an old buffered value arriving late can be worse than
  dropping it, which is a product decision, not just an engineering one.

## Where this fits in a POC vs. a production system

For a proof-of-concept, this can run as a single ROS2 bridge node talking
to a single cloud VM or managed service with CI/CD-deployed containers.
Production hardening on top of that POC would add mutual TLS between
vehicle and cloud, authenticated/authorized streams per vehicle, and a
proper backpressure/QoS policy — none of which change the core
architecture, but all of which matter before this crosses from demo to
fleet-scale deployment.
