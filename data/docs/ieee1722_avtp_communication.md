# IEEE 1722 / AVTP for In-Vehicle Signal Distribution

## Background

IEEE 1722 defines the Audio Video Transport Protocol (AVTP), originally
built for professional audio/video networks under the IEEE 802.1 Audio
Video Bridging (AVB) / Time-Sensitive Networking (TSN) umbrella. Automotive
E/E architectures adopted it because it gives a standardized, low-latency,
Layer-2 (Ethernet, not IP) way to move time-sensitive signals — audio,
video, and increasingly generic sensor/control data — between ECUs on a
switched Ethernet backbone, with the timing guarantees TSN provides
(gPTP-synchronized clocks, credit-based shaping, scheduled traffic).

Using Layer 2 instead of a full IP stack matters for two reasons: it
avoids IP-stack overhead and non-determinism on links that need
microsecond-level jitter bounds, and it lets AVTP Talkers and Listeners
discover and stream to each other without needing a routed network.

## Applying AVTP to non-AV signals

The interesting part of this kind of project is that AVTP's payload
formats aren't limited to audio/video streams — the protocol defines
generic "AVTP Control Format" (ACF) subtypes that can carry structured
control/sensor data such as CAN frames. That makes it possible to take a
signal that originates on a vehicle CAN bus — for example accelerator
pedal angle from the powertrain domain — and tunnel it across the Ethernet
backbone to another domain (say, a Linux- or ROS2-hosted compute node)
without inventing a bespoke framing format. The `open1722` open-source
library implements AVTP/ACF packetization and is a practical starting
point instead of hand-rolling packet construction from the spec.

## End-to-end signal path (example)

1. **Source**: accelerator pedal angle (and similar signals) is read off
   the vehicle CAN bus via a CAN-to-Ethernet gateway/source ECU.
2. **Encode**: the raw CAN frame is mapped into an AVTP/ACF CAN payload —
   this means defining a stream ID, sequence numbering, and the ACF
   message type/length fields, then wrapping the original CAN ID + data
   bytes inside it.
3. **Transport**: the AVTP packet goes out over a standard Ethernet frame
   (EtherType 0x22F0) on the TSN-capable switched network, Layer 2, no
   IP addressing needed for the stream itself.
4. **Listener**: a Listener endpoint (running on whichever domain compute
   node — Linux, QNX, or a guest under a hypervisor) subscribes to the
   stream ID, decodes the ACF/CAN payload back into the original signal,
   and hands it off to whatever's consuming it locally (a ROS2 node, an
   AUTOSAR Adaptive application, etc.).
5. **Round trip**: nothing stops a Listener from also acting as a Talker —
   a domain that processes the signal (e.g. a ROS2 node that recomputes a
   derived value) can re-encode and publish a *new* AVTP stream back onto
   the network, which is what makes two-way, cross-domain signal flow
   possible instead of one-way telemetry.

## Why not just use SOME/IP or plain UDP/IP here

SOME/IP (and IP-based transports generally) are a natural fit for
service-oriented, request/response, or best-effort traffic between
AUTOSAR Adaptive applications. AVTP is chosen instead when the requirement
is deterministic, low-latency delivery of a continuous signal stream
across domain boundaries on a TSN network, and when the source signal
is naturally frame-based (CAN) rather than service-based. In a real
architecture the two coexist: AVTP for the deterministic signal streams
between domains, SOME/IP or REST/gRPC for service calls and cloud-facing
APIs (see `ros2_cloud_bridge.md`).

## Practical bring-up gotchas

- **Stream reservation / gPTP sync**: if the network isn't gPTP-locked, or
  stream reservation (MSRP/SRP) isn't configured correctly, AVTP traffic
  will still "work" on a quiet network but will show jitter or drops under
  load — a common false-negative during early bring-up.
- **Endianness and field packing**: ACF header fields are bit-packed
  per the spec; a mismatch between Talker and Listener implementations
  (or between open1722 versions) here is the single most common source of
  "packet arrives but decodes to garbage" bugs.
- **Domain boundary semantics**: deciding what a "domain" boundary means
  in your architecture (a Linux guest vs a QNX guest vs a bare ECU) affects
  what has to be an AVTP Talker/Listener versus what can stay as plain
  intra-domain IPC.
