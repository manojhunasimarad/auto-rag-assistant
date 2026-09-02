# Device Bring-Up on a Multi-OS Cockpit SoC (Qualcomm SA8295-class Platform)

## Platform shape

Gen-4 cockpit/ADAS SoCs in the Qualcomm SA8295 class are built to host
several operating environments simultaneously on one chip — commonly
Linux, QNX, and Android — split across domains by a hypervisor (see
`xen_hypervisor_bringup.md` for the general pattern; on this class of
silicon it's typically QNX Hypervisor). A customer board built around this
SoC needs full device-driver bring-up across every domain that will use a
given piece of hardware, which is a materially bigger job than bringing up
drivers for a single-OS board, because every device has to be re-evaluated
for "does this domain own it exclusively, share it, or not touch it at
all."

## Domains and what each typically owns

- **Linux domain**: general application processing — often owns camera/ISP
  pipelines, some ADAS compute, and general-purpose I/O.
- **QNX domain**: safety- and real-time-relevant workloads — often owns the
  digital instrument cluster display and time-critical sensor paths.
- **Android domain**: infotainment/HMI-facing workloads — owns the
  infotainment display surface, and needs its own driver stack for
  anything the user directly interacts with (touch, media, and features
  like a voice assistant integration).

## Device bring-up, concretely

For each device class — display, microphone/audio capture, speaker/audio
playback, camera/video — bring-up means: getting the kernel driver
(or QNX resource manager, or Android HAL, depending on domain) talking to
the actual silicon block, validating it in isolation on that one domain,
and then validating it again once the device is shared or split across
domains (e.g., one physical display controller driving a cluster surface
on QNX and an infotainment surface on Android; or one microphone whose
capture path needs to reach both a QNX-side wake-word detector and an
Android-side voice assistant).

## Why "same board, three OSes" is harder than it sounds

The same physical peripheral often needs three different driver stacks
(kernel driver on Linux, resource manager on QNX, HAL on Android) that all
have to agree on how the hardware is partitioned by the hypervisor
underneath them. A change on one domain's driver (e.g. a display timing
parameter) can silently affect what another domain sees on a shared
display controller, so bring-up work here is as much about carefully
sequencing and isolating changes across domains as it is about writing any
single driver.

## ADAS and voice assistant integration

On top of core device bring-up, higher-level application integration work
sits on these domains too — for example, standing up an ADAS application
that consumes the camera/sensor pipeline on one domain, and bringing up a
voice-assistant integration (e.g. Alexa Auto SDK-style integration) that
needs the microphone capture path, wake-word detection, and audio playback
all working correctly together, often spanning the QNX and Android splits
described above.

## Face detection as a bring-up-adjacent example

A face-detection application is a useful bring-up sanity check because it
exercises the full camera pipeline (sensor -> ISP -> buffer delivery to a
user-space application) and needs the display/rendering path to show a
result. Getting an application like this working on Linux but *not* on QNX
on the same board is a normal and diagnostic outcome — it usually points
to a difference in how the camera/ISP driver stack is exposed on QNX
(resource manager APIs, buffer/format negotiation) rather than an
application-logic bug, since the same application logic already proved
itself on Linux.

## Requirements gathering as part of bring-up

On a customer engagement, "bring-up" isn't purely a technical task —
understanding what the customer's target vehicle program actually needs
from the board (which domains need which devices, in what priority order)
directly shapes the bring-up plan and sequencing, which is why
requirement-gathering and bring-up planning tend to be done by the same
engineer rather than handed off separately.
