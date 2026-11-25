# Multi-OS Bring-Up with a Type-1 Hypervisor (Xen / QNX Hypervisor)

## Why virtualize a single automotive compute board at all

A modern automotive high-performance compute (HPC) board — whether it's a
Raspberry Pi-class prototyping board or production silicon like Qualcomm's
SA8295 — is expensive real estate: OEMs would rather host several
functionally-separate operating environments (say, a safety-relevant
AUTOSAR Adaptive Linux guest, a ROS2-based perception/ADAS guest, and an
Android guest for infotainment) on one piece of hardware than ship three
boards. A Type-1 (bare-metal) hypervisor — Xen in the open-source/prototype
case, QNX Hypervisor in several production automotive SoC platforms — makes
that possible by running directly on the hardware and hosting each OS as an
isolated guest ("domain" in Xen terms, "virtual machine" in QNX terms),
each with its own scheduled CPU time, memory partition, and device access.

## What actually has to be brought up

1. **Hypervisor + Dom0/host boot**: the hypervisor itself boots first (via
   the board's bootloader), then brings up the privileged control domain
   (Dom0 in Xen) which owns most physical devices and is responsible for
   creating and managing the other guest domains.
2. **Guest domain definitions**: each guest (Linux, QNX, ROS2-hosting
   Linux, Android) needs a domain configuration describing its allocated
   vCPUs, memory, and which devices are passed through directly versus
   virtualized/shared.
3. **Device partitioning**: this is the hard part in practice. Devices like
   display, audio codec, camera/MIPI-CSI, and shared storage often need to
   be split across guests — e.g., one physical display controller feeding
   a "cluster" surface owned by one guest and an "infotainment" surface
   owned by another, or one audio codec time-shared/mixed across guests.
   This typically means either a device is exclusively passed through to
   one guest (simple, but that guest fully owns it) or a paravirtualized
   front-end/back-end driver split is used, where a back-end driver in a
   privileged domain multiplexes physical access and front-end drivers in
   each guest talk to it over a hypervisor-provided channel.
4. **Inter-domain communication**: guests still need to talk to each other
   (e.g. a signal arriving over IEEE1722/AVTP on the Linux domain's NIC
   needs to reach a ROS2 node in another domain). This is done either via a
   virtual network interface bridged between domains, or a dedicated
   shared-memory/event-channel mechanism the hypervisor exposes.
5. **Per-guest OS boot and drivers**: once the domain exists, it boots its
   own kernel/OS image (Linux, QNX, or Android's kernel) and needs the
   right paravirtualized or passthrough drivers for whatever devices that
   guest depends on — display, audio, camera, storage.

## Bring-up order that avoids the most debugging pain

Bringing up "everything at once" tends to produce failures that are hard to
localize (is it the hypervisor config, the domain config, or the guest's
own driver?). A more debuggable order: (1) get Dom0/host boot solid and
confirm device ownership there, (2) bring up one guest with no shared
devices at all — CPU/memory isolation only — and confirm it boots and is
schedulable, (3) add one passed-through device to that guest and confirm
it end-to-end before adding the next, (4) only once every guest boots
independently, wire up the shared/paravirtualized devices (display split,
audio mix) and inter-domain communication, since those failure modes are
much easier to reason about when you already trust each guest in
isolation.

## Containers on top of the hypervisor

Once a Linux guest is stable, Docker (or another OCI runtime) can run
inside it exactly as it would on bare metal — the hypervisor guest just
looks like a normal Linux machine to Docker. This is a convenient way to
package and redeploy application-level workloads (a fusion service, a
diagnostics tool, a demo application) inside a guest without touching the
guest's base image every time, which matters a lot during iterative
development and client demos.

## Common failure signatures

- A guest that boots but never gets scheduled time under load usually
  means a vCPU pinning/priority misconfiguration in the domain definition,
  not a guest OS bug.
- Display corruption or tearing on a split display setup is very often a
  synchronization issue in the paravirtualized display back-end, not the
  guest's own display driver.
- "Works on native hardware, breaks under the hypervisor" for a driver
  almost always means the driver assumes exclusive/direct hardware access
  that the passthrough or paravirtualization layer doesn't actually give
  it.
