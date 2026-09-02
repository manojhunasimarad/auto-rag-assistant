# AUTOSAR Adaptive Platform — Architecture Overview

## What problem it solves

Classic AUTOSAR (the "Classic Platform") targets statically-configured ECUs
running fixed-function software: everything is wired at compile time, there
is no dynamic memory allocation, and the OS is a small real-time kernel
(OSEK/VDX-style). This works well for a body control module or an engine
ECU, but it cannot host workloads like sensor fusion, path planning, or an
in-vehicle infotainment stack that need POSIX, dynamic process management,
and service-oriented communication. AUTOSAR Adaptive Platform (AP) was
defined to cover exactly this class of workload: high-performance compute
ECUs (HPCs) running Linux (or a POSIX-compliant PSE51/PSE52 OS) that need
safety and a standardized software architecture, but also need the
flexibility of a general-purpose OS.

## Core building blocks

- **Adaptive Applications (AA)** — user-space processes that implement a
  function such as sensor fusion, localization, or a diagnostics service.
  Each AA links against the AUTOSAR Runtime for Adaptive Applications
  (ARA), which exposes the platform's services through C++ APIs.
- **Functional Clusters** — standardized platform services: Execution
  Management (starts/stops/monitors adaptive applications and manages
  functional groups and states), Communication Management (service-oriented
  communication, typically over SOME/IP, but pluggable — see below),
  Diagnostic Log and Trace, Update and Configuration Management,
  Persistency, and State Management.
- **ara::com** — the service-oriented middleware binding. An adaptive
  application declares "provided" and "required" service interfaces
  (methods, events, fields) in an ARXML service description; ara::com
  generates proxy/skeleton code from that description so the application
  code never talks to the wire protocol directly.
- **Execution Manager** — reads the machine manifest and the process
  manifests, decides startup order based on functional groups, and
  supervises process health (restart policies, reporting to the Watchdog).

## Why this matters for a fusion application

A LiDAR/radar/camera fusion adaptive application is a good example of why
the platform shape matters: it needs to (1) subscribe to multiple sensor
event streams with different rates without blocking on any single one, (2)
publish a fused object list as a service that other adaptive applications
(planning, HMI) can consume without knowing where the fusion process runs,
and (3) be restartable independently of the rest of the system if it
crashes. Building this directly on raw POSIX IPC would mean re-inventing
service discovery, versioning, and lifecycle management for every
application — ara::com and Execution Management give you that for free, in
a way that's portable across vendors' Adaptive Platform implementations
(assuming they're standard-conformant).

## Bring-up in practice

Standing up an Adaptive Platform stack on a new board is mostly integration
work, not application work: build a Yocto (or similar) image that includes
the AP binaries and the target OS, get the machine manifest right for the
board's hardware, and get the first adaptive application ("hello world"
service + client) exchanging events end-to-end before adding real
functionality. The failure modes at this stage are almost always manifest
mismatches (service ID / instance ID / event ID not agreeing between
provider and consumer ARXML) or an Execution Manager startup ordering issue
where a required service isn't up yet when a dependent application starts.

## Relationship to virtualization

On an HPC that also has to run non-AUTOSAR workloads (a Linux-based
infotainment stack, a ROS2-based perception stack, or an Android instance
for CarPlay/Android Auto-style features), the Adaptive Platform instance
typically runs in one guest under a Type-1 hypervisor (Xen or QNX
Hypervisor) alongside those other guests, with inter-domain communication
handled either by a virtual network device or a dedicated shared-memory
channel exposed by the hypervisor. See `xen_hypervisor_bringup.md` and
`qualcomm_sa8295_bringup.md` for how that multi-OS split is actually
brought up on real silicon.
