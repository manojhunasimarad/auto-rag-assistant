# Cutting Embedded Build Times with Distributed Compilation (IncrediBuild on AWS EC2)

## The problem

Large embedded software builds — a full BSP/Yocto-class image, or a large
AUTOSAR/QNX/Android stack build — can take 24+ hours on a single build
machine. That's a direct productivity tax: every developer waiting on a
full rebuild to validate a change loses most of a day, and CI feedback
loops become effectively useless at that timescale.

## Why this is a distributed-compilation problem, not just "a faster
machine"

A single machine, however powerful, is bounded by its own core count and
I/O. The build graph for a large embedded project is usually *wide*
(thousands of independent compilation units that don't depend on each
other) even if the overall dependency chain is deep in places. That shape
is exactly what distributed build acceleration tools are built for:
instead of running compilation jobs serially (or only as parallel as one
machine's cores allow), the build graph's independent jobs are farmed out
across a pool of helper machines and the results are collected back into
the single build output, as if it ran locally.

## Applying it on AWS EC2

The environment here was a build running on an AWS EC2 instance. Bringing
in IncrediBuild meant: standing up a pool of additional EC2 instances as
build helpers, installing the IncrediBuild agent on each, configuring the
initiator (the original EC2 build host) to know about the helper pool, and
making sure the build tooling itself (whatever's driving compilation —
`make`, a Yocto `bitbake` invocation, or similar) is compatible with being
distributed, since not every build step parallelizes safely (some steps
have hidden dependencies on shared state that aren't expressed in the
build graph, and those need to stay local/serialized).

## What actually moved the needle

Three things mattered more than "just add more helper machines":

1. **Build graph correctness first**: distributing a build with hidden
   undeclared dependencies just makes the flaky failures show up faster
   and more often, not more reliable. Cleaning up the dependency
   declarations in the build system was a prerequisite, not an
   afterthought.
2. **Right-sizing the helper pool**: more helpers has diminishing (and
   eventually negative, due to coordination/network overhead) returns
   once the pool size exceeds the *width* of the build graph's most
   parallel stage. Sizing the pool against actual build-graph analysis
   beat just adding machines until the number "felt big."
3. **Network/storage locality on EC2**: helper instances need fast access
   to source and intermediate artifacts; placement (same VPC/availability
   zone, appropriately sized EBS/instance storage) had a measurable effect
   on wall-clock time independent of compute parallelism.

## Result

The build time was brought down from 24+ hours to roughly 3 hours end to
end — an ~8x reduction — which was demonstrated to the client as a
concrete before/after comparison on the same codebase and hardware
baseline, not a synthetic benchmark.

## Why this belongs next to the other docs in this project

This is a deliberately different *kind* of problem from the AVTP/AUTOSAR/
hypervisor documents in this corpus — it's a build-infrastructure and
distributed-systems problem, not a communications-protocol or OS bring-up
problem. It's included here to make the retrieval system's job realistic:
a good retriever has to correctly *not* pull this document in for a
question about AVTP packet framing, and correctly pull it in for a
question about reducing CI build times — which is exactly the kind of
precision/recall behavior worth being able to talk about in an interview.
