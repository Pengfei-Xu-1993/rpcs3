# Ascension Live Probe protocol (ABI v1)

This document is the contract between the instrumented RPCS3 build and
`ascension_live_probe.py`. The probe is inert unless
`RPCS3_ASCENSION_LIVE_PROBE=1`, and event production is additionally guarded by
the exact tuple:

- title ID: `BCAS25016`
- app version: `01.12`
- executable hash: `PPU-3a0b43e4a5f4bfea64f53612ee7c5d990f88129c`

The SPU task observer additionally authenticates the verified five-opcode
signature around `0x0928c`; LS addresses are reused by overlays, so PC and
branch target alone are not treated as a module identity.

## Control channel

Windows named pipe: `\\.\pipe\RPCS3AscensionLiveProbe`.

Commands are UTF-8 messages. Each command receives one compact JSON response.
Numeric values accept decimal or a `0x` prefix.

```text
START_CAPTURE path="C:\path with spaces\capture.bin"
STOP_CAPTURE
ARM
DISARM
SET_FILTER PPU_PC=0x73c80c PPU_CALLER=0 PPU_THREAD=0
SET_FILTER SPU_PC=0x928c SPU_TARGET=0xc490 SPU_FORMAT=0x871c0c00
SET_FILTER SPU_SEQUENCE=0 SPU_SOURCE=0 SPU_OUTPUT=0 SPU_SOURCE_START=0 SPU_SOURCE_END=0xffffffff SPU_OUTPUT_START=0 SPU_OUTPUT_END=0xffffffff FRAME_START=0 FRAME_END=0xffffffffffffffff
SET_FILTER RSX_MASK=0xc3b5 RSX_VP=0 RSX_FP=0 RSX_MIN_VERTICES=128 RSX_MAX_VERTICES=8192 RSX_MIN_INDICES=0 RSX_MAX_INDICES=0xffffffff RSX_ADDRESS=0 RSX_PRIMITIVE=0xffffffff FIRST_HITS=0
SET_SAMPLE_RATE 1
SET_MAX_EVENTS 250000
SET_REGISTER_SET PPU 0,1,3,4,5,6,7,8,9,10
SET_REGISTER_SET SPU 0,1,3,4,5,6,7,8,12,20,21,23,32,70,72,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95
SET_STACK_WINDOW 128
ADD_POINTER_FOLLOW ID=1 SOURCE=SOURCE0 SPACE=GUEST DEPTH=0 OFFSET0=0 SIZE=64
ADD_POINTER_FOLLOW ID=2 SOURCE=PPU_R3 SPACE=GUEST DEPTH=1 OFFSET0=0 OFFSET1=0 SIZE=64
REMOVE_POINTER_FOLLOW 2
ONE_SHOT_SNAPSHOT TYPE=SPU_LS COUNT=1
ONE_SHOT_SNAPSHOT TYPE=PPU_STACK COUNT=1
GET_STATS
FLUSH
```

Pointer sources are `PPU_R0`..`PPU_R31`, `SPU_R0`..`SPU_R127`,
`TASK_HEADER`, `TASK_CONTEXT`, `DMA_DESCRIPTOR`, `SOURCE0`, `SOURCE1`,
`AUXILIARY`, and `OUTPUT`. Spaces are `GUEST` and `LS`. For depth `N`,
`OFFSET0..OFFSET(N-1)` are applied before each big-endian 32-bit pointer load;
`OFFSETN` is applied to the final target. Pointer rules can be changed while
disarmed, without rebuilding or restarting RPCS3.

`PPU_PC=0x073c80c` is retained as the initial research filter because that is
where the verified SPU BRSL bytes occur in the loaded `GOWA.SELF` image. It is
not yet proven to be an executing PPU producer call site. A capture with SPU
and RSX records but no PPU records is therefore a valid negative result; the
same build can select a corrected producer PC at runtime.

## Capture file

All integers are native little-endian host integers. The file starts with a
512-byte `file_header_v1` (`ALPROBE1`). Lightweight records are exactly 1024
bytes. Snapshot records have a 64-byte common header followed by their bounded
payload; `total_size` gives the complete record size.

The 64-byte common header is:

```text
u32 magic ('ALPE')
u16 ABI version
u16 event type
u32 total size
u32 flags
u64 event sequence
u64 host timestamp in microseconds
u64 RSX frame ID
u64 producer SPU-event sequence (RSX records)
u32 thread ID
u32 guest PC
u32 caller/LR
u32 target
```

Each lightweight event then contains `u64 values[32]`, `u32 words[64]`, and
eight 56-byte pointer-follow results.

### PPU event (`type=1`)

- `values[register]`: selected PPU GPR value.
- `words[0..1]`: 64-bit selected-register mask.
- `words[2]`: SP (`r1`) guest address.
- `words[3]`: requested stack bytes.
- `words[4]`: captured stack bytes.
- `words[5]`: pointer-result count.
- `words[32..63]`: up to 128 stack bytes.

### SPU task event (`type=2`)

- `words[0..31]`: selected SPU register numbers.
- `values[0..31]`: corresponding scalar lane-3 values.
- `words[32]`: register count.
- `words[33..35]`: task-header, task-context, DMA-descriptor LS addresses.
- `words[36..44]`: task sequence, format, packed count, raw
  `task_header_lsa + 0x04` word, auxiliary EA, source EA 0/1, output
  start/end. The raw `+0x04` word is retained for ABI compatibility; it is
  not a verified guest descriptor address and must not be dereferenced or
  used as an identity candidate.
- `words[45]`: pointer-result count.
- `words[46]`: runtime MFC GET provenance mask (`bit0=task header`,
  `bit1=task context`, `bit2=DMA descriptor`).
- `words[47..49]`: guest EAs corresponding to the three LS structures when all
  required bytes have one consistent mapping in the same SPU's fixed
  16-byte-granularity provenance table and still match guest memory.
- `words[50..52]`: number of recorded GET transfers since each matched source
  transfer. For same-bias partial writes the value conservatively follows the
  oldest surviving contribution in a granule, so it may overestimate age.
  These fields are aggregate diagnostics, not a PPU owner token.
- Event flag bit 5 is set when at least one provenance-mask bit is present.

While the exact executable is authorized and capture is armed, GET-list
inlining is disabled so dynamic direct transfers, six-item list batches, and
single-item list transfers all pass through the same runtime recorder. PUT-list
optimization is unaffected. Successful GETLLAR writes are recorded as 128-byte
GET sources too.

The fixed per-SPU table uses one exact byte-valid mask for each 16-byte LS
granule. It retains a mapping until another recorded GET overwrites that LS
region or a new `ARM` epoch begins, so unrelated transfers cannot evict it.
Different-bias partial writes conservatively invalidate the other bytes in the
granule rather than fabricate a mixed mapping. Resolution covers the complete
observed structure spans (header `0x38`, context `0x0c`, descriptor `0x10`) and
requires byte-for-byte equality between the current LS span and the derived
guest span. Ordinary SPU stores are not intercepted, so these fields remain
diagnostic last-GET evidence rather than a standalone object-ownership proof.
Old ABI-v1 captures decode the reserved words as zero.

### RSX draw event (`type=3`)

- `values[0]`: index hash.
- `values[1]`: vertex-layout hash.
- `values[2]`: matched SPU event host time.
- `words[0..17]`: draw sequence, VP/FP IDs, vertex counts, first vertex,
  stream/index addresses and sizes, attribute mask, stride, primitive, command,
  index type and restart state.
- `words[18..28]`: matched task metadata and output-range overlap. In
  particular, `words[21]` mirrors the raw `task_header_lsa + 0x04` word from
  the mapped SPU event; it is not a verified EA or identity field.
- common-header `producer_sequence`: exact SPU event sequence when mapped.

### Pointer result

```text
u32 rule ID, u32 flags,
u64 source value, u64 final address, u64 content hash,
u32 requested size, u32 captured size, u8 sample[16]
```

The asynchronous writer never formats event text on an emulation/render
thread. The 4096-record event ring is bounded; overruns increment `dropped` and
never block the game. SPU-output to RSX lookup uses a guest-page index and a
4096-entry atomic publication ring, not the legacy mutex/linear scan.
