# Experimental Branch Archive

The following project branches were archived to this fork on 2026-09-02.
They preserve the source produced during the RPCS3 / *God of War: Ascension*
research period. They are evidence-bearing development snapshots, not a set of
recommended patches to combine.

| Branch | Archived HEAD | Scope | Publication status |
|---|---|---|---|
| `codex/texture-replacement-v1` | `60030bfaa9` | Vulkan texture replacement prototype | Experimental archive |
| `codex/dlss-ascension-v13-snapshot` | `d41256cb07` | Early DLSS V13 snapshot | Discontinued prototype |
| `codex/dlss-ascension-v13-handoff` | `6d90c2a3f3` | DLSS/temporal and probe handoff | Discontinued prototype |
| `codex/ascension-live-probe-v1` | `c8fcab3b26` | Runtime-configurable Ascension live probe and analyzer | Diagnostic archive |
| `codex/f9a84330-texture-dlss` | `777d66d707` | Combined texture/DLSS branch plus an RSX idle-wait experiment | Experimental archive |
| `codex/ascension-rsx-one-run-probe` | `198398dab3` | One-run RSX attribution and DMA-threshold experiments | Diagnostic archive |
| `codex/ascension-dma-threshold-8192-candidate` | `0566c9b990` | 8 KiB raw-copy threshold candidate | Rejected as a general optimization |
| `codex/ascension-dma-threshold-32768-candidate` | `9403840bb8` | 32 KiB raw-copy threshold candidate | Rejected as a general optimization |
| `codex/h-fps70-01` | `0361b66109` | Narrow texture-semaphore fallback-label experiment | Rejected experiment |
| `codex/h-s2-rsx-sync-poll-001` | `bb715bcdc4` | Current combined source archive and RSX diagnostic work | Diagnostic archive; no validated 70 FPS result |

The common source lineage and exact experiment boundaries differ by branch.
Do not infer that a later branch supersedes every earlier prototype. Consult the
commit history and the published companion repositories:

- [Ascension performance research lab](https://github.com/Pengfei-Xu-1993/rpcs3-gow-ascension-performance-lab)
- [God of War mesh-fix sources](https://github.com/Pengfei-Xu-1993/rpcs3-god-of-war-mesh-fixes)

All project-specific work listed here is covered by
[AI_DISCLOSURE.md](AI_DISCLOSURE.md). Upstream RPCS3 and third-party code are
not covered by that authorship statement.
