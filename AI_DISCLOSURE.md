# AI Authorship and Human Contribution Disclosure

This fork contains experimental branches created for RPCS3 research involving
*God of War: Ascension*, texture replacement, temporal upscaling, and RSX/SPU
performance diagnostics.

## AI-authored project work

All project-specific source code, scripts, tests, build integration,
documentation, and experimental instrumentation introduced on the relevant
`codex/*` branches were generated and implemented by **OpenAI Codex** under
human direction.

The human project owner, Pengfei Xu (`Pengfei-Xu-1993`), did **not** write the
project-specific code. Git commit metadata may show the human maintainer's
configured name and email for repository traceability; that metadata must not
be interpreted as a claim of human code authorship.

This disclosure applies only to the project-specific changes in this fork. It
does not apply to upstream RPCS3 code or third-party dependencies. Their
authorship, copyright, and licenses remain unchanged.

## Human contribution

The human project owner was responsible for:

- defining goals, constraints, experiment boundaries, and acceptance criteria;
- providing the local hardware and software test environment;
- providing a legally obtained game installation without distributing it;
- creating and selecting savestates and controller/TAS recordings;
- performing manual gameplay, visual, crash, stability, and performance checks;
- reviewing observed results and deciding which experiments to continue or
  reject; and
- authorizing publication of the project source and research tooling.

The human contribution was project direction, test operation, empirical
feedback, review, and publication approval—not implementation of the
project-specific code.

## Status and limitations

These branches are experimental research archives. They are not official RPCS3
releases, are not endorsed by the RPCS3 project, and must not be treated as
upstream-ready performance or compatibility patches. Different experiments
have different evidence levels; the presence of code does not mean that every
experiment improved performance or passed final validation. Several directions
were explicitly rejected after measurement.

No commercial game files, firmware, decryption keys, savestates, caches, raw
captures, NVIDIA DLSS SDK files, or prebuilt DLSS-enabled binaries are included.
Users must provide their own legally obtained game data and any external SDKs.

If any work from these branches is proposed to upstream RPCS3, the human
contributor must personally review and understand it, communicate with the
maintainers, and include the required AI involvement and human-testing
disclosure in the pull request description. This file does not replace that
per-pull-request disclosure.

## 中文说明

本 fork 相关 `codex/*` 实验分支中新增加的代码、脚本、测试、构建集成、文档和
探针均由 OpenAI Codex 在人工指导下生成并实现。人类项目负责人 Pengfei Xu
（`Pengfei-Xu-1993`）没有编写这些项目新增代码；其工作是确定目标和边界、提供
测试环境与合法取得的游戏安装、制作存档及手柄录制、进行人工游玩和稳定性/性能
验证、审核实验结果、决定继续或否定研究方向，并授权公开发布。RPCS3 上游代码及
第三方依赖不属于本声明所称的 AI 生成内容。
