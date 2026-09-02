#pragma once

#include "util/types.hpp"

#include <string_view>

class spu_thread;

namespace ascension::spu_task_probe
{
	struct output_identity
	{
		u64 host_time_us = 0;
		u64 rsx_frame_id = 0;
		u32 task_sequence = 0;
		u32 task_format = 0;
		u32 packed_count = 0;
		u32 source_ea_0 = 0;
		u32 source_ea_1 = 0;
		u32 auxiliary_ea = 0;
		u32 output_ea_base = 0;
		u32 output_ea_end = 0;
		u32 output_relative_offset = 0;
		u32 overlap_bytes = 0;
	};

	constexpr std::string_view gowa_112_executable_hash = "PPU-3a0b43e4a5f4bfea64f53612ee7c5d990f88129c";

	// BCAS25016 v1.12 embeds the relevant SPU overlay as raw bytes in
	// GOWA.SELF. These are addresses in the loaded guest images, not host
	// pointers or offsets in the encrypted SELF file.
	constexpr u32 task_call_pc = 0x0928c;
	constexpr u32 task_dma_target_pc = 0x0c490;
	constexpr u32 task_mfc_put_pc = 0x0afc4;
	constexpr u32 character_task_format = 0x871c0c00;
	constexpr u32 embedded_task_call_ppu_address = 0x073c80c;
	constexpr u32 original_task_call_opcode = 0x33064080; // brsl lr,0xc490
	constexpr u32 task_trap_opcode = 0x00003f10;          // stop 0x3f10
	constexpr u32 task_trap_code = task_trap_opcode & 0x3fff;

	// Enabled only by the dedicated launcher. A normal RPCS3/DLSS launch does
	// not alter the guest image and pays only the custom STOP-code comparison.
	bool enabled();

	// Identity-only mode observes the original guest code without modifying it
	// and never exits the SPU JIT through a synthetic STOP instruction.
	bool mfc_identity_enabled();
	void observe_task_call(
		spu_thread* spu,
		u32 pc,
		u32 target,
		u32 task_header_lsa,
		u32 task_context_lsa,
		u32 dma_descriptor_lsa);

	// Retained as a diagnostic fallback for the verified output PUT site. The
	// normal low-overhead path uses observe_task_call after an in-JIT format
	// guard, so unrelated task calls never cross into host C++.
	void observe_mfc_put(
		spu_thread* spu,
		u32 pc,
		u32 address,
		u32 lsa,
		u32 size,
		u32 command,
		u32 tag,
		u32 stack_lsa,
		u32 task_context_lsa);

	// Patch the one verified instruction in the in-memory, already-decrypted
	// GOWA.SELF image. The exact executable hash and original bytes must match.
	bool install_guest_patch(std::string_view executable_hash, u8* instruction, usz size);

	// Handle the custom STOP inserted above. Returns true only when the trap was
	// authenticated and the overwritten BRSL was emulated by updating LR + PC.
	bool handle_stop(spu_thread& spu, u32 code);

	// Called from the RSX thread. Arms captures from a marker file and writes a
	// completed fixed-size packet away from the SPU hot path.
	void report(u64 rsx_frame_id);

	// Resolve an RSX vertex-stream range back to the most recent verified
	// Ascension SPU skinning task that produced it. The source EA and packed task
	// shape remain stable while the output EA rotates between frames.
	bool identify_output_range(u32 address, u32 size, output_identity& result);
}
