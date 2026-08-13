#pragma once

#include "util/types.hpp"

#include <array>
#include <atomic>
#include <string_view>

class ppu_thread;
class spu_thread;

namespace ascension::live_probe
{
	constexpr u32 event_magic = 0x45504c41; // 'ALPE' in little-endian files
	constexpr u16 event_abi_version = 1;
	constexpr u32 event_record_size = 1024;
	constexpr std::string_view supported_title_id = "BCAS25016";
	constexpr std::string_view supported_app_version = "01.12";
	constexpr std::string_view supported_executable_hash = "PPU-3a0b43e4a5f4bfea64f53612ee7c5d990f88129c";
	constexpr char runtime_gate_helper_symbol[] = "__ascension_live_probe_runtime_gate_v2";
	constexpr std::string_view ppu_cache_directory = "ascension-live-probe-v2/";
	constexpr std::string_view spu_cache_filename_suffix = "-v1-tane-live-probe-v2.dat";
	constexpr std::string_view spu_debug_cache_directory = "llvm-live-probe-v2/";

	enum class event_type : u16
	{
		ppu_call = 1,
		spu_task = 2,
		rsx_draw = 3,
		configuration = 4,
		marker = 5,
		spu_ls_snapshot = 0x100,
		ppu_stack_snapshot = 0x101,
	};

	enum event_flags : u32
	{
		event_flag_authorized = 1u << 0,
		event_flag_output_mapping = 1u << 1,
		event_flag_stack_valid = 1u << 2,
		event_flag_pointer_data = 1u << 3,
		event_flag_snapshot = 1u << 4,
	};

	struct record_header_v1
	{
		u32 magic = event_magic;
		u16 version = event_abi_version;
		u16 type = 0;
		u32 total_size = event_record_size;
		u32 flags = 0;
		u64 sequence = 0;
		u64 host_time_us = 0;
		u64 frame_id = 0;
		u64 producer_sequence = 0;
		u32 thread_id = 0;
		u32 pc = 0;
		u32 caller = 0;
		u32 target = 0;
	};
	static_assert(sizeof(record_header_v1) == 64);

	struct pointer_result_v1
	{
		u32 rule_id = 0;
		u32 flags = 0;
		u64 source_value = 0;
		u64 final_address = 0;
		u64 content_hash = 0;
		u32 requested_size = 0;
		u32 captured_size = 0;
		std::array<u8, 16> sample{};
	};
	static_assert(sizeof(pointer_result_v1) == 56);

	// All lightweight records have one fixed size. Event-specific meanings for
	// values[] and words[] are documented in tools/ascension_live_probe_protocol.md
	// and decoded by tools/ascension_live_probe.py.
	struct event_record_v1
	{
		record_header_v1 header{};
		std::array<u64, 32> values{};
		std::array<u32, 64> words{};
		std::array<pointer_result_v1, 8> pointer_results{};
	};
	static_assert(sizeof(event_record_v1) == event_record_size);

	struct file_header_v1
	{
		std::array<char, 8> magic{'A', 'L', 'P', 'R', 'O', 'B', 'E', '1'};
		u32 version = event_abi_version;
		u32 header_size = 512;
		u32 event_size = sizeof(event_record_v1);
		u32 record_header_size = sizeof(record_header_v1);
		u64 start_time_us = 0;
		u64 written_events = 0;
		u64 dropped_events = 0;
		u64 written_snapshots = 0;
		std::array<char, 16> title_id{};
		std::array<char, 16> app_version{};
		std::array<char, 64> executable_hash{};
		std::array<char, 64> build_label{};
		std::array<u8, 296> reserved{};
	};
	static_assert(sizeof(file_header_v1) == 512);

	struct output_identity
	{
		u64 event_sequence = 0;
		u64 host_time_us = 0;
		u64 frame_id = 0;
		u32 task_sequence = 0;
		u32 task_format = 0;
		u32 packed_count = 0;
		// Raw big-endian word at task_header_lsa + 0x04. Earlier probe
		// revisions mislabeled this as a guest descriptor address; it is not
		// an identity field and must not be dereferenced as one.
		u32 task_header_word_04 = 0;
		u32 source_ea_0 = 0;
		u32 source_ea_1 = 0;
		u32 auxiliary_ea = 0;
		u32 output_ea_base = 0;
		u32 output_ea_end = 0;
		u32 output_relative_offset = 0;
		u32 overlap_bytes = 0;
	};

	struct rsx_draw_event
	{
		u64 frame_id = 0;
		u32 draw_sequence = 0;
		u32 vertex_program_id = 0;
		u32 fragment_program_id = 0;
		u32 vertex_draw_count = 0;
		u32 stream_vertex_count = 0;
		u32 first_vertex = 0;
		u32 stream_address = 0;
		u32 stream_size = 0;
		u32 index_address = 0;
		u32 index_count = 0;
		u64 index_hash = 0;
		u64 layout_hash = 0;
		u16 attribute_mask = 0;
		u8 stride = 0;
		u8 primitive = 0;
		u8 command = 0;
		u8 index_type = 0xff;
		bool indexed_constants = false;
		bool restart_index_enabled = false;
		u32 restart_index = umax;
		output_identity task{};
		bool task_valid = false;
	};

	// This environment switch selects a separate instrumented guest-code cache.
	// It is immutable for one RPCS3 process; all filters below remain runtime
	// configurable through the named-pipe controller.
	bool bootstrap_enabled();

	// Called for every main PPU executable load. Authorization is true only for
	// BCAS25016 v1.12 and the exact verified GOWA.SELF hash.
	void authorize_executable(std::string_view executable_hash);
	bool authorized();

	// PPU JIT blocks load this packed gate directly. Bit 63 is ARM, the low
	// 32 bits contain the currently selected guest call-site PC.
	std::atomic<u64>* ppu_gate_address();
	u64 runtime_gate_value();
	void observe_ppu_call(ppu_thread* ppu, u32 pc, u32 target, u64 caller_lr);

	// The existing verified SPU BRSL site calls this lightweight observer. The
	// callback is present in the dedicated cache once, while task-format and
	// sampling decisions are runtime state.
	void observe_spu_task(
		spu_thread* spu,
		u32 pc,
		u32 target,
		u32 task_header_lsa,
		u32 task_context_lsa,
		u32 dma_descriptor_lsa,
		const u32* lane3_registers);

	// Page-indexed SPU-output lookup used by the RSX path. It replaces the old
	// 2048-record mutex scan for Live Probe events.
	bool identify_output_range(u32 address, u32 size, output_identity& result);

	// Cheap runtime filter used before VKDraw constructs metadata. Ordinary
	// builds return false after one immutable environment check.
	bool rsx_candidate_enabled(u16 attribute_mask, u32 vertex_count, u8 primitive);
	void observe_rsx_draw(const rsx_draw_event& draw);
	u32 report_frame(u64 frame_id);

	// Explicit shutdown is optional; the process-lifetime state also drains and
	// closes its capture from its destructor.
	void shutdown();
}
