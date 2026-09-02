#include "stdafx.h"
#include "AscensionSpuTaskProbe.h"
#include "SPUThread.h"
#include "timers.hpp"

#include "Utilities/File.h"
#include "util/logs.hpp"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

LOG_CHANNEL(ascension_spu_task_log, "Ascension SPU Task Probe");

namespace ascension::spu_task_probe
{
	namespace
	{
		constexpr u32 capture_version = 2;
		constexpr u32 record_limit = 4096;
		constexpr u32 snapshot_interval = 64;
		constexpr u32 snapshot_limit = record_limit / snapshot_interval;
		constexpr u32 local_store_size = 0x40000;
		constexpr u32 task_header_bytes = 0x100;
		constexpr u32 task_context_bytes = 0x200;
		constexpr u32 dma_stack_bytes = 0x100;
		constexpr u32 output_sample_bytes = 0x400;
		constexpr u32 live_output_limit = 2048;
		constexpr u64 live_output_max_age_us = 5'000'000;

		enum capture_phase : u32
		{
			capture_waiting = 0,
			capture_collecting = 1,
			capture_ready = 2,
			capture_writing = 3,
		};

		struct task_record
		{
			u64 sequence = 0;
			u64 host_time_us = 0;
			u64 rsx_frame_id = 0;
			u32 spu_id = 0;
			u32 lv2_id = 0;
			u32 spu_index = 0;
			u32 pc = 0;
			u32 spurs_address = 0;
			u32 reserved = 0;
			u32 task_header_lsa = 0;
			u32 task_context_lsa = 0;
			u32 dma_stack_lsa = 0;
			u32 output_sample_lsa = 0;
			u32 task_sequence = 0;
			u32 task_format = 0;
			u32 packed_count = 0;
			// Raw word at task_header_lsa + 0x04. Preserve packet layout, but do
			// not interpret or export it as a guest descriptor address.
			u32 task_header_word_04 = 0;
			u32 auxiliary_ea = 0;
			u32 source_ea_0 = 0;
			u32 source_ea_1 = 0;
			u32 output_ea_base = 0;
			u32 output_ea_end = 0;
			u32 resource_ea = 0;
			u32 resource_offset_0 = 0;
			u32 resource_offset_1 = 0;
			u32 chunk_offset = 0;
			u32 output_lsa_base = 0;
			u32 output_lsa_data = 0;
			u32 output_lsa_sample = 0;
			u32 output_lsa_end = 0;
			u32 output_dma_ea = 0;
			u32 output_span_after_chunk = 0;
			u32 stack_previous_ea = 0;
			u32 metadata_flags = 0;
			std::array<std::array<u32, 4>, 128> gprs{};
			std::array<u8, task_header_bytes> task_header{};
			std::array<u8, task_context_bytes> task_context{};
			std::array<u8, dma_stack_bytes> dma_stack{};
			std::array<u8, output_sample_bytes> output_sample{};
		};
		static_assert(sizeof(std::array<std::array<u32, 4>, 128>) == sizeof(v128) * 128);

		struct ls_snapshot
		{
			u32 record_index = 0;
			std::array<u32, 3> reserved{};
			std::array<u8, local_store_size> bytes{};
		};

		struct live_output_record
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
		};

		struct capture_file_header
		{
			std::array<char, 8> magic{'G', 'O', 'W', 'A', 'T', 'S', 'K', '2'};
			u32 version = capture_version;
			u32 header_size = sizeof(capture_file_header);
			u32 record_size = sizeof(task_record);
			u32 record_count = 0;
			u32 snapshot_size = sizeof(ls_snapshot);
			u32 snapshot_count = 0;
			u32 local_store_bytes = local_store_size;
			u32 task_pc = task_call_pc;
			u32 dma_target_pc = task_dma_target_pc;
			u32 original_opcode = original_task_call_opcode;
			u32 trap_opcode = task_trap_opcode;
			u32 task_header_size = task_header_bytes;
			u32 task_context_size = task_context_bytes;
			u32 dma_stack_size = dma_stack_bytes;
			u32 output_sample_size = output_sample_bytes;
			u64 observed_traps = 0;
			std::array<u32, 8> reserved{};
		};

		struct probe_state
		{
			std::array<task_record, record_limit> records{};
			std::array<ls_snapshot, snapshot_limit> snapshots{};
			std::atomic<u32> phase{capture_waiting};
			std::atomic<u32> reserved_records{0};
			std::atomic<u32> completed_records{0};
			std::atomic<u64> observed_traps{0};
			std::atomic<u64> current_rsx_frame_id{0};
			std::mutex live_output_mutex;
			std::array<live_output_record, live_output_limit> live_outputs{};
			u32 live_output_cursor = 0;
			u32 live_output_count = 0;
			std::atomic<u64> live_outputs_published{0};
			std::atomic<u64> identity_callbacks{0};
			std::atomic<u64> identity_auth_rejected{0};
			std::atomic<u64> identity_metadata_rejected{0};
			u64 last_identity_report_frame = 0;
			bool identity_announced = false;
			u32 capture_number = 0;
			bool waiting_announced = false;
		};

		probe_state& state()
		{
			static const std::unique_ptr<probe_state> value = std::make_unique<probe_state>();
			return *value;
		}

		std::string arm_path()
		{
			return fs::get_executable_dir() + "Ascension-SPU-Task-v2.arm";
		}

		std::string capture_stem(u32 capture_number)
		{
			return fs::get_executable_dir() + fmt::format("Ascension-SPU-Task-v2-capture-%03u", capture_number);
		}

		u32 find_available_capture_number(u32 first)
		{
			for (u32 number = std::max(1u, first); number < 10000; ++number)
			{
				if (!fs::is_file(capture_stem(number) + ".bin") && !fs::is_file(capture_stem(number) + "-summary.txt"))
					return number;
			}

			return 9999;
		}

		bool ls_range_valid(u32 address, u32 size)
		{
			return address <= local_store_size && size <= local_store_size - address;
		}

		template <usz Size>
		bool copy_ls(std::array<u8, Size>& destination, const spu_thread& spu, u32 address)
		{
			if (!ls_range_valid(address, static_cast<u32>(Size)))
				return false;

			std::memcpy(destination.data(), spu.ls + address, Size);
			return true;
		}

		bool read_ls_u32(const spu_thread& spu, u32 address, u32& value)
		{
			if (!ls_range_valid(address, sizeof(u32)))
				return false;

			value = spu._ref<u32>(address);
			return true;
		}

		void publish_live_output(probe_state& probe, const live_output_record& record)
		{
			std::lock_guard lock(probe.live_output_mutex);
			probe.live_outputs[probe.live_output_cursor] = record;
			probe.live_output_cursor = (probe.live_output_cursor + 1) % live_output_limit;
			probe.live_output_count = std::min(probe.live_output_count + 1, live_output_limit);
		}

		void observe_live_output(spu_thread& spu, probe_state& probe)
		{
			const u32 task_header_lsa = spu.gpr[84]._u32[3];
			const u32 task_context_lsa = spu.gpr[90]._u32[3];
			const u32 dma_descriptor_lsa = spu.gpr[95]._u32[3];

			live_output_record record{};
			if (!read_ls_u32(spu, task_context_lsa + 0x00, record.task_format) ||
				record.task_format != character_task_format)
			{
				return;
			}

			const bool valid =
				read_ls_u32(spu, task_header_lsa + 0x10, record.task_sequence) &&
				read_ls_u32(spu, task_header_lsa + 0x18, record.auxiliary_ea) &&
				read_ls_u32(spu, task_header_lsa + 0x30, record.source_ea_0) &&
				read_ls_u32(spu, task_header_lsa + 0x34, record.source_ea_1) &&
				read_ls_u32(spu, task_context_lsa + 0x08, record.packed_count) &&
				read_ls_u32(spu, dma_descriptor_lsa + 0x00, record.output_ea_base) &&
				read_ls_u32(spu, dma_descriptor_lsa + 0x0c, record.output_ea_end);
			if (!valid || !record.source_ea_0 || !record.packed_count ||
				record.output_ea_end <= record.output_ea_base)
			{
				return;
			}

			record.host_time_us = get_system_time();
			record.rsx_frame_id = probe.current_rsx_frame_id.load(std::memory_order_relaxed);
			publish_live_output(probe, record);
			if (probe.live_outputs_published.fetch_add(1, std::memory_order_relaxed) == 0)
			{
				ascension_spu_task_log.success(
					"Verified character task identity stream is live (format=0x%08x source=0x%08x output=0x%08x-0x%08x).",
					record.task_format,
					record.source_ea_0,
					record.output_ea_base,
					record.output_ea_end);
			}
		}

		bool write_capture(probe_state& probe)
		{
			probe.capture_number = find_available_capture_number(probe.capture_number + 1);
			const std::string stem = capture_stem(probe.capture_number);
			const std::string packet_path = stem + ".bin";
			const std::string csv_path = stem + "-key-registers.csv";
			const std::string summary_path = stem + "-summary.txt";

			capture_file_header header{};
			header.record_count = record_limit;
			header.snapshot_count = snapshot_limit;
			header.observed_traps = probe.observed_traps.load(std::memory_order_relaxed);

			fs::file packet(packet_path, fs::rewrite);
			bool packet_ok = packet && packet.write(&header, sizeof(header)) == sizeof(header);
			packet_ok = packet_ok && packet.write(probe.records.data(), sizeof(probe.records)) == sizeof(probe.records);
			packet_ok = packet_ok && packet.write(probe.snapshots.data(), sizeof(probe.snapshots)) == sizeof(probe.snapshots);

			constexpr std::array<u32, 36> key_registers{
				0, 1, 3, 4, 5, 6, 7, 8, 12, 20, 21, 23, 32,
				70, 72, 79,
				80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
				90, 91, 92, 93, 94, 95, 96, 97, 98, 126};
			std::string csv =
				"sequence,host_time_us,rsx_frame_id,spu_id,lv2_id,spu_index,pc,spurs_address,"
				"task_sequence,task_format,packed_count,task_header_word_04,auxiliary_ea,source_ea_0,source_ea_1,"
				"output_ea_base,output_ea_end,resource_ea,resource_offset_0,resource_offset_1,chunk_offset,"
				"output_lsa_base,output_lsa_data,output_lsa_sample,output_lsa_end,output_dma_ea,output_span_after_chunk,"
				"stack_previous_ea,metadata_flags";
			for (const u32 reg : key_registers)
				fmt::append(csv, ",r%u_lane3", reg);
			fmt::append(csv, "\n");
			csv.reserve(record_limit * 960);

			for (const auto& record : probe.records)
			{
				fmt::append(csv,
					"%llu,%llu,%llu,0x%08x,0x%08x,%u,0x%05x,0x%08x,"
					"0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,"
					"0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x,"
					"0x%05x,0x%05x,0x%05x,0x%05x,0x%08x,0x%08x,0x%08x,0x%08x",
					record.sequence, record.host_time_us, record.rsx_frame_id,
					record.spu_id, record.lv2_id, record.spu_index, record.pc, record.spurs_address,
					record.task_sequence, record.task_format, record.packed_count,
					record.task_header_word_04, record.auxiliary_ea, record.source_ea_0, record.source_ea_1,
					record.output_ea_base, record.output_ea_end, record.resource_ea,
					record.resource_offset_0, record.resource_offset_1, record.chunk_offset,
					record.output_lsa_base, record.output_lsa_data, record.output_lsa_sample, record.output_lsa_end,
					record.output_dma_ea, record.output_span_after_chunk,
					record.stack_previous_ea, record.metadata_flags);
				for (const u32 reg : key_registers)
					fmt::append(csv, ",0x%08x", record.gprs[reg][3]);
				fmt::append(csv, "\n");
			}

			const bool csv_ok = fs::write_file(csv_path, fs::rewrite, csv);
			const std::string summary = fmt::format(
				"RPCS3 God of War: Ascension SPU Task Probe v2\n"
				"================================================\n"
				"title=God of War: Ascension\n"
				"serial=BCAS25016\n"
				"game_version=01.12\n"
				"executable_hash=%s\n"
				"embedded_overlay_ppu_address=0x00733580\n"
				"embedded_overlay_spu_range=0x04000-0x10480\n"
				"task_call_ppu_address=0x%08x\n"
				"task_call_spu_pc=0x%05x\n"
				"task_dma_target_spu_pc=0x%05x\n"
				"original_opcode=0x%08x\n"
				"trap_opcode=0x%08x\n"
				"records=%u\n"
				"ls_snapshots=%u\n"
				"snapshot_interval=%u\n"
				"task_header_bytes_per_record=%u\n"
				"task_context_bytes_per_record=%u\n"
				"dma_stack_bytes_per_record=%u\n"
				"output_sample_bytes_per_record=%u\n"
				"observed_traps=%llu\n"
				"packet=%s\n"
				"key_registers=%s\n\n"
				"Each binary task_record contains frame/time metadata, decoded DMA fields, all 128 SPU registers,\n"
				"the task header, task context, DMA stack window, and a sample from the pending output block.\n"
				"Each LS snapshot is associated with record_index and contains the raw 256 KiB local store.\n"
				"The custom STOP emulates the overwritten BRSL by setting r0 to 0x09290 and PC to 0x0c490.\n",
				gowa_112_executable_hash,
				embedded_task_call_ppu_address,
				task_call_pc,
				task_dma_target_pc,
				original_task_call_opcode,
				task_trap_opcode,
				record_limit,
				snapshot_limit,
				snapshot_interval,
				task_header_bytes,
				task_context_bytes,
				dma_stack_bytes,
				output_sample_bytes,
				header.observed_traps,
				packet_ok ? "saved" : "FAILED",
				csv_ok ? "saved" : "FAILED");
			const bool summary_ok = fs::write_file(summary_path, fs::rewrite, summary);

			ascension_spu_task_log.success(
				"Capture %u written: packet=%s csv=%s summary=%s records=%u LS_snapshots=%u directory='%s'.",
				probe.capture_number,
				packet_ok ? "saved" : "FAILED",
				csv_ok ? "saved" : "FAILED",
				summary_ok ? "saved" : "FAILED",
				record_limit,
				snapshot_limit,
				fs::get_executable_dir());
			return packet_ok && csv_ok && summary_ok;
		}
	}

	bool enabled()
	{
		static const bool value = []
		{
			const char* env = std::getenv("RPCS3_ASCENSION_SPU_TASK_PROBE");
			return env && env[0] && std::strcmp(env, "0") != 0;
		}();
		return value;
	}

	bool mfc_identity_enabled()
	{
		static const bool value = []
		{
			const char* env = std::getenv("RPCS3_ASCENSION_MFC_TASK_IDENTITY");
			return env && env[0] && std::strcmp(env, "0") != 0;
		}();
		return value;
	}

	void observe_task_call(
		spu_thread* spu,
		u32 pc,
		u32 target,
		u32 task_header_lsa,
		u32 task_context_lsa,
		u32 dma_descriptor_lsa)
	{
		if (!mfc_identity_enabled() || !spu || pc != task_call_pc || target != task_dma_target_pc)
			return;

		probe_state& probe = state();
		probe.identity_callbacks.fetch_add(1, std::memory_order_relaxed);

		// Authenticate the exact unmodified BCAS25016 v1.12 call site. Unlike
		// V9-V11, this path never replaces the BRSL with a synthetic STOP.
		if (spu->_ref<u32>(task_call_pc - 12) != 0x04002d03 ||
			spu->_ref<u32>(task_call_pc - 8) != 0x3fe02f84 ||
			spu->_ref<u32>(task_call_pc - 4) != 0x1c080085 ||
			spu->_ref<u32>(task_call_pc) != original_task_call_opcode ||
			spu->_ref<u32>(task_call_pc + 4) != 0x4020007f)
		{
			probe.identity_auth_rejected.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		if (!ls_range_valid(task_header_lsa, 0x38) ||
			!ls_range_valid(task_context_lsa, 0x0c) ||
			!ls_range_valid(dma_descriptor_lsa, 0x10))
		{
			probe.identity_metadata_rejected.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		live_output_record record{};
		const bool valid =
			read_ls_u32(*spu, task_context_lsa + 0x00, record.task_format) &&
			record.task_format == character_task_format &&
			read_ls_u32(*spu, task_context_lsa + 0x08, record.packed_count) &&
			read_ls_u32(*spu, task_header_lsa + 0x10, record.task_sequence) &&
			read_ls_u32(*spu, task_header_lsa + 0x18, record.auxiliary_ea) &&
			read_ls_u32(*spu, task_header_lsa + 0x30, record.source_ea_0) &&
			read_ls_u32(*spu, task_header_lsa + 0x34, record.source_ea_1) &&
			read_ls_u32(*spu, dma_descriptor_lsa + 0x00, record.output_ea_base) &&
			read_ls_u32(*spu, dma_descriptor_lsa + 0x0c, record.output_ea_end);
		if (!valid || !record.source_ea_0 || !record.packed_count ||
			record.output_ea_end <= record.output_ea_base)
		{
			probe.identity_metadata_rejected.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		record.host_time_us = get_system_time();
		record.rsx_frame_id = probe.current_rsx_frame_id.load(std::memory_order_relaxed);
		publish_live_output(probe, record);
		if (probe.live_outputs_published.fetch_add(1, std::memory_order_relaxed) == 0)
		{
			ascension_spu_task_log.success(
				"Verified character task-call identity stream is live without a guest STOP "
				"(format=0x%08x source=0x%08x output=0x%08x-0x%08x).",
				record.task_format,
				record.source_ea_0,
				record.output_ea_base,
				record.output_ea_end);
		}
	}

	void observe_mfc_put(
		spu_thread* spu,
		u32 pc,
		u32 address,
		u32 lsa,
		u32 size,
		u32 command,
		u32 tag,
		u32 stack_lsa,
		u32 task_context_lsa)
	{
		if (!mfc_identity_enabled() || !spu || pc != task_mfc_put_pc ||
			command != 0x20 || tag != 31 || !size || size > 0x4000)
		{
			return;
		}

		// Authenticate the exact Ascension DMA issue site captured from the
		// BCAS25016 v1.12 SPU overlay. No instruction is replaced in V12.
		if (spu->_ref<u32>(task_mfc_put_pc - 12) != 0x1801c489 ||
			spu->_ref<u32>(task_mfc_put_pc - 8) != 0x21a00987 ||
			spu->_ref<u32>(task_mfc_put_pc - 4) != 0x21a00a0c ||
			spu->_ref<u32>(task_mfc_put_pc) != 0x21a00a8b ||
			spu->_ref<u32>(task_mfc_put_pc + 4) != 0x18410403)
		{
			return;
		}

		// The original c490 stack frame retains the task-header LSA at sp+0x10.
		// r86 is passed directly by the LLVM JIT so no 128-register spill is
		// needed merely to identify this output.
		if (!ls_range_valid(stack_lsa, 0x14) || !ls_range_valid(task_context_lsa, 0x0c))
			return;

		u32 task_header_lsa = 0;
		if (!read_ls_u32(*spu, stack_lsa + 0x10, task_header_lsa))
			return;
		if (!ls_range_valid(task_header_lsa, 0x38))
			return;

		live_output_record record{};
		const bool valid =
			read_ls_u32(*spu, task_context_lsa + 0x00, record.task_format) &&
			record.task_format == character_task_format &&
			read_ls_u32(*spu, task_context_lsa + 0x08, record.packed_count) &&
			read_ls_u32(*spu, task_header_lsa + 0x10, record.task_sequence) &&
			read_ls_u32(*spu, task_header_lsa + 0x18, record.auxiliary_ea) &&
			read_ls_u32(*spu, task_header_lsa + 0x30, record.source_ea_0) &&
			read_ls_u32(*spu, task_header_lsa + 0x34, record.source_ea_1);
		const u64 output_end = static_cast<u64>(address) + size;
		if (!valid || !record.source_ea_0 || !record.packed_count ||
			output_end > umax || !ls_range_valid(lsa, size))
		{
			return;
		}

		record.output_ea_base = address;
		record.output_ea_end = static_cast<u32>(output_end);
		record.host_time_us = get_system_time();
		probe_state& probe = state();
		record.rsx_frame_id = probe.current_rsx_frame_id.load(std::memory_order_relaxed);
		publish_live_output(probe, record);
		if (probe.live_outputs_published.fetch_add(1, std::memory_order_relaxed) == 0)
		{
			ascension_spu_task_log.success(
				"Verified character MFC task identity stream is live without a guest STOP "
				"(format=0x%08x source=0x%08x put=0x%08x-0x%08x lsa=0x%05x).",
				record.task_format,
				record.source_ea_0,
				record.output_ea_base,
				record.output_ea_end,
				lsa);
		}
	}

	bool install_guest_patch(std::string_view executable_hash, u8* instruction, usz size)
	{
		if (!enabled() || executable_hash != gowa_112_executable_hash || !instruction || size < sizeof(u32))
			return false;

		constexpr std::array<u8, 4> original_bytes{0x33, 0x06, 0x40, 0x80};
		constexpr std::array<u8, 4> trap_bytes{0x00, 0x00, 0x3f, 0x10};
		if (std::memcmp(instruction, trap_bytes.data(), trap_bytes.size()) == 0)
			return true;
		if (std::memcmp(instruction, original_bytes.data(), original_bytes.size()) != 0)
		{
			ascension_spu_task_log.error(
				"Refusing task patch: GOWA.SELF hash matched but bytes at PPU 0x%08x were %02x %02x %02x %02x, expected 33 06 40 80.",
				embedded_task_call_ppu_address, instruction[0], instruction[1], instruction[2], instruction[3]);
			return false;
		}

		std::memcpy(instruction, trap_bytes.data(), trap_bytes.size());
		ascension_spu_task_log.success(
			"Installed reversible guest task trap at PPU 0x%08x (SPU PC 0x%05x, original BRSL target 0x%05x).",
			embedded_task_call_ppu_address, task_call_pc, task_dma_target_pc);
		return true;
	}

	bool handle_stop(spu_thread& spu, u32 code)
	{
		if (!enabled() || code != task_trap_code || spu.pc != task_call_pc)
			return false;

		// Authenticate the surrounding instructions as well as the trap. This
		// prevents an unrelated program using the same STOP code from ever being
		// redirected into Ascension's DMA helper.
		if (spu._ref<u32>(task_call_pc - 12) != 0x04002d03 ||
			spu._ref<u32>(task_call_pc - 8) != 0x3fe02f84 ||
			spu._ref<u32>(task_call_pc - 4) != 0x1c080085 ||
			spu._ref<u32>(task_call_pc) != task_trap_opcode ||
			spu._ref<u32>(task_call_pc + 4) != 0x4020007f)
		{
			ascension_spu_task_log.error("Rejected unauthenticated STOP 0x%x at SPU PC 0x%05x.", code, spu.pc);
			return false;
		}

		probe_state& probe = state();
		observe_live_output(spu, probe);
		if (probe.phase.load(std::memory_order_acquire) == capture_collecting)
		{
			probe.observed_traps.fetch_add(1, std::memory_order_relaxed);
			const u32 slot = probe.reserved_records.fetch_add(1, std::memory_order_relaxed);
			if (slot < record_limit)
			{
				auto& record = probe.records[slot];
				std::memset(&record, 0, sizeof(record));
				record.sequence = slot;
				record.host_time_us = get_system_time();
				record.rsx_frame_id = probe.current_rsx_frame_id.load(std::memory_order_relaxed);
				record.spu_id = spu.id;
				record.lv2_id = spu.lv2_id;
				record.spu_index = spu.index;
				record.pc = spu.pc;
				record.spurs_address = spu.spurs_addr;
				std::memcpy(record.gprs.data(), spu.gpr.data(), sizeof(record.gprs));

				record.task_header_lsa = spu.gpr[84]._u32[3];
				record.task_context_lsa = spu.gpr[90]._u32[3];
				record.dma_stack_lsa = spu.gpr[5]._u32[3];
				const u32 dma_descriptor_lsa = spu.gpr[95]._u32[3];

				if (copy_ls(record.task_header, spu, record.task_header_lsa))
					record.metadata_flags |= 1u << 0;
				if (copy_ls(record.task_context, spu, record.task_context_lsa))
					record.metadata_flags |= 1u << 1;
				if (copy_ls(record.dma_stack, spu, record.dma_stack_lsa))
					record.metadata_flags |= 1u << 2;

				bool header_ok = true;
				header_ok &= read_ls_u32(spu, record.task_header_lsa + 0x04, record.task_header_word_04);
				header_ok &= read_ls_u32(spu, record.task_header_lsa + 0x10, record.task_sequence);
				header_ok &= read_ls_u32(spu, record.task_header_lsa + 0x18, record.auxiliary_ea);
				header_ok &= read_ls_u32(spu, record.task_header_lsa + 0x30, record.source_ea_0);
				header_ok &= read_ls_u32(spu, record.task_header_lsa + 0x34, record.source_ea_1);
				if (header_ok)
					record.metadata_flags |= 1u << 4;

				bool context_ok = true;
				context_ok &= read_ls_u32(spu, record.task_context_lsa + 0x00, record.task_format);
				context_ok &= read_ls_u32(spu, record.task_context_lsa + 0x08, record.packed_count);
				context_ok &= read_ls_u32(spu, record.task_context_lsa + 0x50, record.output_lsa_base);
				context_ok &= read_ls_u32(spu, record.task_context_lsa + 0x54, record.output_lsa_data);
				context_ok &= read_ls_u32(spu, record.task_context_lsa + 0x58, record.output_lsa_end);
				if (context_ok)
					record.metadata_flags |= 1u << 5;

				bool dma_ok = true;
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x00, record.output_ea_base);
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x0c, record.output_ea_end);
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x10, record.resource_ea);
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x14, record.resource_offset_0);
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x18, record.resource_offset_1);
				dma_ok &= read_ls_u32(spu, dma_descriptor_lsa + 0x20, record.chunk_offset);
				dma_ok &= read_ls_u32(spu, record.dma_stack_lsa + 0x40, record.stack_previous_ea);
				if (dma_ok)
					record.metadata_flags |= 1u << 6;

				if (context_ok && dma_ok)
				{
					const u64 output_dma_ea = static_cast<u64>(record.output_ea_base) + record.chunk_offset;
					const u64 output_sample_lsa = static_cast<u64>(record.output_lsa_data) + record.chunk_offset;
					if (output_dma_ea <= umax)
						record.output_dma_ea = static_cast<u32>(output_dma_ea);
					if (record.output_ea_end > record.output_dma_ea)
						record.output_span_after_chunk = record.output_ea_end - record.output_dma_ea;
					if (output_sample_lsa <= umax)
					{
						record.output_sample_lsa = static_cast<u32>(output_sample_lsa);
						record.output_lsa_sample = record.output_sample_lsa;
						if (copy_ls(record.output_sample, spu, record.output_sample_lsa))
							record.metadata_flags |= 1u << 3;
					}
				}

				if ((slot % snapshot_interval) == 0)
				{
					auto& snapshot = probe.snapshots[slot / snapshot_interval];
					snapshot.record_index = slot;
					std::memcpy(snapshot.bytes.data(), spu.ls, snapshot.bytes.size());
				}

				if (probe.completed_records.fetch_add(1, std::memory_order_acq_rel) + 1 == record_limit)
				{
					probe.phase.store(capture_ready, std::memory_order_release);
					ascension_spu_task_log.notice(
						"Captured %u task calls and %u distributed LS snapshots; waiting for the RSX thread to write files.",
						record_limit, snapshot_limit);
				}
			}
		}

		// Exact emulation of `brsl lr,0xc490`: clear the other LR lanes, put
		// the return address in scalar lane 3, and redispatch at the call target.
		spu.gpr[0] = v128::from32r((task_call_pc + 4) & 0x3fffc);
		spu.pc = task_dma_target_pc;
		return true;
	}

	void report(u64 rsx_frame_id)
	{
		if (!enabled() && !mfc_identity_enabled())
			return;

		probe_state& probe = state();
		probe.current_rsx_frame_id.store(rsx_frame_id, std::memory_order_relaxed);
		if (mfc_identity_enabled())
		{
			if (!probe.identity_announced)
			{
				probe.identity_announced = true;
				ascension_spu_task_log.notice(
					"V13 no-STOP task identity is enabled; the in-JIT format guard admits only format 0x%08x.",
					character_task_format);
			}

			if (!probe.last_identity_report_frame || rsx_frame_id - probe.last_identity_report_frame >= 600)
			{
				probe.last_identity_report_frame = rsx_frame_id;
				ascension_spu_task_log.notice(
					"V13 task identity counters: callbacks=%llu published=%llu auth_rejected=%llu metadata_rejected=%llu.",
					probe.identity_callbacks.load(std::memory_order_relaxed),
					probe.live_outputs_published.load(std::memory_order_relaxed),
					probe.identity_auth_rejected.load(std::memory_order_relaxed),
					probe.identity_metadata_rejected.load(std::memory_order_relaxed));
			}
		}
		if (!enabled())
			return;

		const u32 phase = probe.phase.load(std::memory_order_acquire);
		if (phase == capture_waiting)
		{
			if (!probe.waiting_announced)
			{
				probe.waiting_announced = true;
				ascension_spu_task_log.notice(
					"Guest task trap is active and waiting for arm file '%s'. No task payload is copied before arming.",
					arm_path());
			}

			const std::string path = arm_path();
			if (!fs::is_file(path))
				return;

			fs::remove_file(path);
			probe.reserved_records.store(0, std::memory_order_relaxed);
			probe.completed_records.store(0, std::memory_order_relaxed);
			probe.observed_traps.store(0, std::memory_order_relaxed);
			probe.phase.store(capture_collecting, std::memory_order_release);
			ascension_spu_task_log.notice(
				"Capture armed: collecting %u complete task register sets and %u distributed LS snapshots.",
				record_limit, snapshot_limit);
			return;
		}

		u32 expected = capture_ready;
		if (phase != capture_ready || !probe.phase.compare_exchange_strong(
			expected, capture_writing, std::memory_order_acq_rel))
		{
			return;
		}

		write_capture(probe);
		probe.waiting_announced = false;
		probe.phase.store(capture_waiting, std::memory_order_release);
	}

	bool identify_output_range(u32 address, u32 size, output_identity& result)
	{
		result = {};
		if ((!enabled() && !mfc_identity_enabled()) || !size)
			return false;

		const u64 query_start = address;
		const u64 query_end = query_start + size;
		const u64 now = get_system_time();
		probe_state& probe = state();
		std::lock_guard lock(probe.live_output_mutex);

		u64 best_overlap = 0;
		for (u32 offset = 0; offset < probe.live_output_count; ++offset)
		{
			const u32 index = (probe.live_output_cursor + live_output_limit - 1 - offset) % live_output_limit;
			const auto& candidate = probe.live_outputs[index];
			if (!candidate.host_time_us || now < candidate.host_time_us ||
				now - candidate.host_time_us > live_output_max_age_us)
			{
				continue;
			}

			const u64 overlap_start = std::max<u64>(query_start, candidate.output_ea_base);
			const u64 overlap_end = std::min<u64>(query_end, candidate.output_ea_end);
			if (overlap_end <= overlap_start)
				continue;

			const u64 overlap = overlap_end - overlap_start;
			if (overlap <= best_overlap)
				continue;

			best_overlap = overlap;
			result.host_time_us = candidate.host_time_us;
			result.rsx_frame_id = candidate.rsx_frame_id;
			result.task_sequence = candidate.task_sequence;
			result.task_format = candidate.task_format;
			result.packed_count = candidate.packed_count;
			result.source_ea_0 = candidate.source_ea_0;
			result.source_ea_1 = candidate.source_ea_1;
			result.auxiliary_ea = candidate.auxiliary_ea;
			result.output_ea_base = candidate.output_ea_base;
			result.output_ea_end = candidate.output_ea_end;
			result.output_relative_offset = static_cast<u32>(overlap_start - candidate.output_ea_base);
			result.overlap_bytes = static_cast<u32>(overlap);
		}

		return best_overlap != 0;
	}
}
