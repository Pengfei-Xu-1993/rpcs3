#include "stdafx.h"
#include "AscensionLiveProbe.h"

#include "PPUThread.h"
#include "SPUThread.h"
#include "Emu/Memory/vm.h"
#include "Emu/System.h"
#include "Utilities/File.h"
#include "timers.hpp"
#include "util/logs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

LOG_CHANNEL(ascension_live_probe_log, "Ascension Live Probe");

namespace ascension::live_probe
{
	namespace
	{
		constexpr u32 event_ring_size = 4096;
		constexpr u32 event_ring_mask = event_ring_size - 1;
		constexpr u32 output_ring_size = 4096;
		constexpr u32 output_ring_mask = output_ring_size - 1;
		constexpr u32 guest_page_count = 1u << 20;
		constexpr u32 snapshot_slot_count = 2;
		constexpr u32 snapshot_capacity = 0x40000;
		constexpr u32 max_pointer_rules = 8;
		constexpr u64 output_max_age_us = 5'000'000;
		constexpr u64 armed_bit = 1ull << 63;
		constexpr u32 default_ppu_pc = 0x073c80c;
		constexpr u32 expected_spu_pc = 0x0928c;
		constexpr u32 expected_spu_target = 0x0c490;
		constexpr u32 expected_spu_call_opcode = 0x33064080;

		enum class pointer_source : u32
		{
			ppu_gpr = 1,
			spu_gpr_lane3 = 2,
			task_header = 3,
			task_context = 4,
			dma_descriptor = 5,
			source_0 = 6,
			source_1 = 7,
			auxiliary = 8,
			output = 9,
		};

		enum class pointer_space : u32
		{
			guest = 1,
			spu_ls = 2,
		};

		struct pointer_rule
		{
			std::atomic<u32> enabled{0};
			u32 id = 0;
			pointer_source source = pointer_source::ppu_gpr;
			pointer_space space = pointer_space::guest;
			u32 source_index = 0;
			u32 depth = 0;
			std::array<s32, 5> offsets{};
			u32 capture_size = 16;
		};

		struct runtime_filters
		{
			std::atomic<u32> ppu_pc{default_ppu_pc};
			std::atomic<u32> ppu_caller{0};
			std::atomic<u32> ppu_thread{0};
			std::atomic<u32> spu_pc{expected_spu_pc};
			std::atomic<u32> spu_target{expected_spu_target};
			std::atomic<u32> spu_format{0};
			std::atomic<u32> spu_thread{0};
			std::atomic<u32> spu_task_sequence{0};
			std::atomic<u32> spu_source{0};
			std::atomic<u32> spu_output{0};
			std::atomic<u32> spu_source_start{0};
			std::atomic<u32> spu_source_end{umax};
			std::atomic<u32> spu_output_start{0};
			std::atomic<u32> spu_output_end{umax};
			std::atomic<u64> frame_start{0};
			std::atomic<u64> frame_end{umax};
			std::atomic<u32> rsx_attribute_mask{0};
			std::atomic<u32> rsx_vertex_program{0};
			std::atomic<u32> rsx_fragment_program{0};
			std::atomic<u32> rsx_min_vertices{0};
			std::atomic<u32> rsx_max_vertices{umax};
			std::atomic<u32> rsx_min_indices{0};
			std::atomic<u32> rsx_max_indices{umax};
			std::atomic<u32> rsx_address{0};
			std::atomic<u32> rsx_primitive{umax};
		};

		struct event_slot
		{
			std::atomic<u64> sequence{0};
			event_record_v1 record{};
		};

		struct output_slot
		{
			std::atomic<u64> stamp{0};
			// Atomic scalar fields make the page-index lookup data-race-free even
			// when a 4096-entry slot wraps while the RSX thread is reading it.
			std::array<std::atomic<u64>, 13> values{};
		};

		struct snapshot_slot
		{
			std::atomic<u32> state{0}; // 0=free, 1=writer owns, 2=ready
			record_header_v1 header{};
			u32 payload_size = 0;
			std::array<u8, snapshot_capacity> payload{};
		};

		struct probe_state
		{
			std::atomic<bool> executable_authorized{false};
			std::atomic<bool> armed{false};
			std::atomic<bool> capturing{false};
			std::atomic<bool> stopping{false};
			std::atomic<u64> ppu_gate{0};
			std::atomic<u64> current_frame{0};
			std::atomic<u32> current_rsx_draw_sequence{0};
			std::atomic<u64> next_event_sequence{1};
			std::atomic<u64> observed_events{0};
			std::atomic<u64> accepted_events{0};
			std::atomic<u64> dropped_events{0};
			std::atomic<u64> filtered_events{0};
			std::atomic<u64> written_events{0};
			std::atomic<u64> written_snapshots{0};
			std::atomic<u32> sample_rate{1};
			std::atomic<u64> max_events{0};
			std::atomic<u64> ppu_register_mask{~0ull};
			std::array<std::atomic<u64>, 2> spu_register_mask{{0, 0}};
			std::atomic<u32> stack_window{128};
			std::atomic<u32> snapshot_kind{0};
			std::atomic<u32> snapshot_remaining{0};
			runtime_filters filters{};
			std::array<pointer_rule, max_pointer_rules> pointer_rules{};

			std::array<event_slot, event_ring_size> events{};
			std::atomic<u64> enqueue_pos{0};
			std::atomic<u64> dequeue_pos{0};
			std::array<snapshot_slot, snapshot_slot_count> snapshots{};

			std::array<output_slot, output_ring_size> outputs{};
			std::atomic<u64> next_output_id{1};
			std::unique_ptr<std::atomic<u64>[]> output_pages;

			fs::file capture_file{};
			file_header_v1 capture_header{};
			std::string capture_path{};
			std::jthread worker{};

			probe_state()
				: output_pages(std::make_unique<std::atomic<u64>[]>(guest_page_count))
			{
				for (u64 i = 0; i < event_ring_size; ++i)
					events[i].sequence.store(i, std::memory_order_relaxed);
				constexpr std::array<u32, 32> default_spu_registers{
					0, 1, 3, 4, 5, 6, 7, 8, 12, 20, 21, 23, 32, 70, 72, 79,
					80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
				u64 masks[2]{};
				for (const u32 reg : default_spu_registers)
					masks[reg / 64] |= 1ull << (reg % 64);
				spu_register_mask[0].store(masks[0], std::memory_order_relaxed);
				spu_register_mask[1].store(masks[1], std::memory_order_relaxed);
				for (u32 i = 0; i < guest_page_count; ++i)
					output_pages[i].store(0, std::memory_order_relaxed);
				worker = std::jthread([this](std::stop_token token) { worker_main(token); });
			}

			~probe_state()
			{
				stopping.store(true, std::memory_order_release);
				worker.request_stop();
				if (worker.joinable())
					worker.join();
				close_capture();
			}

			bool push(const event_record_v1& record)
			{
				u64 position = enqueue_pos.load(std::memory_order_relaxed);
				for (;;)
				{
					event_slot& slot = events[position & event_ring_mask];
					const u64 sequence = slot.sequence.load(std::memory_order_acquire);
					const s64 difference = static_cast<s64>(sequence - position);
					if (difference == 0)
					{
						if (enqueue_pos.compare_exchange_weak(position, position + 1, std::memory_order_relaxed))
						{
							slot.record = record;
							slot.sequence.store(position + 1, std::memory_order_release);
							return true;
						}
					}
					else if (difference < 0)
					{
						dropped_events.fetch_add(1, std::memory_order_relaxed);
						return false;
					}
					else
					{
						position = enqueue_pos.load(std::memory_order_relaxed);
					}
				}
			}

			bool pop(event_record_v1& record)
			{
				u64 position = dequeue_pos.load(std::memory_order_relaxed);
				for (;;)
				{
					event_slot& slot = events[position & event_ring_mask];
					const u64 sequence = slot.sequence.load(std::memory_order_acquire);
					const s64 difference = static_cast<s64>(sequence - (position + 1));
					if (difference == 0)
					{
						if (dequeue_pos.compare_exchange_weak(position, position + 1, std::memory_order_relaxed))
						{
							record = slot.record;
							slot.sequence.store(position + event_ring_size, std::memory_order_release);
							return true;
						}
					}
					else if (difference < 0)
					{
						return false;
					}
					else
					{
						position = dequeue_pos.load(std::memory_order_relaxed);
					}
				}
			}

			void reset_ring()
			{
				// START_CAPTURE is accepted only while disarmed, so no producer can
				// race this bounded reset.
				enqueue_pos.store(0, std::memory_order_relaxed);
				dequeue_pos.store(0, std::memory_order_relaxed);
				for (u64 i = 0; i < event_ring_size; ++i)
					events[i].sequence.store(i, std::memory_order_relaxed);
			}

			void close_capture()
			{
				capturing.store(false, std::memory_order_release);
				if (!capture_file)
					return;

				drain();
				capture_header.written_events = written_events.load(std::memory_order_relaxed);
				capture_header.dropped_events = dropped_events.load(std::memory_order_relaxed);
				capture_header.written_snapshots = written_snapshots.load(std::memory_order_relaxed);
				capture_file.seek(0);
				capture_file.write(&capture_header, sizeof(capture_header));
				capture_file.sync();
				capture_file.close();
				ascension_live_probe_log.success(
					"Capture closed: events=%llu snapshots=%llu dropped=%llu path='%s'.",
					capture_header.written_events,
					capture_header.written_snapshots,
					capture_header.dropped_events,
					capture_path);
			}

			bool open_capture(std::string path)
			{
				if (armed.load(std::memory_order_acquire))
					return false;
				close_capture();
				if (path.empty())
				{
					if (const char* env = std::getenv("RPCS3_ASCENSION_LIVE_PROBE_OUTPUT"); env && env[0])
						path = env;
					else
						path = fs::get_executable_dir() + "Ascension-Live-Probe.bin";
				}

				capture_file.open(path, fs::rewrite);
				if (!capture_file)
					return false;

				capture_path = std::move(path);
				capture_header = {};
				capture_header.start_time_us = get_system_time();
				auto copy_text = [](auto& destination, std::string_view source)
				{
					std::memcpy(destination.data(), source.data(), std::min(destination.size() - 1, source.size()));
				};
				copy_text(capture_header.title_id, Emu.GetTitleID());
				copy_text(capture_header.app_version, Emu.GetAppVersion());
				copy_text(capture_header.executable_hash, supported_executable_hash);
				copy_text(capture_header.build_label, "Ascension Live Probe ABI v1");
				if (capture_file.write(&capture_header, sizeof(capture_header)) != sizeof(capture_header))
				{
					capture_file.close();
					return false;
				}

				observed_events.store(0, std::memory_order_relaxed);
				accepted_events.store(0, std::memory_order_relaxed);
				dropped_events.store(0, std::memory_order_relaxed);
				filtered_events.store(0, std::memory_order_relaxed);
				written_events.store(0, std::memory_order_relaxed);
				written_snapshots.store(0, std::memory_order_relaxed);
				next_event_sequence.store(1, std::memory_order_relaxed);
				reset_ring();
				capturing.store(true, std::memory_order_release);
				ascension_live_probe_log.success("Capture opened at '%s'.", capture_path);
				return true;
			}

			void drain()
			{
				if (!capture_file)
					return;

				event_record_v1 record{};
				while (pop(record))
				{
					if (capture_file.write(&record, sizeof(record)) == sizeof(record))
						written_events.fetch_add(1, std::memory_order_relaxed);
				}

				for (auto& slot : snapshots)
				{
					u32 expected = 2;
					if (!slot.state.compare_exchange_strong(expected, 1, std::memory_order_acquire))
						continue;
					const bool ok = capture_file.write(&slot.header, sizeof(slot.header)) == sizeof(slot.header) &&
						capture_file.write(slot.payload.data(), slot.payload_size) == slot.payload_size;
					if (ok)
						written_snapshots.fetch_add(1, std::memory_order_relaxed);
					slot.state.store(0, std::memory_order_release);
				}
			}

			void update_ppu_gate()
			{
				const bool enable = armed.load(std::memory_order_acquire) &&
					capturing.load(std::memory_order_acquire) &&
					executable_authorized.load(std::memory_order_acquire);
				const u64 pc = filters.ppu_pc.load(std::memory_order_relaxed);
				ppu_gate.store((enable ? armed_bit : 0) | pc, std::memory_order_release);
			}

			static std::string trim(std::string value)
			{
				const auto first = value.find_first_not_of(" \t\r\n");
				if (first == std::string::npos)
					return {};
				const auto last = value.find_last_not_of(" \t\r\n");
				return value.substr(first, last - first + 1);
			}

			static std::string upper(std::string value)
			{
				std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
				return value;
			}

			static bool parse_u64(std::string_view text, u64& value)
			{
				int base = 10;
				if (text.starts_with("0x") || text.starts_with("0X"))
				{
					text.remove_prefix(2);
					base = 16;
				}
				const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value, base);
				return error == std::errc{} && end == text.data() + text.size();
			}

			static bool parse_s64(std::string_view text, s64& value)
			{
				int base = 10;
				bool negative = false;
				if (text.starts_with('-'))
				{
					negative = true;
					text.remove_prefix(1);
				}
				if (text.starts_with("0x") || text.starts_with("0X"))
				{
					text.remove_prefix(2);
					base = 16;
				}
				u64 magnitude = 0;
				const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), magnitude, base);
				if (error != std::errc{} || end != text.data() + text.size() || magnitude > 0x80000000ull)
					return false;
				value = negative ? -static_cast<s64>(magnitude) : static_cast<s64>(magnitude);
				return true;
			}

			static std::pair<std::string, std::string> split_token(const std::string& token)
			{
				const auto equals = token.find('=');
				if (equals == std::string::npos)
					return {upper(token), {}};
				return {upper(token.substr(0, equals)), token.substr(equals + 1)};
			}

			std::string stats() const
			{
				return fmt::format(
					"{\"ok\":true,\"authorized\":%s,\"armed\":%s,\"capturing\":%s,"
					"\"frame\":%llu,\"observed\":%llu,\"accepted\":%llu,\"filtered\":%llu,"
					"\"written\":%llu,\"snapshots\":%llu,\"dropped\":%llu}",
					executable_authorized.load() ? "true" : "false",
					armed.load() ? "true" : "false",
					capturing.load() ? "true" : "false",
					current_frame.load(), observed_events.load(), accepted_events.load(), filtered_events.load(),
					written_events.load(), written_snapshots.load(), dropped_events.load());
			}

			std::string handle_command(std::string command)
			{
				command = trim(std::move(command));
				if (command.empty())
					return "{\"ok\":false,\"error\":\"empty command\"}";

				std::vector<std::string> tokens;
				for (usz begin = 0; begin < command.size();)
				{
					while (begin < command.size() && std::isspace(static_cast<unsigned char>(command[begin]))) ++begin;
					if (begin == command.size()) break;
					usz end = begin;
					bool quoted = false;
					while (end < command.size())
					{
						if (command[end] == '\"') quoted = !quoted;
						if (!quoted && std::isspace(static_cast<unsigned char>(command[end]))) break;
						++end;
					}
					auto token = command.substr(begin, end - begin);
					token.erase(std::remove(token.begin(), token.end(), '\"'), token.end());
					tokens.emplace_back(std::move(token));
					begin = end;
				}
				if (tokens.empty())
					return "{\"ok\":false,\"error\":\"empty command\"}";

				const std::string verb = upper(tokens[0]);
				if (verb == "GET_STATS")
					return stats();
				if (verb == "FLUSH")
				{
					drain();
					if (capture_file) capture_file.sync();
					return stats();
				}
				if (verb == "DISARM")
				{
					armed.store(false, std::memory_order_release);
					update_ppu_gate();
					return stats();
				}
				if (verb == "ARM")
				{
					if (!capture_file)
						return "{\"ok\":false,\"error\":\"START_CAPTURE first\"}";
					armed.store(true, std::memory_order_release);
					update_ppu_gate();
					return stats();
				}
				if (verb == "START_CAPTURE")
				{
					std::string path;
					for (usz i = 1; i < tokens.size(); ++i)
					{
						auto [key, value] = split_token(tokens[i]);
						if (key == "PATH") path = value;
					}
					return open_capture(std::move(path)) ? stats() : "{\"ok\":false,\"error\":\"capture open failed or probe armed\"}";
				}
				if (verb == "STOP_CAPTURE")
				{
					armed.store(false, std::memory_order_release);
					update_ppu_gate();
					close_capture();
					return stats();
				}
				if (verb == "SET_SAMPLE_RATE" || verb == "SET_MAX_EVENTS" || verb == "SET_STACK_WINDOW")
				{
					u64 value = 0;
					if (tokens.size() != 2 || !parse_u64(tokens[1], value))
						return "{\"ok\":false,\"error\":\"numeric argument required\"}";
					if (verb == "SET_SAMPLE_RATE") sample_rate.store(std::max<u32>(1, static_cast<u32>(value)));
					if (verb == "SET_MAX_EVENTS") max_events.store(value);
					if (verb == "SET_STACK_WINDOW") stack_window.store(std::min<u32>(128, static_cast<u32>(value)));
					return stats();
				}
				if (verb == "SET_REGISTER_SET")
				{
					if (tokens.size() < 3)
						return "{\"ok\":false,\"error\":\"usage: SET_REGISTER_SET PPU|SPU list\"}";
					const std::string kind = upper(tokens[1]);
					u64 masks[2]{};
					std::string list = tokens[2];
					for (char& c : list) if (c == ',') c = ' ';
					for (usz begin = 0; begin < list.size();)
					{
						while (begin < list.size() && list[begin] == ' ') ++begin;
						usz end = list.find(' ', begin);
						if (end == std::string::npos) end = list.size();
						u64 reg = 0;
						if (begin < end && parse_u64(std::string_view(list).substr(begin, end - begin), reg) && reg < (kind == "PPU" ? 32 : 128))
							masks[reg / 64] |= 1ull << (reg % 64);
						begin = end;
					}
					if (kind == "PPU") ppu_register_mask.store(masks[0]);
					else if (kind == "SPU") { spu_register_mask[0].store(masks[0]); spu_register_mask[1].store(masks[1]); }
					else return "{\"ok\":false,\"error\":\"register kind must be PPU or SPU\"}";
					return stats();
				}
				if (verb == "SET_FILTER")
				{
					for (usz i = 1; i < tokens.size(); ++i)
					{
						auto [key, text] = split_token(tokens[i]);
						u64 value = 0;
						if (!parse_u64(text, value)) continue;
						if (key == "PPU_PC") filters.ppu_pc.store(static_cast<u32>(value));
						else if (key == "PPU_CALLER") filters.ppu_caller.store(static_cast<u32>(value));
						else if (key == "PPU_THREAD") filters.ppu_thread.store(static_cast<u32>(value));
						else if (key == "SPU_PC") filters.spu_pc.store(static_cast<u32>(value));
						else if (key == "SPU_TARGET") filters.spu_target.store(static_cast<u32>(value));
						else if (key == "SPU_FORMAT") filters.spu_format.store(static_cast<u32>(value));
						else if (key == "SPU_THREAD") filters.spu_thread.store(static_cast<u32>(value));
						else if (key == "SPU_SEQUENCE") filters.spu_task_sequence.store(static_cast<u32>(value));
						else if (key == "SPU_SOURCE") filters.spu_source.store(static_cast<u32>(value));
						else if (key == "SPU_OUTPUT") filters.spu_output.store(static_cast<u32>(value));
						else if (key == "SPU_SOURCE_START") filters.spu_source_start.store(static_cast<u32>(value));
						else if (key == "SPU_SOURCE_END") filters.spu_source_end.store(static_cast<u32>(value));
						else if (key == "SPU_OUTPUT_START") filters.spu_output_start.store(static_cast<u32>(value));
						else if (key == "SPU_OUTPUT_END") filters.spu_output_end.store(static_cast<u32>(value));
						else if (key == "FRAME_START") filters.frame_start.store(value);
						else if (key == "FRAME_END") filters.frame_end.store(value);
						else if (key == "RSX_MASK") filters.rsx_attribute_mask.store(static_cast<u32>(value));
						else if (key == "RSX_VP") filters.rsx_vertex_program.store(static_cast<u32>(value));
						else if (key == "RSX_FP") filters.rsx_fragment_program.store(static_cast<u32>(value));
						else if (key == "RSX_MIN_VERTICES") filters.rsx_min_vertices.store(static_cast<u32>(value));
						else if (key == "RSX_MAX_VERTICES") filters.rsx_max_vertices.store(static_cast<u32>(value));
						else if (key == "RSX_MIN_INDICES") filters.rsx_min_indices.store(static_cast<u32>(value));
						else if (key == "RSX_MAX_INDICES") filters.rsx_max_indices.store(static_cast<u32>(value));
						else if (key == "RSX_ADDRESS") filters.rsx_address.store(static_cast<u32>(value));
						else if (key == "RSX_PRIMITIVE") filters.rsx_primitive.store(static_cast<u32>(value));
						else if (key == "FIRST_HITS") max_events.store(value);
					}
					update_ppu_gate();
					return stats();
				}
				if (verb == "REMOVE_POINTER_FOLLOW")
				{
					if (armed.load(std::memory_order_acquire))
						return "{\"ok\":false,\"error\":\"DISARM before changing pointer rules\"}";
					u64 id = 0;
					if (tokens.size() != 2 || !parse_u64(tokens[1], id))
						return "{\"ok\":false,\"error\":\"rule id required\"}";
					for (auto& rule : pointer_rules) if (rule.id == id) rule.enabled.store(0, std::memory_order_release);
					return stats();
				}
				if (verb == "ADD_POINTER_FOLLOW")
				{
					if (armed.load(std::memory_order_acquire))
						return "{\"ok\":false,\"error\":\"DISARM before changing pointer rules\"}";
					pointer_rule* destination = nullptr;
					for (auto& rule : pointer_rules) if (!rule.enabled.load(std::memory_order_acquire)) { destination = &rule; break; }
					if (!destination) return "{\"ok\":false,\"error\":\"all pointer slots are in use\"}";
					pointer_rule candidate{};
					for (usz i = 1; i < tokens.size(); ++i)
					{
						auto [key, text] = split_token(tokens[i]);
						u64 value = 0;
						if (key == "SOURCE")
						{
							const auto source = upper(text);
							if (source.starts_with("PPU_R")) { candidate.source = pointer_source::ppu_gpr; parse_u64(source.substr(5), value); candidate.source_index = static_cast<u32>(value); }
							else if (source.starts_with("SPU_R")) { candidate.source = pointer_source::spu_gpr_lane3; parse_u64(source.substr(5), value); candidate.source_index = static_cast<u32>(value); }
							else if (source == "TASK_HEADER") candidate.source = pointer_source::task_header;
							else if (source == "TASK_CONTEXT") candidate.source = pointer_source::task_context;
							else if (source == "DMA_DESCRIPTOR") candidate.source = pointer_source::dma_descriptor;
							else if (source == "SOURCE0") candidate.source = pointer_source::source_0;
							else if (source == "SOURCE1") candidate.source = pointer_source::source_1;
							else if (source == "AUXILIARY") candidate.source = pointer_source::auxiliary;
							else if (source == "OUTPUT") candidate.source = pointer_source::output;
						}
						else if (key == "SPACE") candidate.space = upper(text) == "LS" ? pointer_space::spu_ls : pointer_space::guest;
						else if (key.starts_with("OFFSET"))
						{
							u64 index = 0;
							s64 offset = 0;
							if (parse_u64(key.substr(6), index) && index < candidate.offsets.size() && parse_s64(text, offset))
								candidate.offsets[index] = static_cast<s32>(offset);
						}
						else if (parse_u64(text, value))
						{
							if (key == "ID") candidate.id = static_cast<u32>(value);
							else if (key == "DEPTH") candidate.depth = std::min<u32>(4, static_cast<u32>(value));
							else if (key == "SIZE") candidate.capture_size = std::min<u32>(256, static_cast<u32>(value));
						}
					}
					if (!candidate.id) return "{\"ok\":false,\"error\":\"non-zero rule ID required\"}";
					destination->id = candidate.id;
					destination->source = candidate.source;
					destination->space = candidate.space;
					destination->source_index = candidate.source_index;
					destination->depth = candidate.depth;
					destination->offsets = candidate.offsets;
					destination->capture_size = candidate.capture_size;
					destination->enabled.store(1, std::memory_order_release);
					return stats();
				}
				if (verb == "ONE_SHOT_SNAPSHOT")
				{
					u32 kind = static_cast<u32>(event_type::spu_ls_snapshot);
					u32 count = 1;
					for (usz i = 1; i < tokens.size(); ++i)
					{
						auto [key, text] = split_token(tokens[i]);
						if (key == "TYPE" && upper(text) == "PPU_STACK") kind = static_cast<u32>(event_type::ppu_stack_snapshot);
						u64 value = 0;
						if (key == "COUNT" && parse_u64(text, value)) count = std::min<u32>(16, static_cast<u32>(value));
					}
					snapshot_kind.store(kind, std::memory_order_release);
					snapshot_remaining.store(count, std::memory_order_release);
					return stats();
				}

				return "{\"ok\":false,\"error\":\"unknown command\"}";
			}

			void worker_main(std::stop_token token)
			{
#ifdef _WIN32
				constexpr wchar_t pipe_name[] = L"\\\\.\\pipe\\RPCS3AscensionLiveProbe";
				HANDLE pipe = CreateNamedPipeW(
					pipe_name,
					PIPE_ACCESS_DUPLEX,
					PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_NOWAIT,
					1, 65536, 65536, 0, nullptr);
				bool connected = false;
				if (pipe == INVALID_HANDLE_VALUE)
					ascension_live_probe_log.error("Could not create control pipe (Win32 error %u).", GetLastError());
#endif
				while (!token.stop_requested() && !stopping.load(std::memory_order_acquire))
				{
					drain();
#ifdef _WIN32
					if (pipe != INVALID_HANDLE_VALUE)
					{
						if (!connected)
						{
							if (ConnectNamedPipe(pipe, nullptr)) connected = true;
							else
							{
								const DWORD error = GetLastError();
								connected = error == ERROR_PIPE_CONNECTED;
							}
						}
						if (connected)
						{
							std::array<char, 4096> buffer{};
							DWORD received = 0;
							if (ReadFile(pipe, buffer.data(), static_cast<DWORD>(buffer.size() - 1), &received, nullptr) && received)
							{
								const std::string response = handle_command(std::string(buffer.data(), received));
								DWORD sent = 0;
								WriteFile(pipe, response.data(), static_cast<DWORD>(response.size()), &sent, nullptr);
							}
							else if (const DWORD error = GetLastError(); error == ERROR_BROKEN_PIPE)
							{
								DisconnectNamedPipe(pipe);
								connected = false;
							}
						}
					}
#endif
					std::this_thread::sleep_for(std::chrono::milliseconds(2));
				}
#ifdef _WIN32
				if (pipe != INVALID_HANDLE_VALUE)
				{
					if (connected) DisconnectNamedPipe(pipe);
					CloseHandle(pipe);
				}
#endif
				drain();
			}
		};

		probe_state& state()
		{
			static const std::unique_ptr<probe_state> value = std::make_unique<probe_state>();
			return *value;
		}

		bool frame_allowed(const probe_state& probe, u64 frame)
		{
			return frame >= probe.filters.frame_start.load(std::memory_order_relaxed) &&
				frame <= probe.filters.frame_end.load(std::memory_order_relaxed);
		}

		bool event_allowed(probe_state& probe)
		{
			if (!probe.capturing.load(std::memory_order_acquire) || !probe.armed.load(std::memory_order_acquire) ||
				!probe.executable_authorized.load(std::memory_order_acquire))
				return false;

			const u64 observed = probe.observed_events.fetch_add(1, std::memory_order_relaxed);
			if (observed % probe.sample_rate.load(std::memory_order_relaxed))
				return false;
			const u64 maximum = probe.max_events.load(std::memory_order_relaxed);
			if (maximum && probe.accepted_events.load(std::memory_order_relaxed) >= maximum)
				return false;
			probe.accepted_events.fetch_add(1, std::memory_order_relaxed);
			return true;
		}

		u64 hash_bytes(const void* data, usz size)
		{
			const auto* bytes = static_cast<const u8*>(data);
			u64 hash = 1469598103934665603ull;
			for (usz i = 0; i < size; ++i)
			{
				hash ^= bytes[i];
				hash *= 1099511628211ull;
			}
			return hash;
		}

		bool range_valid(pointer_space space, const spu_thread* spu, u64 address, u32 size)
		{
			if (address > umax || !size)
				return false;
			if (space == pointer_space::spu_ls)
				return spu && address <= 0x40000 && size <= 0x40000 - address;
			return vm::check_addr(address, vm::page_readable, size);
		}

		const u8* range_pointer(pointer_space space, const spu_thread* spu, u32 address)
		{
			return space == pointer_space::spu_ls ? spu->ls + address : static_cast<const u8*>(vm::base(address));
		}

		u32 read_be_u32(const u8* bytes)
		{
			return (static_cast<u32>(bytes[0]) << 24) | (static_cast<u32>(bytes[1]) << 16) |
				(static_cast<u32>(bytes[2]) << 8) | bytes[3];
		}

		struct pointer_context
		{
			const ppu_thread* ppu = nullptr;
			const spu_thread* spu = nullptr;
			const u32* spu_lane3 = nullptr;
			u32 task_header = 0;
			u32 task_context = 0;
			u32 dma_descriptor = 0;
			u32 source_0 = 0;
			u32 source_1 = 0;
			u32 auxiliary = 0;
			u32 output = 0;
		};

		u64 pointer_source_value(const pointer_rule& rule, const pointer_context& context)
		{
			switch (rule.source)
			{
			case pointer_source::ppu_gpr: return context.ppu && rule.source_index < 32 ? context.ppu->gpr[rule.source_index] : 0;
			case pointer_source::spu_gpr_lane3: return context.spu_lane3 && rule.source_index < 128 ? context.spu_lane3[rule.source_index] : 0;
			case pointer_source::task_header: return context.task_header;
			case pointer_source::task_context: return context.task_context;
			case pointer_source::dma_descriptor: return context.dma_descriptor;
			case pointer_source::source_0: return context.source_0;
			case pointer_source::source_1: return context.source_1;
			case pointer_source::auxiliary: return context.auxiliary;
			case pointer_source::output: return context.output;
			}
			return 0;
		}

		u32 collect_pointers(event_record_v1& event, const pointer_context& context)
		{
			probe_state& probe = state();
			u32 count = 0;
			for (const auto& rule : probe.pointer_rules)
			{
				if (count == event.pointer_results.size() || !rule.enabled.load(std::memory_order_acquire))
					continue;
				auto& result = event.pointer_results[count++];
				result.rule_id = rule.id;
				result.source_value = pointer_source_value(rule, context);
				u64 address = result.source_value;
				for (u32 depth = 0; depth < rule.depth; ++depth)
				{
					address = static_cast<u64>(static_cast<s64>(address) + rule.offsets[depth]);
					if (!range_valid(rule.space, context.spu, address, 4))
					{
						address = 0;
						break;
					}
					address = read_be_u32(range_pointer(rule.space, context.spu, static_cast<u32>(address)));
				}
				if (address)
					address = static_cast<u64>(static_cast<s64>(address) + rule.offsets[rule.depth]);
				result.final_address = address;
				result.requested_size = rule.capture_size;
				if (!range_valid(rule.space, context.spu, address, rule.capture_size))
					continue;
				const u8* data = range_pointer(rule.space, context.spu, static_cast<u32>(address));
				result.captured_size = rule.capture_size;
				result.content_hash = hash_bytes(data, rule.capture_size);
				std::memcpy(result.sample.data(), data, std::min<usz>(result.sample.size(), rule.capture_size));
				result.flags = 1;
			}
			if (count)
				event.header.flags |= event_flag_pointer_data;
			return count;
		}

		void maybe_snapshot(event_type type, const void* data, u32 size, u32 pc, u32 thread_id, u64 frame)
		{
			probe_state& probe = state();
			if (probe.snapshot_kind.load(std::memory_order_acquire) != static_cast<u32>(type))
				return;
			u32 remaining = probe.snapshot_remaining.load(std::memory_order_relaxed);
			while (remaining && !probe.snapshot_remaining.compare_exchange_weak(remaining, remaining - 1, std::memory_order_acq_rel)) {}
			if (!remaining)
				return;

			for (auto& slot : probe.snapshots)
			{
				u32 expected = 0;
				if (!slot.state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel))
					continue;
				slot.header = {};
				slot.header.type = static_cast<u16>(type);
				slot.header.flags = event_flag_authorized | event_flag_snapshot;
				slot.header.sequence = probe.next_event_sequence.fetch_add(1, std::memory_order_relaxed);
				slot.header.host_time_us = get_system_time();
				slot.header.frame_id = frame;
				slot.header.thread_id = thread_id;
				slot.header.pc = pc;
				slot.payload_size = std::min<u32>(size, snapshot_capacity);
				slot.header.total_size = sizeof(record_header_v1) + slot.payload_size;
				std::memcpy(slot.payload.data(), data, slot.payload_size);
				slot.state.store(2, std::memory_order_release);
				return;
			}
			probe.dropped_events.fetch_add(1, std::memory_order_relaxed);
		}

		void publish_output(const output_identity& identity)
		{
			if (identity.output_ea_end <= identity.output_ea_base)
				return;
			probe_state& probe = state();
			const u64 id = probe.next_output_id.fetch_add(1, std::memory_order_relaxed);
			output_slot& slot = probe.outputs[id & output_ring_mask];
			slot.stamp.store(id * 2 + 1, std::memory_order_release);
			const std::array<u64, 13> values{
				id,
				identity.event_sequence,
				identity.host_time_us,
				identity.frame_id,
				identity.task_sequence,
				identity.task_format,
				identity.packed_count,
				identity.descriptor_ea,
				identity.source_ea_0,
				identity.source_ea_1,
				identity.auxiliary_ea,
				identity.output_ea_base,
				identity.output_ea_end};
			for (u32 index = 0; index < values.size(); ++index)
				slot.values[index].store(values[index], std::memory_order_relaxed);
			slot.stamp.store(id * 2 + 2, std::memory_order_release);

			const u32 first = identity.output_ea_base >> 12;
			const u32 last = (identity.output_ea_end - 1) >> 12;
			for (u32 page = first; page <= last && page < guest_page_count; ++page)
				probe.output_pages[page].store(id, std::memory_order_release);
		}
	}

	bool bootstrap_enabled()
	{
		static const bool enabled = []
		{
			const char* env = std::getenv("RPCS3_ASCENSION_LIVE_PROBE");
			return env && env[0] && std::strcmp(env, "0") != 0;
		}();
		if (enabled)
			(void)state();
		return enabled;
	}

	void authorize_executable(std::string_view executable_hash)
	{
		if (!bootstrap_enabled())
			return;
		probe_state& probe = state();
		const bool match = Emu.GetTitleID() == supported_title_id &&
			Emu.GetAppVersion() == supported_app_version &&
			executable_hash == supported_executable_hash;
		if (match && !probe.executable_authorized.exchange(true, std::memory_order_acq_rel))
		{
			ascension_live_probe_log.success(
				"Authorized exact target %s v%s (%s); runtime probe remains disarmed until commanded.",
				supported_title_id, supported_app_version, supported_executable_hash);
		}
		probe.update_ppu_gate();
	}

	bool authorized()
	{
		return bootstrap_enabled() && state().executable_authorized.load(std::memory_order_acquire);
	}

	std::atomic<u64>* ppu_gate_address()
	{
		return bootstrap_enabled() ? &state().ppu_gate : nullptr;
	}

	void observe_ppu_call(ppu_thread* ppu, u32 pc, u32 target, u64 caller_lr)
	{
		if (!ppu || !bootstrap_enabled())
			return;
		probe_state& probe = state();
		const u64 gate = probe.ppu_gate.load(std::memory_order_acquire);
		if (!(gate & armed_bit) || static_cast<u32>(gate) != pc)
			return;
		const u32 caller_filter = probe.filters.ppu_caller.load(std::memory_order_relaxed);
		const u32 thread_filter = probe.filters.ppu_thread.load(std::memory_order_relaxed);
		const u64 frame = probe.current_frame.load(std::memory_order_relaxed);
		if ((caller_filter && static_cast<u32>(caller_lr) != caller_filter) ||
			(thread_filter && ppu->id != thread_filter) || !frame_allowed(probe, frame))
		{
			probe.filtered_events.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (!event_allowed(probe))
			return;

		event_record_v1 event{};
		event.header.type = static_cast<u16>(event_type::ppu_call);
		event.header.flags = event_flag_authorized;
		event.header.sequence = probe.next_event_sequence.fetch_add(1, std::memory_order_relaxed);
		event.header.host_time_us = get_system_time();
		event.header.frame_id = frame;
		event.header.thread_id = ppu->id;
		event.header.pc = pc;
		event.header.caller = static_cast<u32>(caller_lr);
		event.header.target = target;
		const u64 mask = probe.ppu_register_mask.load(std::memory_order_relaxed);
		for (u32 reg = 0; reg < 32; ++reg)
			if (mask & (1ull << reg)) event.values[reg] = ppu->gpr[reg];
		event.words[0] = static_cast<u32>(mask);
		event.words[1] = static_cast<u32>(mask >> 32);
		const u32 stack_address = static_cast<u32>(ppu->gpr[1]);
		const u32 stack_size = probe.stack_window.load(std::memory_order_relaxed);
		event.words[2] = stack_address;
		event.words[3] = stack_size;
		if (stack_size && vm::check_addr(stack_address, vm::page_readable, stack_size))
		{
			std::memcpy(&event.words[32], vm::base(stack_address), stack_size);
			event.words[4] = stack_size;
			event.header.flags |= event_flag_stack_valid;
		}
		pointer_context context{};
		context.ppu = ppu;
		event.words[5] = collect_pointers(event, context);
		probe.push(event);
		if (event.header.flags & event_flag_stack_valid)
			maybe_snapshot(event_type::ppu_stack_snapshot, vm::base(stack_address), stack_size, pc, ppu->id, frame);
	}

	void observe_spu_task(spu_thread* spu, u32 pc, u32 target, u32 task_header_lsa, u32 task_context_lsa, u32 dma_descriptor_lsa, const u32* lane3_registers)
	{
		if (!spu || !bootstrap_enabled())
			return;
		probe_state& probe = state();
		if (!probe.capturing.load(std::memory_order_acquire) || !probe.armed.load(std::memory_order_acquire) ||
			!probe.executable_authorized.load(std::memory_order_acquire))
			return;
		// Authenticate the exact five-instruction signature previously verified
		// for the Ascension skinning overlay. PC/target alone are not a sufficient
		// SPU-module identity because unrelated overlays reuse LS addresses.
		if (spu->_ref<u32>(expected_spu_pc - 12) != 0x04002d03 ||
			spu->_ref<u32>(expected_spu_pc - 8) != 0x3fe02f84 ||
			spu->_ref<u32>(expected_spu_pc - 4) != 0x1c080085 ||
			spu->_ref<u32>(expected_spu_pc) != expected_spu_call_opcode ||
			spu->_ref<u32>(expected_spu_pc + 4) != 0x4020007f)
		{
			probe.filtered_events.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		const u64 frame = probe.current_frame.load(std::memory_order_relaxed);
		if (!frame_allowed(probe, frame) ||
			(probe.filters.spu_pc.load(std::memory_order_relaxed) && probe.filters.spu_pc.load(std::memory_order_relaxed) != pc) ||
			(probe.filters.spu_target.load(std::memory_order_relaxed) && probe.filters.spu_target.load(std::memory_order_relaxed) != target) ||
			(probe.filters.spu_thread.load(std::memory_order_relaxed) && probe.filters.spu_thread.load(std::memory_order_relaxed) != spu->id))
		{
			probe.filtered_events.fetch_add(1, std::memory_order_relaxed);
			return;
		}

		auto read_ls = [&](u32 address, u32& value)
		{
			if (address > 0x40000 || sizeof(u32) > 0x40000 - address) return false;
			value = spu->_ref<u32>(address);
			return true;
		};
		u32 task_sequence = 0, task_format = 0, packed_count = 0, descriptor_ea = 0;
		u32 auxiliary = 0, source_0 = 0, source_1 = 0, output_base = 0, output_end = 0;
		const bool valid =
			read_ls(task_header_lsa + 0x04, descriptor_ea) &&
			read_ls(task_header_lsa + 0x10, task_sequence) &&
			read_ls(task_header_lsa + 0x18, auxiliary) &&
			read_ls(task_header_lsa + 0x30, source_0) &&
			read_ls(task_header_lsa + 0x34, source_1) &&
			read_ls(task_context_lsa + 0x00, task_format) &&
			read_ls(task_context_lsa + 0x08, packed_count) &&
			read_ls(dma_descriptor_lsa + 0x00, output_base) &&
			read_ls(dma_descriptor_lsa + 0x0c, output_end);
		if (!valid)
			return;

		const bool passes =
			(!probe.filters.spu_format.load(std::memory_order_relaxed) || probe.filters.spu_format.load(std::memory_order_relaxed) == task_format) &&
			(!probe.filters.spu_task_sequence.load(std::memory_order_relaxed) || probe.filters.spu_task_sequence.load(std::memory_order_relaxed) == task_sequence) &&
			(!probe.filters.spu_source.load(std::memory_order_relaxed) || probe.filters.spu_source.load(std::memory_order_relaxed) == source_0) &&
			source_0 >= probe.filters.spu_source_start.load(std::memory_order_relaxed) &&
			source_0 <= probe.filters.spu_source_end.load(std::memory_order_relaxed) &&
			output_end >= probe.filters.spu_output_start.load(std::memory_order_relaxed) &&
			output_base <= probe.filters.spu_output_end.load(std::memory_order_relaxed) &&
			(!probe.filters.spu_output.load(std::memory_order_relaxed) ||
				(probe.filters.spu_output.load(std::memory_order_relaxed) >= output_base && probe.filters.spu_output.load(std::memory_order_relaxed) < output_end));
		if (!passes)
		{
			probe.filtered_events.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (!event_allowed(probe))
			return;

		event_record_v1 event{};
		event.header.type = static_cast<u16>(event_type::spu_task);
		event.header.flags = event_flag_authorized;
		event.header.sequence = probe.next_event_sequence.fetch_add(1, std::memory_order_relaxed);
		event.header.host_time_us = get_system_time();
		event.header.frame_id = frame;
		event.header.thread_id = spu->id;
		event.header.pc = pc;
		event.header.target = target;
		event.words[32] = 0;
		for (u32 reg = 0; reg < 128 && event.words[32] < 32; ++reg)
		{
			if (!(probe.spu_register_mask[reg / 64].load(std::memory_order_relaxed) & (1ull << (reg % 64)))) continue;
			const u32 slot = event.words[32]++;
			event.words[slot] = reg;
			event.values[slot] = lane3_registers ? lane3_registers[reg] : 0;
		}
		event.words[33] = task_header_lsa;
		event.words[34] = task_context_lsa;
		event.words[35] = dma_descriptor_lsa;
		event.words[36] = task_sequence;
		event.words[37] = task_format;
		event.words[38] = packed_count;
		event.words[39] = descriptor_ea;
		event.words[40] = auxiliary;
		event.words[41] = source_0;
		event.words[42] = source_1;
		event.words[43] = output_base;
		event.words[44] = output_end;
		pointer_context context{};
		context.spu = spu;
		context.spu_lane3 = lane3_registers;
		context.task_header = task_header_lsa;
		context.task_context = task_context_lsa;
		context.dma_descriptor = dma_descriptor_lsa;
		context.source_0 = source_0;
		context.source_1 = source_1;
		context.auxiliary = auxiliary;
		context.output = output_base;
		event.words[45] = collect_pointers(event, context);

		output_identity identity{};
		identity.event_sequence = event.header.sequence;
		identity.host_time_us = event.header.host_time_us;
		identity.frame_id = frame;
		identity.task_sequence = task_sequence;
		identity.task_format = task_format;
		identity.packed_count = packed_count;
		identity.descriptor_ea = descriptor_ea;
		identity.source_ea_0 = source_0;
		identity.source_ea_1 = source_1;
		identity.auxiliary_ea = auxiliary;
		identity.output_ea_base = output_base;
		identity.output_ea_end = output_end;
		publish_output(identity);
		probe.push(event);
		maybe_snapshot(event_type::spu_ls_snapshot, spu->ls, 0x40000, pc, spu->id, frame);
	}

	bool identify_output_range(u32 address, u32 size, output_identity& result)
	{
		result = {};
		if (!bootstrap_enabled() || !authorized() || !size)
			return false;
		probe_state& probe = state();
		const u64 query_end = static_cast<u64>(address) + size;
		if (query_end > 0x1'0000'0000ull)
			return false;
		const u32 first = address >> 12;
		const u32 last = static_cast<u32>((query_end - 1) >> 12);
		u64 best_overlap = 0;
		const u64 now = get_system_time();
		for (u32 page = first; page <= last && page < guest_page_count; ++page)
		{
			const u64 id = probe.output_pages[page].load(std::memory_order_acquire);
			if (!id) continue;
			const output_slot& slot = probe.outputs[id & output_ring_mask];
			const u64 expected = id * 2 + 2;
			if (slot.stamp.load(std::memory_order_acquire) != expected) continue;
			output_identity candidate{};
			const u64 publish_id = slot.values[0].load(std::memory_order_relaxed);
			candidate.event_sequence = slot.values[1].load(std::memory_order_relaxed);
			candidate.host_time_us = slot.values[2].load(std::memory_order_relaxed);
			candidate.frame_id = slot.values[3].load(std::memory_order_relaxed);
			candidate.task_sequence = static_cast<u32>(slot.values[4].load(std::memory_order_relaxed));
			candidate.task_format = static_cast<u32>(slot.values[5].load(std::memory_order_relaxed));
			candidate.packed_count = static_cast<u32>(slot.values[6].load(std::memory_order_relaxed));
			candidate.descriptor_ea = static_cast<u32>(slot.values[7].load(std::memory_order_relaxed));
			candidate.source_ea_0 = static_cast<u32>(slot.values[8].load(std::memory_order_relaxed));
			candidate.source_ea_1 = static_cast<u32>(slot.values[9].load(std::memory_order_relaxed));
			candidate.auxiliary_ea = static_cast<u32>(slot.values[10].load(std::memory_order_relaxed));
			candidate.output_ea_base = static_cast<u32>(slot.values[11].load(std::memory_order_relaxed));
			candidate.output_ea_end = static_cast<u32>(slot.values[12].load(std::memory_order_relaxed));
			if (publish_id != id || slot.stamp.load(std::memory_order_acquire) != expected ||
				now < candidate.host_time_us || now - candidate.host_time_us > output_max_age_us) continue;
			const u64 overlap_start = std::max<u64>(address, candidate.output_ea_base);
			const u64 overlap_end = std::min<u64>(query_end, candidate.output_ea_end);
			if (overlap_end <= overlap_start || overlap_end - overlap_start <= best_overlap) continue;
			best_overlap = overlap_end - overlap_start;
			result = candidate;
			result.output_relative_offset = static_cast<u32>(overlap_start - candidate.output_ea_base);
			result.overlap_bytes = static_cast<u32>(best_overlap);
		}
		return best_overlap != 0;
	}

	bool rsx_candidate_enabled(u16 attribute_mask, u32 vertex_count, u8 primitive)
	{
		if (!bootstrap_enabled())
			return false;
		probe_state& probe = state();
		if (!probe.capturing.load(std::memory_order_acquire) || !probe.armed.load(std::memory_order_acquire) ||
			!probe.executable_authorized.load(std::memory_order_acquire)) return false;
		const u32 mask = probe.filters.rsx_attribute_mask.load(std::memory_order_relaxed);
		const u32 primitive_filter = probe.filters.rsx_primitive.load(std::memory_order_relaxed);
		return (!mask || mask == attribute_mask) &&
			vertex_count >= probe.filters.rsx_min_vertices.load(std::memory_order_relaxed) &&
			vertex_count <= probe.filters.rsx_max_vertices.load(std::memory_order_relaxed) &&
			(primitive_filter == umax || primitive_filter == primitive);
	}

	void observe_rsx_draw(const rsx_draw_event& draw)
	{
		if (!bootstrap_enabled()) return;
		probe_state& probe = state();
		if (!frame_allowed(probe, draw.frame_id) ||
			(probe.filters.rsx_vertex_program.load(std::memory_order_relaxed) && probe.filters.rsx_vertex_program.load(std::memory_order_relaxed) != draw.vertex_program_id) ||
			(probe.filters.rsx_fragment_program.load(std::memory_order_relaxed) && probe.filters.rsx_fragment_program.load(std::memory_order_relaxed) != draw.fragment_program_id) ||
			draw.index_count < probe.filters.rsx_min_indices.load(std::memory_order_relaxed) ||
			draw.index_count > probe.filters.rsx_max_indices.load(std::memory_order_relaxed) ||
			(probe.filters.rsx_address.load(std::memory_order_relaxed) &&
				(probe.filters.rsx_address.load(std::memory_order_relaxed) < draw.stream_address ||
				probe.filters.rsx_address.load(std::memory_order_relaxed) >= static_cast<u64>(draw.stream_address) + draw.stream_size)))
		{
			probe.filtered_events.fetch_add(1, std::memory_order_relaxed);
			return;
		}
		if (!event_allowed(probe)) return;
		event_record_v1 event{};
		event.header.type = static_cast<u16>(event_type::rsx_draw);
		event.header.flags = event_flag_authorized | (draw.task_valid ? event_flag_output_mapping : 0);
		event.header.sequence = probe.next_event_sequence.fetch_add(1, std::memory_order_relaxed);
		event.header.host_time_us = get_system_time();
		event.header.frame_id = draw.frame_id;
		event.header.producer_sequence = draw.task_valid ? draw.task.event_sequence : 0;
		event.words[0] = draw.draw_sequence;
		event.words[1] = draw.vertex_program_id;
		event.words[2] = draw.fragment_program_id;
		event.words[3] = draw.vertex_draw_count;
		event.words[4] = draw.stream_vertex_count;
		event.words[5] = draw.first_vertex;
		event.words[6] = draw.stream_address;
		event.words[7] = draw.stream_size;
		event.words[8] = draw.index_address;
		event.words[9] = draw.index_count;
		event.words[10] = draw.attribute_mask;
		event.words[11] = draw.stride;
		event.words[12] = draw.primitive;
		event.words[13] = draw.command;
		event.words[14] = draw.index_type;
		event.words[15] = draw.indexed_constants;
		event.words[16] = draw.restart_index_enabled;
		event.words[17] = draw.restart_index;
		event.values[0] = draw.index_hash;
		event.values[1] = draw.layout_hash;
		if (draw.task_valid)
		{
			event.values[2] = draw.task.host_time_us;
			event.words[18] = draw.task.task_sequence;
			event.words[19] = draw.task.task_format;
			event.words[20] = draw.task.packed_count;
			event.words[21] = draw.task.descriptor_ea;
			event.words[22] = draw.task.source_ea_0;
			event.words[23] = draw.task.source_ea_1;
			event.words[24] = draw.task.auxiliary_ea;
			event.words[25] = draw.task.output_ea_base;
			event.words[26] = draw.task.output_ea_end;
			event.words[27] = draw.task.output_relative_offset;
			event.words[28] = draw.task.overlap_bytes;
		}
		probe.push(event);
	}

	u32 report_frame(u64 frame_id)
	{
		if (!bootstrap_enabled())
			return 0;
		probe_state& probe = state();
		if (probe.current_frame.exchange(frame_id, std::memory_order_relaxed) != frame_id)
		{
			probe.current_rsx_draw_sequence.store(0, std::memory_order_relaxed);
			return 0;
		}
		return probe.current_rsx_draw_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	}

	void shutdown()
	{
		if (!bootstrap_enabled()) return;
		probe_state& probe = state();
		probe.armed.store(false, std::memory_order_release);
		probe.update_ppu_gate();
		probe.close_capture();
	}
}
