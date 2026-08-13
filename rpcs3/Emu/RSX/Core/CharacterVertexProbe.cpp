#include "stdafx.h"
#include "CharacterVertexProbe.h"

#include "../gcm_enums.h"
#include "../NV47/FW/draw_call.inc.h"
#include "Utilities/File.h"
#include "util/logs.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

LOG_CHANNEL(character_vertex_log, "Character Vertex Probe");

namespace rsx::character_vertex_probe
{
	namespace
	{
		constexpr u32 character_attribute_mask = 0xc3b5;
		constexpr u32 spu_writer_probe_version = 2;
		constexpr u32 temporal_rsx_probe_version = 3;
		constexpr u32 motion_qualification_probe_version = 5;
		constexpr u32 motion_raster_probe_version = 6;
		constexpr u32 task_identity_motion_version = 7;
		constexpr usz watch_slot_count = 256;
		constexpr u32 bloom_bit_count = 1u << 16;
		constexpr u32 bloom_word_count = bloom_bit_count / 64;
		constexpr u32 guest_page_shift = 12;
		constexpr u32 max_watched_range_size = 8u * 1024u * 1024u;
		constexpr usz local_store_size = 0x40000;
		constexpr usz register_count = 128;
		constexpr usz register_size = 16;
		constexpr usz register_bytes = register_count * register_size;
		constexpr usz temporal_record_limit = 32768;
		constexpr u64 temporal_capture_frame_span = 360;
		constexpr u32 temporal_vertex_sample_count = 64;
		constexpr usz motion_qualification_record_limit = 16384;
		constexpr u64 motion_qualification_frame_span = 360;
		constexpr u32 motion_minimum_projected_samples = 4;
		constexpr usz motion_raster_vertex_limit = 1'500'000;
		constexpr f32 motion_raster_world_rms_limit = 8.f;
		constexpr u64 fnv_offset_basis = 14695981039346656037ull;
		constexpr u64 fnv_prime = 1099511628211ull;

		enum capture_phase : u32
		{
			capture_idle = 0,
			capture_copying = 1,
			capture_ready = 2,
			capture_written = 3,
		};

		struct watch_slot
		{
			// Guest start in the low 32 bits and exclusive end in the high 32.
			std::atomic<u64> range{0};
			std::atomic<u64> frame_and_program{0};
			std::atomic<u64> draw_info{0};
		};

		struct capture_metadata
		{
			u32 spu_id = 0;
			u32 lv2_id = 0;
			u32 pc = 0;
			u64 block_hash = 0;
			u32 address = 0;
			u32 lsa = 0;
			u32 size = 0;
			u8 command = 0;
			u8 tag = 0;
			u32 watch_start = 0;
			u32 watch_end = 0;
			u32 watch_frame = 0;
			u32 vertex_program_id = 0;
			u32 vertex_count = 0;
			u16 attribute_mask = 0;
			u8 stride = 0;
		};

		struct probe_state
		{
			std::array<watch_slot, watch_slot_count> watches{};
			std::array<std::atomic<u64>, bloom_word_count> page_bloom{};
			std::array<u8, local_store_size> local_store{};
			std::array<u8, register_bytes> registers{};
			capture_metadata capture{};

			std::atomic<u32> capture_phase{capture_idle};
			std::atomic<u32> watch_cursor{0};
			std::atomic<u64> watched_range_count{0};
			std::atomic<u64> put_count{0};
			std::atomic<u64> bloom_match_count{0};
			std::atomic<u64> overlap_count{0};
		};

		enum temporal_capture_phase : u8
		{
			temporal_waiting = 0,
			temporal_capturing = 1,
			temporal_ready = 2,
			temporal_written = 3,
		};

		struct temporal_record
		{
			u64 sequence = 0;
			u32 draw_in_frame = 0;
			rsx_draw_desc draw{};
			u64 index_hash = 0;
			u64 constants_hash = 0;
			u64 stream_hash = 0;
			u64 non_position_hash = 0;
			u64 identity_hash = 0;
			std::array<u64, 16> attribute_hashes{};
			u32 finite_position_samples = 0;
			std::array<f32, 3> position_min{};
			std::array<f32, 3> position_max{};
			std::array<f32, 12> position_samples{};
			u8 position_sample_valid_mask = 0;
			std::array<f32, temporal_vertex_sample_count * 3> sampled_positions{};
			u64 sampled_position_valid_mask = 0;
			u8 sampled_position_count = 0;
			std::vector<std::array<f32, 3>> full_positions;
			std::vector<u32> triangle_indices;
		};

		struct temporal_probe_state
		{
			std::vector<temporal_record> records;
			temporal_capture_phase phase = temporal_waiting;
			u64 first_frame = 0;
			u64 last_frame = 0;
			u64 current_frame = umax;
			u64 last_status_frame = 0;
			u32 draw_in_frame = 0;
			bool announced = false;
		};

		struct motion_qualification_record
		{
			u64 sequence = 0;
			u64 frame_id = 0;
			u64 identity_hash = 0;
			u32 draw_in_frame = 0;
			u32 vertex_program_id = 0;
			u32 fragment_program_id = 0;
			u32 vertex_draw_count = 0;
			u32 stream_vertex_count = 0;
			u32 first_vertex = 0;
			u32 current_address = 0;
			u32 previous_address = 0;
			u32 projected_samples = 0;
			f32 mean_total_x = 0.f;
			f32 mean_total_y = 0.f;
			f32 rms_total_pixels = 0.f;
			f32 max_total_pixels = 0.f;
			f32 mean_camera_x = 0.f;
			f32 mean_camera_y = 0.f;
			f32 rms_camera_pixels = 0.f;
			f32 mean_object_x = 0.f;
			f32 mean_object_y = 0.f;
			f32 rms_object_pixels = 0.f;
			f32 max_object_pixels = 0.f;
			f32 rms_world_delta = 0.f;
			f32 max_world_delta = 0.f;
			f32 camera_confidence = 0.f;
		};

		enum motion_qualification_phase : u8
		{
			motion_waiting_for_arm = 0,
			motion_capturing = 1,
			motion_written = 2,
		};

		struct motion_qualification_state
		{
			std::atomic<bool> collect_draws{false};
			motion_qualification_phase phase = motion_waiting_for_arm;
			std::vector<temporal_record> current_records;
			std::vector<temporal_record> previous_records;
			std::vector<motion_qualification_record> results;
			u64 current_frame = umax;
			u64 previous_frame = umax;
			u64 first_frame = 0;
			u64 last_frame = 0;
			u64 observed_draw_sequence = 0;
			u64 matched_mesh_pairs = 0;
			u64 rejected_projection_pairs = 0;
			u64 ambiguous_identity_groups = 0;
			u64 reassociated_mesh_pairs = 0;
			u64 unmatched_current_meshes = 0;
			u64 raster_frames = 0;
			u64 raster_meshes = 0;
			u64 raster_rejected_meshes = 0;
			u64 raster_vertices = 0;
			u64 camera_unavailable_frames = 0;
			u64 last_status_frame = 0;
			u32 draw_in_frame = 0;
			bool announced = false;
			motion_raster_frame pending_raster_frame;
		};

		probe_state& state()
		{
			static probe_state value;
			return value;
		}

		temporal_probe_state& temporal_state()
		{
			static temporal_probe_state value;
			return value;
		}

		motion_qualification_state& motion_state()
		{
			static motion_qualification_state value;
			return value;
		}

		u32 probe_version()
		{
			static const u32 value = []
			{
				const char* env = std::getenv("RPCS3_CHARACTER_VERTEX_PROBE");
				if (!env || !env[0] || std::strcmp(env, "0") == 0)
					return 0u;

				const u32 requested = static_cast<u32>(std::strtoul(env, nullptr, 10));
				if (requested >= task_identity_motion_version)
					return task_identity_motion_version;
				if (requested >= motion_raster_probe_version)
					return motion_raster_probe_version;
				if (requested >= motion_qualification_probe_version)
					return motion_qualification_probe_version;
				if (requested >= temporal_rsx_probe_version)
					return temporal_rsx_probe_version;
				return spu_writer_probe_version;
			}();
			return value;
		}

		void hash_bytes(u64& hash, const void* data, usz size)
		{
			const auto* bytes = static_cast<const u8*>(data);
			for (usz i = 0; i < size; ++i)
			{
				hash ^= bytes[i];
				hash *= fnv_prime;
			}
		}

		template <typename T>
		void hash_value(u64& hash, const T& value)
		{
			hash_bytes(hash, &value, sizeof(value));
		}

		u64 sampled_buffer_hash(const void* data, u32 size)
		{
			if (!data || !size)
				return 0;

			constexpr u32 max_samples = 256;
			const auto* bytes = static_cast<const u8*>(data);
			u64 hash = fnv_offset_basis;
			hash_value(hash, size);
			const u32 sample_count = std::min(size, max_samples);
			for (u32 i = 0; i < sample_count; ++i)
			{
				const u32 offset = sample_count == 1 ? 0 : static_cast<u32>((static_cast<u64>(i) * (size - 1)) / (sample_count - 1));
				hash ^= bytes[offset];
				hash *= fnv_prime;
			}
			return hash;
		}

		u16 read_be16(const u8* data)
		{
			return static_cast<u16>((static_cast<u16>(data[0]) << 8) | data[1]);
		}

		u32 read_be32(const u8* data)
		{
			return (static_cast<u32>(data[0]) << 24) |
				(static_cast<u32>(data[1]) << 16) |
				(static_cast<u32>(data[2]) << 8) |
				static_cast<u32>(data[3]);
		}

		f32 half_to_float(u16 value)
		{
			const f32 sign = (value & 0x8000) ? -1.f : 1.f;
			const u32 exponent = (value >> 10) & 0x1f;
			const u32 mantissa = value & 0x3ff;
			if (!exponent)
				return mantissa ? sign * std::ldexp(static_cast<f32>(mantissa), -24) : std::copysign(0.f, sign);
			if (exponent == 0x1f)
				return mantissa ? std::numeric_limits<f32>::quiet_NaN() : std::copysign(std::numeric_limits<f32>::infinity(), sign);
			return sign * std::ldexp(1.f + static_cast<f32>(mantissa) / 1024.f, static_cast<s32>(exponent) - 15);
		}

		bool decode_position(const vertex_attribute_desc& attribute, const u8* data, std::array<f32, 3>& result)
		{
			const u32 count = std::min<u32>(attribute.component_count, 3);
			if (!count)
				return false;

			for (u32 component = 0; component < count; ++component)
			{
				switch (static_cast<rsx::vertex_base_type>(attribute.type))
				{
				case rsx::vertex_base_type::f:
					result[component] = std::bit_cast<f32>(read_be32(data + component * 4));
					break;
				case rsx::vertex_base_type::sf:
					result[component] = half_to_float(read_be16(data + component * 2));
					break;
				case rsx::vertex_base_type::s1:
				{
					const s16 value = static_cast<s16>(read_be16(data + component * 2));
					result[component] = std::max(-1.f, static_cast<f32>(value) / 32767.f);
					break;
				}
				case rsx::vertex_base_type::s32k:
					result[component] = static_cast<f32>(static_cast<s16>(read_be16(data + component * 2)));
					break;
				case rsx::vertex_base_type::ub:
					result[component] = static_cast<f32>(data[component]) / 255.f;
					break;
				case rsx::vertex_base_type::ub256:
					result[component] = static_cast<f32>(data[component]);
					break;
				case rsx::vertex_base_type::cmp:
					return false;
				}
			}

			for (u32 component = count; component < 3; ++component)
				result[component] = 0.f;
			return std::isfinite(result[0]) && std::isfinite(result[1]) && std::isfinite(result[2]);
		}

		u64 hash_transform_constants(const void* transform_constants, const u16* constant_ids, u32 constant_id_count, bool indexed)
		{
			if (!transform_constants)
				return 0;

			const auto* constants = static_cast<const u8*>(transform_constants);
			u64 hash = fnv_offset_basis;
			if (!indexed && constant_ids && constant_id_count)
			{
				for (u32 i = 0; i < constant_id_count; ++i)
				{
					const u16 id = constant_ids[i];
					if (id >= 512)
						continue;
					hash_value(hash, id);
					hash_bytes(hash, constants + id * 16, 16);
				}
			}
			else
			{
				// Indexed shaders may address the full 512-vector file. Sampling 64
				// vectors retains temporal evidence without hashing 8 KiB per draw.
				for (u32 i = 0; i < 64; ++i)
				{
					const u16 id = static_cast<u16>((i * 511u) / 63u);
					hash_value(hash, id);
					hash_bytes(hash, constants + id * 16, 16);
				}
			}
			return hash;
		}

		u64 make_identity_hash(const temporal_record& record)
		{
			u64 hash = fnv_offset_basis;
			if (record.draw.task_identity_valid)
			{
				constexpr u32 task_identity_tag = 0x47574154; // 'GWAT'
				hash_value(hash, task_identity_tag);
				hash_value(hash, record.draw.task_format);
				hash_value(hash, record.draw.task_packed_count);
				hash_value(hash, record.draw.task_source_ea);
				hash_value(hash, record.draw.task_output_relative_offset);
			}
			hash_value(hash, record.draw.vertex_program_id);
			hash_value(hash, record.draw.fragment_program_id);
			hash_value(hash, record.draw.vertex_draw_count);
			hash_value(hash, record.draw.stream_vertex_count);
			hash_value(hash, record.draw.first_vertex);
			hash_value(hash, record.draw.attribute_mask);
			hash_value(hash, record.draw.stride);
			hash_value(hash, record.draw.primitive);
			hash_value(hash, record.draw.command);
			hash_value(hash, record.draw.restart_index_enabled);
			hash_value(hash, record.draw.restart_index);
			hash_value(hash, record.index_hash);
			for (u32 i = 0; i < record.draw.attribute_count; ++i)
			{
				const auto& attribute = record.draw.attributes[i];
				hash_value(hash, attribute.index);
				hash_value(hash, attribute.type);
				hash_value(hash, attribute.component_count);
				hash_value(hash, attribute.byte_size);
				hash_value(hash, attribute.offset);
				hash_value(hash, attribute.frequency);
				hash_value(hash, attribute.modulo);
			}
			return hash;
		}

		void append_triangle(std::vector<u32>& result, u32 a, u32 b, u32 c, u32 vertex_count)
		{
			if (a >= vertex_count || b >= vertex_count || c >= vertex_count || a == b || b == c || a == c)
				return;

			result.push_back(a);
			result.push_back(b);
			result.push_back(c);
		}

		void decode_triangle_indices(
			const rsx_draw_desc& draw,
			const void* index_data,
			u32 index_size,
			u32 vertex_count,
			std::vector<u32>& result)
		{
			std::vector<u32> source;
			if (static_cast<rsx::draw_command>(draw.command) == rsx::draw_command::indexed)
			{
				if (!index_data || !index_size || (draw.index_type != static_cast<u8>(rsx::index_array_type::u16) &&
					draw.index_type != static_cast<u8>(rsx::index_array_type::u32)))
				{
					return;
				}

				const auto* bytes = static_cast<const u8*>(index_data);
				const u32 element_size = draw.index_type == static_cast<u8>(rsx::index_array_type::u16) ? 2u : 4u;
				const u32 element_count = std::min(draw.index_count, index_size / element_size);
				source.reserve(element_count);
				for (u32 i = 0; i < element_count; ++i)
				{
					const u32 raw = element_size == 2 ? read_be16(bytes + i * 2) : read_be32(bytes + i * 4);
					if (draw.restart_index_enabled && raw == draw.restart_index)
					{
						source.push_back(umax);
						continue;
					}

					source.push_back(raw >= draw.first_vertex ? raw - draw.first_vertex : umax);
				}
			}
			else if (static_cast<rsx::draw_command>(draw.command) == rsx::draw_command::array)
			{
				const u32 element_count = std::min(draw.vertex_draw_count, vertex_count);
				source.resize(element_count);
				for (u32 i = 0; i < element_count; ++i)
					source[i] = i;
			}
			else
			{
				return;
			}

			result.clear();
			result.reserve(source.size() * 2);
			const auto primitive = static_cast<rsx::primitive_type>(draw.primitive);
			switch (primitive)
			{
			case rsx::primitive_type::triangles:
				for (usz i = 0; i + 2 < source.size(); i += 3)
					append_triangle(result, source[i], source[i + 1], source[i + 2], vertex_count);
				break;
			case rsx::primitive_type::triangle_strip:
			{
				std::array<u32, 2> previous{umax, umax};
				u32 strip_vertex = 0;
				for (const u32 index : source)
				{
					if (index == umax)
					{
						previous = {umax, umax};
						strip_vertex = 0;
						continue;
					}
					if (strip_vertex >= 2)
					{
						if (strip_vertex & 1)
							append_triangle(result, previous[1], previous[0], index, vertex_count);
						else
							append_triangle(result, previous[0], previous[1], index, vertex_count);
					}
					previous[0] = previous[1];
					previous[1] = index;
					strip_vertex++;
				}
				break;
			}
			case rsx::primitive_type::triangle_fan:
			case rsx::primitive_type::polygon:
			{
				u32 anchor = umax;
				u32 previous = umax;
				for (const u32 index : source)
				{
					if (index == umax)
					{
						anchor = previous = umax;
						continue;
					}
					if (anchor == umax)
						anchor = index;
					else if (previous != umax)
						append_triangle(result, anchor, previous, index, vertex_count);
					previous = index;
				}
				break;
			}
			case rsx::primitive_type::quads:
				for (usz i = 0; i + 3 < source.size(); i += 4)
				{
					append_triangle(result, source[i], source[i + 1], source[i + 2], vertex_count);
					append_triangle(result, source[i + 2], source[i + 3], source[i], vertex_count);
				}
				break;
			case rsx::primitive_type::quad_strip:
				for (usz i = 0; i + 3 < source.size(); i += 2)
				{
					append_triangle(result, source[i], source[i + 1], source[i + 2], vertex_count);
					append_triangle(result, source[i + 1], source[i + 3], source[i + 2], vertex_count);
				}
				break;
			default:
				break;
			}
		}

		u32 bloom_index(u32 page)
		{
			return (page * 0x9e3779b1u) >> 16;
		}

		void mark_pages(u32 start, u32 end)
		{
			auto& probe = state();
			const u32 first_page = start >> guest_page_shift;
			const u32 last_page = (end - 1) >> guest_page_shift;

			for (u32 page = first_page; page <= last_page; ++page)
			{
				const u32 bit = bloom_index(page);
				probe.page_bloom[bit >> 6].fetch_or(1ull << (bit & 63), std::memory_order_relaxed);
			}
		}

		bool pages_may_be_watched(u32 start, u32 end)
		{
			const auto& bloom = state().page_bloom;
			const u32 first_page = start >> guest_page_shift;
			const u32 last_page = (end - 1) >> guest_page_shift;

			for (u32 page = first_page; page <= last_page; ++page)
			{
				const u32 bit = bloom_index(page);
				if (bloom[bit >> 6].load(std::memory_order_relaxed) & (1ull << (bit & 63)))
					return true;
			}

			return false;
		}

		u32 register_lane_u32(const std::array<u8, register_bytes>& registers, usz reg, usz lane)
		{
			u32 value = 0;
			std::memcpy(&value, registers.data() + reg * register_size + lane * sizeof(u32), sizeof(value));
			return value;
		}

		void append_registers(std::string& context, const std::array<u8, register_bytes>& registers)
		{
			fmt::append(context, "\nSPU registers (host v128 lane order; scalar ABI value is lane 3):\n");
			for (usz reg = 0; reg < register_count; ++reg)
			{
				fmt::append(context, "r%03u = %08x %08x %08x %08x\n",
					reg,
					register_lane_u32(registers, reg, 0),
					register_lane_u32(registers, reg, 1),
					register_lane_u32(registers, reg, 2),
					register_lane_u32(registers, reg, 3));
			}
		}

		void append_stack_window(std::string& context, const probe_state& probe)
		{
			const u32 sp = register_lane_u32(probe.registers, 1, 3);
			fmt::append(context, "\nStack window from r1=0x%05x (raw LS bytes):\n", sp);
			if (sp >= local_store_size)
			{
				fmt::append(context, "stack pointer is outside local store\n");
				return;
			}

			const u32 end = std::min<u32>(static_cast<u32>(local_store_size), sp + 0x180);
			for (u32 address = sp; address < end; address += 16)
			{
				fmt::append(context, "%05x:", address);
				for (u32 byte = 0; byte < 16 && address + byte < end; ++byte)
					fmt::append(context, " %02x", probe.local_store[address + byte]);
				fmt::append(context, "\n");
			}
		}

		void write_capture_files(probe_state& probe)
		{
			const std::string base = fs::get_executable_dir();
			const std::string ls_path = base + "Character-Vertex-SPU-LS-v2.bin";
			const std::string registers_path = base + "Character-Vertex-SPU-Registers-v2.bin";
			const std::string context_path = base + "Character-Vertex-SPU-Context-v2.txt";

			const bool ls_ok = fs::write_file(ls_path, fs::rewrite, probe.local_store.data(), probe.local_store.size());
			const bool registers_ok = fs::write_file(registers_path, fs::rewrite, probe.registers.data(), probe.registers.size());

			const auto& capture = probe.capture;
			std::string context = fmt::format(
				"RPCS3 Character Vertex SPU Capture v2\n"
				"====================================\n"
				"This is the first verified SPU PUT overlapping an RSX character-position stream.\n"
				"The reported PC may be a JIT block PC; block_hash plus the live LS image identifies the exact overlay.\n\n"
				"spu_id=0x%08x\n"
				"lv2_id=0x%08x\n"
				"reported_pc=0x%05x\n"
				"block_hash=0x%016llx\n"
				"mfc_command=0x%02x\n"
				"mfc_tag=%u\n"
				"put_address=0x%08x\n"
				"put_lsa=0x%05x\n"
				"put_size=0x%x\n"
				"watched_start=0x%08x\n"
				"watched_end=0x%08x\n"
				"watch_frame=%u\n"
				"vertex_program_id=%u\n"
				"vertex_count=%u\n"
				"attribute_mask=0x%04x\n"
				"stride=%u\n"
				"r0_lr_lane3=0x%05x\n"
				"r1_sp_lane3=0x%05x\n",
				capture.spu_id,
				capture.lv2_id,
				capture.pc,
				capture.block_hash,
				capture.command,
				capture.tag,
				capture.address,
				capture.lsa,
				capture.size,
				capture.watch_start,
				capture.watch_end,
				capture.watch_frame,
				capture.vertex_program_id,
				capture.vertex_count,
				capture.attribute_mask,
				capture.stride,
				register_lane_u32(probe.registers, 0, 3),
				register_lane_u32(probe.registers, 1, 3));

			append_registers(context, probe.registers);
			append_stack_window(context, probe);
			const bool context_ok = fs::write_file(context_path, fs::rewrite, context);

			character_vertex_log.notice(
				"Character vertex one-shot files: LS=%s registers=%s context=%s directory='%s'.",
				ls_ok ? "saved" : "FAILED",
				registers_ok ? "saved" : "FAILED",
				context_ok ? "saved" : "FAILED",
				base);
		}

		temporal_record make_temporal_record(
			const rsx_draw_desc& draw,
			const void* stream_data,
			const void* index_data,
			u32 index_size,
			const void* transform_constants,
			const u16* constant_ids,
			u32 constant_id_count,
			u64 sequence,
			u32 draw_in_frame,
			bool full_diagnostics,
			bool full_geometry)
		{
			temporal_record record{};
			record.sequence = sequence;
			record.draw_in_frame = draw_in_frame;
			record.draw = draw;
			record.draw.attribute_count = std::min<u8>(draw.attribute_count, 16);
			record.index_hash = sampled_buffer_hash(index_data, index_size);
			if (full_diagnostics)
			{
				record.constants_hash = hash_transform_constants(
					transform_constants,
					constant_ids,
					constant_id_count,
					draw.indexed_constants);
			}

			const auto* stream = static_cast<const u8*>(stream_data);
			const u32 safe_vertex_count = draw.stride && stream_data
				? std::min(draw.stream_vertex_count, draw.stream_size / draw.stride)
				: 0;
			const u32 sample_count = std::min(safe_vertex_count, temporal_vertex_sample_count);
			const vertex_attribute_desc* position_attribute = nullptr;
			record.stream_hash = full_diagnostics ? fnv_offset_basis : 0;
			record.non_position_hash = full_diagnostics ? fnv_offset_basis : 0;

			for (u32 attribute_index = 0; attribute_index < record.draw.attribute_count; ++attribute_index)
			{
				const auto& attribute = record.draw.attributes[attribute_index];
				if (attribute.index >= 16 || !attribute.byte_size ||
					static_cast<u32>(attribute.offset) + attribute.byte_size > draw.stride)
				{
					continue;
				}

				if (attribute.index == 0)
					position_attribute = &record.draw.attributes[attribute_index];

				if (!full_diagnostics)
					continue;

				u64 hash = fnv_offset_basis;
				hash_value(hash, attribute.index);
				hash_value(hash, attribute.type);
				hash_value(hash, attribute.component_count);
				hash_value(hash, attribute.byte_size);
				hash_value(hash, attribute.offset);
				hash_value(hash, safe_vertex_count);
				for (u32 sample = 0; sample < sample_count; ++sample)
				{
					const u32 vertex = sample_count == 1 ? 0 : static_cast<u32>((static_cast<u64>(sample) * (safe_vertex_count - 1)) / (sample_count - 1));
					const u8* value = stream + static_cast<usz>(vertex) * draw.stride + attribute.offset;
					hash_bytes(hash, value, attribute.byte_size);
				}

				record.attribute_hashes[attribute.index] = hash;
				hash_value(record.stream_hash, attribute.index);
				hash_value(record.stream_hash, hash);
				if (attribute.index != 0)
				{
					hash_value(record.non_position_hash, attribute.index);
					hash_value(record.non_position_hash, hash);
				}
			}

			record.position_min.fill(std::numeric_limits<f32>::infinity());
			record.position_max.fill(-std::numeric_limits<f32>::infinity());
			record.position_samples.fill(std::numeric_limits<f32>::quiet_NaN());
			record.sampled_positions.fill(std::numeric_limits<f32>::quiet_NaN());
			record.sampled_position_count = static_cast<u8>(sample_count);
			if (position_attribute && sample_count)
			{
				for (u32 sample = 0; sample < sample_count; ++sample)
				{
					const u32 vertex = sample_count == 1 ? 0 : static_cast<u32>((static_cast<u64>(sample) * (safe_vertex_count - 1)) / (sample_count - 1));
					const u8* value = stream + static_cast<usz>(vertex) * draw.stride + position_attribute->offset;
					std::array<f32, 3> position{};
					if (!decode_position(*position_attribute, value, position))
						continue;

					record.sampled_position_valid_mask |= 1ull << sample;
					for (u32 component = 0; component < 3; ++component)
						record.sampled_positions[sample * 3 + component] = position[component];

					record.finite_position_samples++;
					for (u32 component = 0; component < 3; ++component)
					{
						record.position_min[component] = std::min(record.position_min[component], position[component]);
						record.position_max[component] = std::max(record.position_max[component], position[component]);
					}
				}

				for (u32 slot = 0; slot < 4; ++slot)
				{
					const u32 vertex = safe_vertex_count == 1 ? 0 : static_cast<u32>((static_cast<u64>(slot) * (safe_vertex_count - 1)) / 3);
					const u8* value = stream + static_cast<usz>(vertex) * draw.stride + position_attribute->offset;
					std::array<f32, 3> position{};
					if (!decode_position(*position_attribute, value, position))
						continue;

					record.position_sample_valid_mask |= static_cast<u8>(1u << slot);
					for (u32 component = 0; component < 3; ++component)
						record.position_samples[slot * 3 + component] = position[component];
				}
			}

			if (full_geometry && position_attribute && safe_vertex_count)
			{
				record.full_positions.resize(safe_vertex_count);
				for (u32 vertex = 0; vertex < safe_vertex_count; ++vertex)
				{
					const u8* value = stream + static_cast<usz>(vertex) * draw.stride + position_attribute->offset;
					if (!decode_position(*position_attribute, value, record.full_positions[vertex]))
						record.full_positions[vertex].fill(std::numeric_limits<f32>::quiet_NaN());
				}

				decode_triangle_indices(draw, index_data, index_size, safe_vertex_count, record.triangle_indices);
			}

			if (!record.finite_position_samples)
			{
				record.position_min.fill(std::numeric_limits<f32>::quiet_NaN());
				record.position_max.fill(std::numeric_limits<f32>::quiet_NaN());
			}

			record.identity_hash = make_identity_hash(record);
			return record;
		}

		bool project_to_ndc(
			const std::array<f32, 16>& matrix,
			const f32* position,
			std::array<f32, 2>& result)
		{
			const f32 x = position[0];
			const f32 y = position[1];
			const f32 z = position[2];
			const f32 clip_x = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
			const f32 clip_y = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
			const f32 clip_w = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15];
			if (!std::isfinite(clip_x) || !std::isfinite(clip_y) || !std::isfinite(clip_w) || clip_w <= 1.e-6f)
				return false;

			result = {clip_x / clip_w, clip_y / clip_w};
			return std::isfinite(result[0]) && std::isfinite(result[1]) &&
				std::abs(result[0]) <= 8.f && std::abs(result[1]) <= 8.f;
		}

		bool transform_to_clip(
			const std::array<f32, 16>& matrix,
			const std::array<f32, 3>& position,
			std::array<f32, 4>& result)
		{
			for (u32 row = 0; row < 4; ++row)
			{
				result[row] = matrix[row * 4] * position[0] +
					matrix[row * 4 + 1] * position[1] +
					matrix[row * 4 + 2] * position[2] +
					matrix[row * 4 + 3];
				if (!std::isfinite(result[row]))
					return false;
			}

			return std::abs(result[3]) > 1.e-7f;
		}

		bool sampled_world_rms(
			const temporal_record& current,
			const temporal_record& previous,
			f32& result)
		{
			const u32 sample_count = std::min<u32>(current.sampled_position_count, previous.sampled_position_count);
			double squared_sum = 0.;
			u32 valid_samples = 0;
			for (u32 sample = 0; sample < sample_count; ++sample)
			{
				const u64 sample_bit = 1ull << sample;
				if (!(current.sampled_position_valid_mask & sample_bit) || !(previous.sampled_position_valid_mask & sample_bit))
					continue;

				const f32* current_position = current.sampled_positions.data() + sample * 3;
				const f32* previous_position = previous.sampled_positions.data() + sample * 3;
				const f32 dx = previous_position[0] - current_position[0];
				const f32 dy = previous_position[1] - current_position[1];
				const f32 dz = previous_position[2] - current_position[2];
				const f32 squared = dx * dx + dy * dy + dz * dz;
				if (!std::isfinite(squared))
					continue;

				squared_sum += squared;
				valid_samples++;
			}

			if (valid_samples < motion_minimum_projected_samples)
				return false;

			result = static_cast<f32>(std::sqrt(squared_sum / valid_samples));
			return std::isfinite(result);
		}

		struct motion_pair_candidate
		{
			const temporal_record* current = nullptr;
			const temporal_record* previous = nullptr;
			usz current_occurrence = 0;
			usz previous_occurrence = 0;
			f32 world_rms = 0.f;
		};

		bool append_motion_raster_mesh(
			const temporal_record& current,
			const temporal_record& previous,
			const motion_frame_desc& frame,
			f32 world_rms,
			motion_raster_frame& result)
		{
			if (world_rms > motion_raster_world_rms_limit || current.full_positions.empty() ||
				previous.full_positions.size() != current.full_positions.size() || current.triangle_indices.empty())
			{
				return false;
			}

			const usz initial_size = result.vertices.size();
			for (usz triangle = 0; triangle + 2 < current.triangle_indices.size(); triangle += 3)
			{
				if (result.vertices.size() + 3 > motion_raster_vertex_limit)
					break;

				std::array<motion_raster_vertex, 3> vertices{};
				bool valid = true;
				for (u32 corner = 0; corner < 3; ++corner)
				{
					const u32 index = current.triangle_indices[triangle + corner];
					if (index >= current.full_positions.size() ||
						!transform_to_clip(frame.current_view_projection, current.full_positions[index], vertices[corner].current_clip) ||
						!transform_to_clip(frame.previous_view_projection, previous.full_positions[index], vertices[corner].previous_clip))
					{
						valid = false;
						break;
					}
				}

				if (valid)
					result.vertices.insert(result.vertices.end(), vertices.begin(), vertices.end());
			}

			return result.vertices.size() > initial_size;
		}

		bool make_motion_qualification_record(
			const temporal_record& current,
			const temporal_record& previous,
			const motion_frame_desc& frame,
			u64 sequence,
			motion_qualification_record& result)
		{
			if (!frame.render_width || !frame.render_height ||
				current.identity_hash != previous.identity_hash ||
				!current.sampled_position_count || !previous.sampled_position_count)
			{
				return false;
			}

			const u32 sample_count = std::min<u32>(current.sampled_position_count, previous.sampled_position_count);
			const f32 half_width = static_cast<f32>(frame.render_width) * 0.5f;
			const f32 half_height = static_cast<f32>(frame.render_height) * 0.5f;
			const f32 maximum_accepted_motion = std::hypot(static_cast<f32>(frame.render_width), static_cast<f32>(frame.render_height)) * 0.75f;

			double total_x_sum = 0.;
			double total_y_sum = 0.;
			double total_squared_sum = 0.;
			double camera_x_sum = 0.;
			double camera_y_sum = 0.;
			double camera_squared_sum = 0.;
			double object_x_sum = 0.;
			double object_y_sum = 0.;
			double object_squared_sum = 0.;
			double world_squared_sum = 0.;
			f32 maximum_total = 0.f;
			f32 maximum_object = 0.f;
			f32 maximum_world = 0.f;
			u32 projected_samples = 0;

			for (u32 sample = 0; sample < sample_count; ++sample)
			{
				const u64 sample_bit = 1ull << sample;
				if (!(current.sampled_position_valid_mask & sample_bit) || !(previous.sampled_position_valid_mask & sample_bit))
					continue;

				const f32* current_position = current.sampled_positions.data() + sample * 3;
				const f32* previous_position = previous.sampled_positions.data() + sample * 3;
				std::array<f32, 2> current_ndc{};
				std::array<f32, 2> previous_ndc{};
				std::array<f32, 2> camera_previous_ndc{};
				if (!project_to_ndc(frame.current_view_projection, current_position, current_ndc) ||
					!project_to_ndc(frame.previous_view_projection, previous_position, previous_ndc) ||
					!project_to_ndc(frame.previous_view_projection, current_position, camera_previous_ndc))
				{
					continue;
				}

				// Only qualify samples that contribute to, or lie very close to, the
				// current viewport. Previous positions may legitimately leave it.
				if (std::abs(current_ndc[0]) > 1.25f || std::abs(current_ndc[1]) > 1.25f)
					continue;

				const f32 total_x = (previous_ndc[0] - current_ndc[0]) * half_width;
				const f32 total_y = (previous_ndc[1] - current_ndc[1]) * half_height;
				const f32 camera_x = (camera_previous_ndc[0] - current_ndc[0]) * half_width;
				const f32 camera_y = (camera_previous_ndc[1] - current_ndc[1]) * half_height;
				const f32 object_x = total_x - camera_x;
				const f32 object_y = total_y - camera_y;
				const f32 total_length = std::hypot(total_x, total_y);
				const f32 camera_length = std::hypot(camera_x, camera_y);
				const f32 object_length = std::hypot(object_x, object_y);
				const f32 dx = previous_position[0] - current_position[0];
				const f32 dy = previous_position[1] - current_position[1];
				const f32 dz = previous_position[2] - current_position[2];
				const f32 world_length = std::sqrt(dx * dx + dy * dy + dz * dz);
				if (!std::isfinite(total_length) || !std::isfinite(camera_length) ||
					!std::isfinite(object_length) || !std::isfinite(world_length) ||
					total_length > maximum_accepted_motion)
				{
					continue;
				}

				total_x_sum += total_x;
				total_y_sum += total_y;
				total_squared_sum += static_cast<double>(total_length) * total_length;
				camera_x_sum += camera_x;
				camera_y_sum += camera_y;
				camera_squared_sum += static_cast<double>(camera_length) * camera_length;
				object_x_sum += object_x;
				object_y_sum += object_y;
				object_squared_sum += static_cast<double>(object_length) * object_length;
				world_squared_sum += static_cast<double>(world_length) * world_length;
				maximum_total = std::max(maximum_total, total_length);
				maximum_object = std::max(maximum_object, object_length);
				maximum_world = std::max(maximum_world, world_length);
				projected_samples++;
			}

			if (projected_samples < motion_minimum_projected_samples)
				return false;

			const f32 inverse_count = 1.f / projected_samples;
			result.sequence = sequence;
			result.frame_id = frame.frame_id;
			result.identity_hash = current.identity_hash;
			result.draw_in_frame = current.draw_in_frame;
			result.vertex_program_id = current.draw.vertex_program_id;
			result.fragment_program_id = current.draw.fragment_program_id;
			result.vertex_draw_count = current.draw.vertex_draw_count;
			result.stream_vertex_count = current.draw.stream_vertex_count;
			result.first_vertex = current.draw.first_vertex;
			result.current_address = current.draw.stream_address;
			result.previous_address = previous.draw.stream_address;
			result.projected_samples = projected_samples;
			result.mean_total_x = static_cast<f32>(total_x_sum * inverse_count);
			result.mean_total_y = static_cast<f32>(total_y_sum * inverse_count);
			result.rms_total_pixels = static_cast<f32>(std::sqrt(total_squared_sum * inverse_count));
			result.max_total_pixels = maximum_total;
			result.mean_camera_x = static_cast<f32>(camera_x_sum * inverse_count);
			result.mean_camera_y = static_cast<f32>(camera_y_sum * inverse_count);
			result.rms_camera_pixels = static_cast<f32>(std::sqrt(camera_squared_sum * inverse_count));
			result.mean_object_x = static_cast<f32>(object_x_sum * inverse_count);
			result.mean_object_y = static_cast<f32>(object_y_sum * inverse_count);
			result.rms_object_pixels = static_cast<f32>(std::sqrt(object_squared_sum * inverse_count));
			result.max_object_pixels = maximum_object;
			result.rms_world_delta = static_cast<f32>(std::sqrt(world_squared_sum * inverse_count));
			result.max_world_delta = maximum_world;
			result.camera_confidence = frame.camera_confidence;
			return true;
		}

		std::string motion_arm_path()
		{
			return fs::get_executable_dir() + (probe_version() == motion_raster_probe_version
				? "Character-Motion-Raster-v6.arm"
				: "Character-Motion-v5.arm");
		}

		void write_motion_qualification_files(motion_qualification_state& probe)
		{
			const std::string base = fs::get_executable_dir();
			const bool raster_mode = probe_version() == motion_raster_probe_version;
			const std::string csv_path = base + (raster_mode ? "Character-Motion-Raster-v6.csv" : "Character-Motion-Qualification-v5.csv");
			const std::string summary_path = base + (raster_mode ? "Character-Motion-Raster-v6-summary.txt" : "Character-Motion-Qualification-v5-summary.txt");
			std::string csv;
			csv.reserve(probe.results.size() * 360);
			csv = "sequence,frame,draw_in_frame,identity_hash,vp_id,fp_id,draw_count,stream_vertices,first_vertex,current_address,previous_address,projected_samples,mean_total_x,mean_total_y,rms_total_pixels,max_total_pixels,mean_camera_x,mean_camera_y,rms_camera_pixels,mean_object_x,mean_object_y,rms_object_pixels,max_object_pixels,rms_world_delta,max_world_delta,camera_confidence\n";

			std::unordered_set<u64> unique_identities;
			std::unordered_set<u64> moving_identities;
			std::unordered_set<u64> multi_address_identities;
			std::unordered_map<u64, std::pair<u64, u32>> identity_runs;
			std::unordered_map<u64, u32> longest_runs;
			double total_rms_sum = 0.;
			double object_rms_sum = 0.;
			double confidence_sum = 0.;
			u64 projected_sample_sum = 0;
			f32 maximum_total_motion = 0.f;
			f32 maximum_object_motion = 0.f;

			for (const auto& record : probe.results)
			{
				unique_identities.insert(record.identity_hash);
				if (record.rms_object_pixels >= 0.05f || record.rms_world_delta >= 1.e-4f)
					moving_identities.insert(record.identity_hash);
				if (record.current_address != record.previous_address)
					multi_address_identities.insert(record.identity_hash);

				auto& run = identity_runs[record.identity_hash];
				run.second = run.first + 1 == record.frame_id ? run.second + 1 : 1;
				run.first = record.frame_id;
				longest_runs[record.identity_hash] = std::max(longest_runs[record.identity_hash], run.second);
				total_rms_sum += record.rms_total_pixels;
				object_rms_sum += record.rms_object_pixels;
				confidence_sum += record.camera_confidence;
				projected_sample_sum += record.projected_samples;
				maximum_total_motion = std::max(maximum_total_motion, record.max_total_pixels);
				maximum_object_motion = std::max(maximum_object_motion, record.max_object_pixels);

				fmt::append(csv,
					"%llu,%llu,%u,0x%016llx,%u,%u,%u,%u,%u,0x%08x,0x%08x,%u,"
					"%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.6g\n",
					record.sequence,
					record.frame_id,
					record.draw_in_frame,
					record.identity_hash,
					record.vertex_program_id,
					record.fragment_program_id,
					record.vertex_draw_count,
					record.stream_vertex_count,
					record.first_vertex,
					record.current_address,
					record.previous_address,
					record.projected_samples,
					record.mean_total_x,
					record.mean_total_y,
					record.rms_total_pixels,
					record.max_total_pixels,
					record.mean_camera_x,
					record.mean_camera_y,
					record.rms_camera_pixels,
					record.mean_object_x,
					record.mean_object_y,
					record.rms_object_pixels,
					record.max_object_pixels,
					record.rms_world_delta,
					record.max_world_delta,
					record.camera_confidence);
			}

			usz identities_with_long_runs = 0;
			for (const auto& entry : longest_runs)
			{
				if (entry.second >= 30)
					identities_with_long_runs++;
			}

			const double inverse_result_count = probe.results.empty() ? 0. : 1. / probe.results.size();
			const std::string summary = fmt::format(
				"RPCS3 Character Motion %s\n"
				"========================================\n"
				"This capture pairs final current/previous guest vertices by mesh identity and minimum sampled world-space distance, then projects them with the qualified current/previous camera matrices.\n"
				"Vectors use the DLSS low-resolution pixel convention: previous pixel minus current pixel.\n\n"
				"first_frame=%llu\n"
				"last_frame=%llu\n"
				"frame_span=%llu\n"
				"matched_mesh_pairs=%llu\n"
				"valid_projected_pairs=%llu\n"
				"rejected_projection_pairs=%llu\n"
				"ambiguous_identity_groups=%llu\n"
				"reassociated_mesh_pairs=%llu\n"
				"unmatched_current_meshes=%llu\n"
				"raster_frames=%llu\n"
				"raster_meshes=%llu\n"
				"raster_rejected_meshes=%llu\n"
				"raster_vertices=%llu\n"
				"camera_unavailable_frames=%llu\n"
				"unique_mesh_identities=%llu\n"
				"multi_address_identities=%llu\n"
				"identities_with_object_motion=%llu\n"
				"identities_with_30_frame_runs=%llu\n"
				"average_projected_samples=%.3f\n"
				"average_total_rms_pixels=%.6f\n"
				"average_object_rms_pixels=%.6f\n"
				"maximum_total_pixels=%.6f\n"
				"maximum_object_pixels=%.6f\n"
				"average_camera_confidence=%.6f\n\n"
				"A useful result has consecutive identities, multiple rotating guest addresses, finite total motion, and a non-zero object residual distinct from camera motion.\n",
				raster_mode ? "Raster v6" : "Qualification v5",
				probe.first_frame,
				probe.last_frame,
				probe.last_frame >= probe.first_frame ? probe.last_frame - probe.first_frame + 1 : 0,
				probe.matched_mesh_pairs,
				probe.results.size(),
				probe.rejected_projection_pairs,
				probe.ambiguous_identity_groups,
				probe.reassociated_mesh_pairs,
				probe.unmatched_current_meshes,
				probe.raster_frames,
				probe.raster_meshes,
				probe.raster_rejected_meshes,
				probe.raster_vertices,
				probe.camera_unavailable_frames,
				unique_identities.size(),
				multi_address_identities.size(),
				moving_identities.size(),
				identities_with_long_runs,
				probe.results.empty() ? 0. : projected_sample_sum * inverse_result_count,
				total_rms_sum * inverse_result_count,
				object_rms_sum * inverse_result_count,
				maximum_total_motion,
				maximum_object_motion,
				confidence_sum * inverse_result_count);

			const bool csv_ok = fs::write_file(csv_path, fs::rewrite, csv);
			const bool summary_ok = fs::write_file(summary_path, fs::rewrite, summary);
			character_vertex_log.notice(
				"Character motion %s files: csv=%s summary=%s records=%llu directory='%s'.",
				raster_mode ? "raster v6" : "qualification v5",
				csv_ok ? "saved" : "FAILED",
				summary_ok ? "saved" : "FAILED",
				probe.results.size(),
				base);
		}

		void write_temporal_files(temporal_probe_state& probe)
		{
			const std::string base = fs::get_executable_dir();
			const std::string csv_path = base + "Character-Temporal-RSX-v3.csv";
			const std::string summary_path = base + "Character-Temporal-RSX-v3-summary.txt";

			std::string csv;
			csv.reserve(probe.records.size() * 720);
			csv = "sequence,frame,target_draw_in_frame,vp_id,fp_id,primitive,command,draw_count,stream_vertices,first_vertex,stream_address,stream_size,stride,attribute_mask,index_address,index_count,index_type,index_hash,constants_hash,stream_hash,non_position_hash,identity_hash,finite_position_samples,min_x,min_y,min_z,max_x,max_y,max_z,position_sample_mask,p0_x,p0_y,p0_z,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,p3_x,p3_y,p3_z,layout";
			for (u32 i = 0; i < 16; ++i)
				fmt::append(csv, ",a%02u_hash", i);
			fmt::append(csv, "\n");

			std::unordered_set<u32> unique_addresses;
			std::unordered_set<u64> unique_identities;
			std::unordered_map<u64, std::unordered_set<u32>> identity_addresses;
			std::unordered_map<u64, u64> last_position_hash;
			std::unordered_set<u64> position_changing_identities;

			for (const auto& record : probe.records)
			{
				unique_addresses.insert(record.draw.stream_address);
				unique_identities.insert(record.identity_hash);
				identity_addresses[record.identity_hash].insert(record.draw.stream_address);
				const u64 position_hash = record.attribute_hashes[0];
				if (const auto found = last_position_hash.find(record.identity_hash);
					found != last_position_hash.end() && found->second != position_hash)
				{
					position_changing_identities.insert(record.identity_hash);
				}
				last_position_hash[record.identity_hash] = position_hash;

				fmt::append(csv,
					"%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%u,0x%08x,%u,%u,0x%04x,0x%08x,%u,%u,0x%016llx,0x%016llx,0x%016llx,0x%016llx,0x%016llx,%u,"
					"%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,0x%02x,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,\"",
					record.sequence,
					record.draw.frame_id,
					record.draw_in_frame,
					record.draw.vertex_program_id,
					record.draw.fragment_program_id,
					record.draw.primitive,
					record.draw.command,
					record.draw.vertex_draw_count,
					record.draw.stream_vertex_count,
					record.draw.first_vertex,
					record.draw.stream_address,
					record.draw.stream_size,
					record.draw.stride,
					record.draw.attribute_mask,
					record.draw.index_address,
					record.draw.index_count,
					record.draw.index_type,
					record.index_hash,
					record.constants_hash,
					record.stream_hash,
					record.non_position_hash,
					record.identity_hash,
					record.finite_position_samples,
					record.position_min[0], record.position_min[1], record.position_min[2],
					record.position_max[0], record.position_max[1], record.position_max[2],
					record.position_sample_valid_mask,
					record.position_samples[0], record.position_samples[1], record.position_samples[2],
					record.position_samples[3], record.position_samples[4], record.position_samples[5],
					record.position_samples[6], record.position_samples[7], record.position_samples[8],
					record.position_samples[9], record.position_samples[10], record.position_samples[11]);

				for (u32 i = 0; i < record.draw.attribute_count; ++i)
				{
					const auto& attribute = record.draw.attributes[i];
					if (i)
						fmt::append(csv, "|");
					fmt::append(csv, "%u:%u:%u:%u:%u:%u:%u",
						attribute.index,
						attribute.type,
						attribute.component_count,
						attribute.byte_size,
						attribute.offset,
						attribute.frequency,
						attribute.modulo ? 1 : 0);
				}
				fmt::append(csv, "\"");
				for (const u64 hash : record.attribute_hashes)
					fmt::append(csv, ",0x%016llx", hash);
				fmt::append(csv, "\n");
			}

			usz multi_address_identities = 0;
			for (const auto& [identity, addresses] : identity_addresses)
			{
				if (addresses.size() > 1)
					multi_address_identities++;
			}

			const std::string summary = fmt::format(
				"RPCS3 Character Temporal RSX Capture v3\n"
				"=======================================\n"
				"This capture samples final guest vertex streams at the RSX draw boundary.\n"
				"It does not scan SPU DMA traffic and it does not copy complete vertex buffers.\n\n"
				"first_frame=%llu\n"
				"last_frame=%llu\n"
				"frame_span=%llu\n"
				"records=%llu\n"
				"record_limit=%llu\n"
				"unique_stream_addresses=%llu\n"
				"unique_identity_signatures=%llu\n"
				"identities_using_multiple_addresses=%llu\n"
				"identities_with_sampled_position_changes=%llu\n\n"
				"Identity combines shader IDs, topology/index sample hash, counts and vertex layout, but excludes the stream address.\n"
				"a00_hash is the sampled position attribute. Other attribute hashes reveal stable UV/topology-associated data.\n"
				"A multi-address identity with changing a00_hash is the key evidence needed for temporal vertex correspondence.\n",
				probe.first_frame,
				probe.last_frame,
				probe.last_frame >= probe.first_frame ? probe.last_frame - probe.first_frame + 1 : 0,
				probe.records.size(),
				temporal_record_limit,
				unique_addresses.size(),
				unique_identities.size(),
				multi_address_identities,
				position_changing_identities.size());

			const bool csv_ok = fs::write_file(csv_path, fs::rewrite, csv);
			const bool summary_ok = fs::write_file(summary_path, fs::rewrite, summary);
			character_vertex_log.notice(
				"Character temporal RSX v3 files: csv=%s summary=%s records=%llu directory='%s'.",
				csv_ok ? "saved" : "FAILED",
				summary_ok ? "saved" : "FAILED",
				probe.records.size(),
				base);
		}

		void report_temporal(u64 frame_id)
		{
			auto& probe = temporal_state();
			if (probe.phase == temporal_written)
				return;

			if (!probe.announced)
			{
				character_vertex_log.notice(
					"Character temporal RSX probe v3 enabled: target mask=0x%04x, frame span=%llu, record limit=%llu; SPU DMA observation is disabled.",
					character_attribute_mask,
					temporal_capture_frame_span,
					temporal_record_limit);
				probe.announced = true;
			}

			if (probe.phase == temporal_capturing &&
				(frame_id >= probe.first_frame + temporal_capture_frame_span || probe.records.size() >= temporal_record_limit))
			{
				probe.phase = temporal_ready;
			}

			if (probe.phase == temporal_ready)
			{
				write_temporal_files(probe);
				probe.phase = temporal_written;
				character_vertex_log.notice(
					"Character temporal RSX v3 capture completed at frame=%llu; the probe is now inactive.",
					frame_id);
			}

			if (!probe.last_status_frame || frame_id >= probe.last_status_frame + 120)
			{
				character_vertex_log.notice(
					"Character temporal RSX v3 status: frame=%llu phase=%u records=%llu first=%llu last=%llu.",
					frame_id,
					static_cast<u32>(probe.phase),
					probe.records.size(),
					probe.first_frame,
					probe.last_frame);
				probe.last_status_frame = frame_id;
			}
		}
	}

	bool enabled()
	{
		return probe_version() != 0;
	}

	bool spu_writer_enabled()
	{
		return probe_version() == spu_writer_probe_version;
	}

	bool rsx_draw_enabled()
	{
		if (probe_version() == task_identity_motion_version)
			return true;
		if (probe_version() == motion_qualification_probe_version || probe_version() == motion_raster_probe_version)
			return motion_state().collect_draws.load(std::memory_order_relaxed);
		if (probe_version() != temporal_rsx_probe_version)
			return false;

		const auto phase = temporal_state().phase;
		return phase == temporal_waiting || phase == temporal_capturing;
	}

	bool task_identity_required()
	{
		return probe_version() == task_identity_motion_version;
	}

	void observe_rsx_draw(
		const rsx_draw_desc& draw,
		const void* stream_data,
		const void* index_data,
		u32 index_size,
		const void* transform_constants,
		const u16* constant_ids,
		u32 constant_id_count)
	{
		const u32 version = probe_version();
		if ((version != temporal_rsx_probe_version && version != motion_qualification_probe_version &&
			version != motion_raster_probe_version && version != task_identity_motion_version) ||
			!stream_data || !draw.stream_size || !draw.stride)
			return;

		if (version == motion_qualification_probe_version || version == motion_raster_probe_version ||
			version == task_identity_motion_version)
		{
			auto& probe = motion_state();
			if (version == task_identity_motion_version && !draw.task_identity_valid)
				return;
			if ((version != task_identity_motion_version && !probe.collect_draws.load(std::memory_order_relaxed)) ||
				probe.phase != motion_capturing)
				return;

			if (probe.current_frame != draw.frame_id)
			{
				probe.current_records.clear();
				probe.current_frame = draw.frame_id;
				probe.draw_in_frame = 0;
			}

			probe.current_records.emplace_back(make_temporal_record(
				draw,
				stream_data,
				index_data,
				index_size,
				transform_constants,
				constant_ids,
				constant_id_count,
				probe.observed_draw_sequence++,
				probe.draw_in_frame++,
				false,
				version == motion_raster_probe_version || version == task_identity_motion_version));
			return;
		}

		auto& probe = temporal_state();
		if (probe.phase == temporal_ready || probe.phase == temporal_written)
			return;

		if (probe.phase == temporal_waiting)
		{
			probe.records.reserve(temporal_record_limit);
			probe.first_frame = draw.frame_id;
			probe.phase = temporal_capturing;
			character_vertex_log.notice(
				"Character temporal RSX v3 capture started at frame=%llu vp=%u vertices=%u address=0x%08x.",
				draw.frame_id,
				draw.vertex_program_id,
				draw.vertex_draw_count,
				draw.stream_address);
		}

		if (draw.frame_id >= probe.first_frame + temporal_capture_frame_span || probe.records.size() >= temporal_record_limit)
		{
			probe.phase = temporal_ready;
			return;
		}

		if (probe.current_frame != draw.frame_id)
		{
			probe.current_frame = draw.frame_id;
			probe.draw_in_frame = 0;
		}

		probe.records.emplace_back(make_temporal_record(
			draw,
			stream_data,
			index_data,
			index_size,
			transform_constants,
			constant_ids,
			constant_id_count,
			probe.records.size(),
			probe.draw_in_frame++,
			true,
			false));
		probe.last_frame = draw.frame_id;
		if (probe.records.size() >= temporal_record_limit)
			probe.phase = temporal_ready;
	}

	void finalize_motion_frame(const motion_frame_desc& frame)
	{
		const u32 version = probe_version();
		if (version != motion_qualification_probe_version && version != motion_raster_probe_version &&
			version != task_identity_motion_version)
			return;
		const bool task_identity_mode = version == task_identity_motion_version;
		const bool raster_mode = version == motion_raster_probe_version || task_identity_mode;
		const u32 active_version = task_identity_mode
			? task_identity_motion_version
			: (raster_mode ? motion_raster_probe_version : motion_qualification_probe_version);

		auto& probe = motion_state();
		if (!probe.announced)
		{
			probe.announced = true;
			if (task_identity_mode)
			{
				probe.current_records.reserve(64);
				probe.previous_records.reserve(64);
				probe.first_frame = frame.frame_id + 1;
				probe.last_frame = frame.frame_id;
				probe.phase = motion_capturing;
				probe.collect_draws.store(true, std::memory_order_relaxed);
				character_vertex_log.success(
					"Ascension task-identity geometry motion v%u is active continuously from frame=%llu; no Arm step or bounded diagnostic capture is used.",
					active_version,
					probe.first_frame);
			}
			else
			{
				character_vertex_log.notice(
					"Character motion %s v%u is waiting for arm file '%s'. No vertex sampling occurs before it is armed.",
					raster_mode ? "raster" : "qualification",
					active_version,
					motion_arm_path());
			}
		}

		if (!task_identity_mode && probe.phase == motion_waiting_for_arm)
		{
			const std::string arm_path = motion_arm_path();
			if (!fs::is_file(arm_path))
				return;

			fs::remove_file(arm_path);
			probe.current_records.reserve(256);
			probe.previous_records.reserve(256);
			probe.results.reserve(motion_qualification_record_limit);
			probe.first_frame = frame.frame_id + 1;
			probe.last_frame = frame.frame_id;
			probe.phase = motion_capturing;
			probe.collect_draws.store(true, std::memory_order_relaxed);
			character_vertex_log.notice(
				"Character motion %s v%u armed at frame=%llu; capture begins at frame=%llu for %llu frames.",
				raster_mode ? "raster" : "qualification",
				active_version,
				frame.frame_id,
				probe.first_frame,
				motion_qualification_frame_span);
			return;
		}

		if (probe.phase != motion_capturing || frame.frame_id < probe.first_frame)
			return;

		const bool current_frame_available = probe.current_frame == frame.frame_id && !probe.current_records.empty();
		const bool consecutive_geometry = probe.previous_frame != umax && probe.previous_frame + 1 == frame.frame_id;
		const bool camera_available = frame.has_camera_matrices && !frame.reset_accumulation && !frame.camera_cut_detected;
		if (raster_mode)
		{
			probe.pending_raster_frame.frame_id = frame.frame_id;
			probe.pending_raster_frame.render_width = frame.render_width;
			probe.pending_raster_frame.render_height = frame.render_height;
			probe.pending_raster_frame.matched_meshes = 0;
			probe.pending_raster_frame.rejected_meshes = 0;
			probe.pending_raster_frame.vertices.clear();
		}
		if (current_frame_available && consecutive_geometry && camera_available)
		{
			std::unordered_map<u64, std::vector<const temporal_record*>> current_by_identity;
			std::unordered_map<u64, std::vector<const temporal_record*>> previous_by_identity;
			current_by_identity.reserve(probe.current_records.size());
			previous_by_identity.reserve(probe.previous_records.size());
			for (const auto& current : probe.current_records)
				current_by_identity[current.identity_hash].push_back(&current);
			for (const auto& previous : probe.previous_records)
				previous_by_identity[previous.identity_hash].push_back(&previous);

			for (const auto& [identity, current_group] : current_by_identity)
			{
				const auto found = previous_by_identity.find(identity);
				if (found == previous_by_identity.end())
				{
					probe.unmatched_current_meshes += current_group.size();
					continue;
				}

				const auto& previous_group = found->second;
				if (current_group.size() > 1 || previous_group.size() > 1)
					probe.ambiguous_identity_groups++;

				std::vector<motion_pair_candidate> candidates;
				candidates.reserve(current_group.size() * previous_group.size());
				for (usz current_occurrence = 0; current_occurrence < current_group.size(); ++current_occurrence)
				{
					for (usz previous_occurrence = 0; previous_occurrence < previous_group.size(); ++previous_occurrence)
					{
						f32 world_rms = 0.f;
						if (!sampled_world_rms(*current_group[current_occurrence], *previous_group[previous_occurrence], world_rms))
							continue;

						candidates.push_back({
							.current = current_group[current_occurrence],
							.previous = previous_group[previous_occurrence],
							.current_occurrence = current_occurrence,
							.previous_occurrence = previous_occurrence,
							.world_rms = world_rms,
						});
					}
				}

				std::sort(candidates.begin(), candidates.end(), [](const motion_pair_candidate& lhs, const motion_pair_candidate& rhs)
				{
					if (lhs.world_rms != rhs.world_rms)
						return lhs.world_rms < rhs.world_rms;
					const u32 lhs_draw_distance = lhs.current->draw_in_frame > lhs.previous->draw_in_frame
						? lhs.current->draw_in_frame - lhs.previous->draw_in_frame
						: lhs.previous->draw_in_frame - lhs.current->draw_in_frame;
					const u32 rhs_draw_distance = rhs.current->draw_in_frame > rhs.previous->draw_in_frame
						? rhs.current->draw_in_frame - rhs.previous->draw_in_frame
						: rhs.previous->draw_in_frame - rhs.current->draw_in_frame;
					return lhs_draw_distance < rhs_draw_distance;
				});

				std::vector<bool> current_used(current_group.size());
				std::vector<bool> previous_used(previous_group.size());
				usz matched_current = 0;
				for (const auto& candidate : candidates)
				{
					if (current_used[candidate.current_occurrence] || previous_used[candidate.previous_occurrence])
						continue;

					current_used[candidate.current_occurrence] = true;
					previous_used[candidate.previous_occurrence] = true;
					matched_current++;
					if (candidate.current_occurrence != candidate.previous_occurrence)
						probe.reassociated_mesh_pairs++;

					probe.matched_mesh_pairs++;
					if (raster_mode)
					{
						if (append_motion_raster_mesh(
								*candidate.current,
								*candidate.previous,
								frame,
								candidate.world_rms,
								probe.pending_raster_frame))
						{
							probe.pending_raster_frame.matched_meshes++;
							probe.raster_meshes++;
						}
						else
						{
							probe.pending_raster_frame.rejected_meshes++;
							probe.raster_rejected_meshes++;
						}
					}

					if (!task_identity_mode)
					{
						motion_qualification_record result{};
						if (make_motion_qualification_record(
								*candidate.current,
								*candidate.previous,
								frame,
								probe.results.size(),
								result))
						{
							probe.results.emplace_back(result);
						}
						else
						{
							probe.rejected_projection_pairs++;
						}
					}

					if (!task_identity_mode && probe.results.size() >= motion_qualification_record_limit)
						break;
				}

				probe.unmatched_current_meshes += current_group.size() - matched_current;
				if (!task_identity_mode && probe.results.size() >= motion_qualification_record_limit)
					break;
			}
		}
		else if (!camera_available)
		{
			probe.camera_unavailable_frames++;
		}

		if (raster_mode && !probe.pending_raster_frame.vertices.empty())
		{
			probe.raster_frames++;
			probe.raster_vertices += probe.pending_raster_frame.vertices.size();
		}

		if (current_frame_available)
		{
			// Keep two reserved buffers and swap their roles. Moving here would
			// discard the old previous-frame allocation and force a fresh vertex
			// record allocation on almost every captured frame.
			probe.previous_records.swap(probe.current_records);
			probe.current_records.clear();
			probe.previous_frame = frame.frame_id;
		}
		else
		{
			probe.previous_records.clear();
			probe.previous_frame = umax;
		}
		probe.current_frame = umax;
		probe.draw_in_frame = 0;
		probe.last_frame = frame.frame_id;

		if (!probe.last_status_frame || frame.frame_id >= probe.last_status_frame + 120)
		{
			character_vertex_log.notice(
				"Character motion %s v%u status: frame=%llu pairs=%llu valid=%llu rejected=%llu ambiguous=%llu reassociated=%llu unmatched=%llu raster_vertices=%llu camera_missing=%llu.",
				task_identity_mode ? "task-identity" : (raster_mode ? "raster" : "qualification"),
				active_version,
				frame.frame_id,
				probe.matched_mesh_pairs,
				probe.results.size(),
				probe.rejected_projection_pairs,
				probe.ambiguous_identity_groups,
				probe.reassociated_mesh_pairs,
				probe.unmatched_current_meshes,
				probe.raster_vertices,
				probe.camera_unavailable_frames);
			probe.last_status_frame = frame.frame_id;
		}

		if (!task_identity_mode && (frame.frame_id >= probe.first_frame + motion_qualification_frame_span - 1 ||
			probe.results.size() >= motion_qualification_record_limit)
			)
		{
			probe.collect_draws.store(false, std::memory_order_relaxed);
			write_motion_qualification_files(probe);
			probe.phase = motion_written;
			character_vertex_log.notice(
				"Character motion %s v%u completed at frame=%llu; vertex sampling is now inactive.",
				raster_mode ? "raster" : "qualification",
				active_version,
				frame.frame_id);
		}
	}

	bool take_motion_raster_frame(u64 frame_id, motion_raster_frame& result)
	{
		if (probe_version() != motion_raster_probe_version && probe_version() != task_identity_motion_version)
			return false;

		auto& pending = motion_state().pending_raster_frame;
		if (pending.frame_id != frame_id || pending.vertices.empty())
			return false;

		result.frame_id = pending.frame_id;
		result.render_width = pending.render_width;
		result.render_height = pending.render_height;
		result.matched_meshes = pending.matched_meshes;
		result.rejected_meshes = pending.rejected_meshes;
		result.vertices.clear();
		result.vertices.swap(pending.vertices);
		pending.frame_id = 0;
		pending.render_width = 0;
		pending.render_height = 0;
		pending.matched_meshes = 0;
		pending.rejected_meshes = 0;
		return true;
	}

	void watch_vertex_range(
		u64 frame_id,
		u32 vertex_program_id,
		u32 vertex_count,
		u16 attribute_mask,
		u8 stride,
		u32 address,
		u32 size)
	{
		auto& probe = state();
		if (!spu_writer_enabled() || probe.capture_phase.load(std::memory_order_relaxed) != capture_idle || !size || size > max_watched_range_size)
			return;

		const u64 end64 = static_cast<u64>(address) + size;
		if (end64 > std::numeric_limits<u32>::max())
			return;

		const u32 end = static_cast<u32>(end64);
		const u64 packed_range = (static_cast<u64>(end) << 32) | address;
		const u64 frame_and_program = ((frame_id & 0xffffffffull) << 32) | vertex_program_id;
		const u64 draw_info = vertex_count | (static_cast<u64>(attribute_mask) << 32) | (static_cast<u64>(stride) << 48);

		for (auto& watch : probe.watches)
		{
			if (watch.range.load(std::memory_order_acquire) == packed_range)
			{
				watch.frame_and_program.store(frame_and_program, std::memory_order_relaxed);
				watch.draw_info.store(draw_info, std::memory_order_relaxed);
				mark_pages(address, end);
				return;
			}
		}

		auto& watch = probe.watches[probe.watch_cursor.fetch_add(1, std::memory_order_relaxed) & (watch_slot_count - 1)];
		watch.range.store(0, std::memory_order_release);
		watch.frame_and_program.store(frame_and_program, std::memory_order_relaxed);
		watch.draw_info.store(draw_info, std::memory_order_relaxed);
		watch.range.store(packed_range, std::memory_order_release);
		mark_pages(address, end);
		probe.watched_range_count.fetch_add(1, std::memory_order_relaxed);
	}

	void observe_spu_put(
		u32 spu_id,
		u32 lv2_id,
		u32 pc,
		u64 block_hash,
		u32 address,
		u32 lsa,
		u32 size,
		u8 command,
		u8 tag,
		const void* local_store,
		const void* registers)
	{
		auto& probe = state();
		if (!spu_writer_enabled() || probe.capture_phase.load(std::memory_order_relaxed) != capture_idle || !size || !local_store || !registers)
			return;

		probe.put_count.fetch_add(1, std::memory_order_relaxed);
		const u64 end64 = static_cast<u64>(address) + size;
		if (end64 > std::numeric_limits<u32>::max())
			return;

		const u32 end = static_cast<u32>(end64);
		if (!pages_may_be_watched(address, end))
			return;

		probe.bloom_match_count.fetch_add(1, std::memory_order_relaxed);
		for (const auto& watch : probe.watches)
		{
			const u64 packed_range = watch.range.load(std::memory_order_acquire);
			if (!packed_range)
				continue;

			const u32 watch_start = static_cast<u32>(packed_range);
			const u32 watch_end = static_cast<u32>(packed_range >> 32);
			if (address >= watch_end || end <= watch_start)
				continue;

			probe.overlap_count.fetch_add(1, std::memory_order_relaxed);
			u32 expected = capture_idle;
			if (!probe.capture_phase.compare_exchange_strong(expected, capture_copying, std::memory_order_acq_rel))
				return;

			const u64 frame_and_program = watch.frame_and_program.load(std::memory_order_relaxed);
			const u64 draw_info = watch.draw_info.load(std::memory_order_relaxed);
			probe.capture.spu_id = spu_id;
			probe.capture.lv2_id = lv2_id;
			probe.capture.pc = pc;
			probe.capture.block_hash = block_hash;
			probe.capture.address = address;
			probe.capture.lsa = lsa;
			probe.capture.size = size;
			probe.capture.command = command;
			probe.capture.tag = tag;
			probe.capture.watch_start = watch_start;
			probe.capture.watch_end = watch_end;
			probe.capture.watch_frame = static_cast<u32>(frame_and_program >> 32);
			probe.capture.vertex_program_id = static_cast<u32>(frame_and_program);
			probe.capture.vertex_count = static_cast<u32>(draw_info);
			probe.capture.attribute_mask = static_cast<u16>(draw_info >> 32);
			probe.capture.stride = static_cast<u8>(draw_info >> 48);
			std::memcpy(probe.local_store.data(), local_store, probe.local_store.size());
			std::memcpy(probe.registers.data(), registers, probe.registers.size());
			probe.capture_phase.store(capture_ready, std::memory_order_release);
			return;
		}
	}

	void report(u64 frame_id)
	{
		if (!enabled())
			return;

		if (probe_version() == temporal_rsx_probe_version)
		{
			report_temporal(frame_id);
			return;
		}
		if (probe_version() == motion_qualification_probe_version || probe_version() == motion_raster_probe_version)
			return;

		static bool announced = false;
		static bool result_announced = false;
		static u64 last_report_frame = 0;
		auto& probe = state();

		if (!announced)
		{
			character_vertex_log.notice(
				"Character vertex probe v2 enabled: target attribute mask=0x%04x; the SPU hot path self-disables after one verified overlap.",
				character_attribute_mask);
			announced = true;
		}

		if (probe.capture_phase.load(std::memory_order_acquire) == capture_ready)
		{
			write_capture_files(probe);
			probe.capture_phase.store(capture_written, std::memory_order_release);
		}

		if (probe.capture_phase.load(std::memory_order_acquire) == capture_written && !result_announced)
		{
			const auto& capture = probe.capture;
			character_vertex_log.notice(
				"Character vertex one-shot captured: spu=0x%08x lv2=0x%08x reported_pc=0x%05x block_hash=0x%016llx "
				"cmd=0x%02x tag=%u put=[0x%08x,+0x%x] lsa=0x%05x watched=[0x%08x,0x%08x) "
				"watch_frame=%u vp=%u vertices=%u mask=0x%04x stride=%u; probe is now inactive.",
				capture.spu_id,
				capture.lv2_id,
				capture.pc,
				capture.block_hash,
				capture.command,
				capture.tag,
				capture.address,
				capture.size,
				capture.lsa,
				capture.watch_start,
				capture.watch_end,
				capture.watch_frame,
				capture.vertex_program_id,
				capture.vertex_count,
				capture.attribute_mask,
				capture.stride);
			result_announced = true;
		}

		if (!last_report_frame || frame_id >= last_report_frame + 300)
		{
			character_vertex_log.notice(
				"Character vertex probe v2 status: frame=%llu phase=%u watched=%llu SPU_puts=%llu bloom_matches=%llu overlaps=%llu.",
				frame_id,
				probe.capture_phase.load(std::memory_order_relaxed),
				probe.watched_range_count.load(std::memory_order_relaxed),
				probe.put_count.load(std::memory_order_relaxed),
				probe.bloom_match_count.load(std::memory_order_relaxed),
				probe.overlap_count.load(std::memory_order_relaxed));
			last_report_frame = frame_id;
		}
	}
}
