#pragma once

#include "util/types.hpp"

#include <array>
#include <vector>

namespace rsx::character_vertex_probe
{
	struct vertex_attribute_desc
	{
		u8 index = 0;
		u8 type = 0;
		u8 component_count = 0;
		u8 byte_size = 0;
		u16 offset = 0;
		u16 frequency = 0;
		bool modulo = false;
	};

	struct rsx_draw_desc
	{
		u64 frame_id = 0;
		u32 vertex_program_id = 0;
		u32 fragment_program_id = 0;
		u32 vertex_draw_count = 0;
		u32 stream_vertex_count = 0;
		u32 first_vertex = 0;
		u32 stream_address = 0;
		u32 stream_size = 0;
		u32 index_address = 0;
		u32 index_count = 0;
		u16 attribute_mask = 0;
		u8 stride = 0;
		u8 primitive = 0;
		u8 command = 0;
		u8 index_type = 0xff;
		u8 attribute_count = 0;
		bool indexed_constants = false;
		bool restart_index_enabled = false;
		u32 restart_index = umax;
		u32 task_sequence = 0;
		u32 task_format = 0;
		u32 task_packed_count = 0;
		u32 task_source_ea = 0;
		u32 task_output_relative_offset = 0;
		u32 task_overlap_bytes = 0;
		std::array<vertex_attribute_desc, 16> attributes{};
		bool task_identity_valid = false;
	};

	// Triangle-list vertex consumed by the renderer-specific motion-vector
	// rasterizer. Both clips use the same final viewport convention as DLSS.
	struct motion_raster_vertex
	{
		std::array<f32, 4> current_clip{};
		std::array<f32, 4> previous_clip{};
	};
	static_assert(sizeof(motion_raster_vertex) == sizeof(f32) * 8);

	struct motion_raster_frame
	{
		u64 frame_id = 0;
		u32 render_width = 0;
		u32 render_height = 0;
		u32 matched_meshes = 0;
		u32 rejected_meshes = 0;
		std::vector<motion_raster_vertex> vertices;
	};

	// Renderer-neutral frame packet used by the V5 qualification probe. The
	// matrices have already been converted to the final normalized viewport
	// convention consumed by the DLSS camera-motion reconstruction path.
	struct motion_frame_desc
	{
		u64 frame_id = 0;
		u32 render_width = 0;
		u32 render_height = 0;
		std::array<f32, 16> current_view_projection{};
		std::array<f32, 16> previous_view_projection{};
		f32 camera_confidence = 0.f;
		bool has_camera_matrices = false;
		bool reset_accumulation = true;
		bool camera_cut_detected = false;
	};

	// All character diagnostics are opt-in. V3 samples only selected RSX draws;
	// normal RPCS3 runs take only the single cached enabled() branch.
	bool enabled();

	// True only for the original V2 SPU-writer capture. V3 deliberately keeps
	// the SPU hot path completely out of its temporal RSX experiment.
	bool spu_writer_enabled();
	bool rsx_draw_enabled();
	bool task_identity_required();

	// Capture one draw that consumes a likely character vertex stream. V3 keeps
	// only bounded, sampled data in memory and writes it once after the capture
	// window, avoiding per-draw file I/O and full-buffer copies.
	void observe_rsx_draw(
		const rsx_draw_desc& draw,
		const void* stream_data,
		const void* index_data,
		u32 index_size,
		const void* transform_constants,
		const u16* constant_ids,
		u32 constant_id_count);

	// Finalize one V5 frame after the temporal camera selector has produced its
	// current/previous matrices. V5 pairs sampled final vertices by mesh identity
	// and nearest sampled world-space distance, then reports total, camera-only
	// and object-residual motion in DLSS pixels.
	void finalize_motion_frame(const motion_frame_desc& frame);

	// Moves the completed packet for frame_id to the Vulkan consumer. Returns
	// false when the V6 raster probe is disabled or no qualified geometry exists.
	bool take_motion_raster_frame(u64 frame_id, motion_raster_frame& result);

	// Register a guest-memory range that RSX consumed as a likely pre-skinned
	// character position stream. The range remains watched so that a later SPU
	// update of a reused/double-buffered stream can be attributed to its writer.
	void watch_vertex_range(
		u64 frame_id,
		u32 vertex_program_id,
		u32 vertex_count,
		u16 attribute_mask,
		u8 stride,
		u32 address,
		u32 size);

	// Observe a single SPU MFC PUT. A small page bloom filter rejects unrelated
	// writes before any watched ranges are scanned. The first verified overlap
	// captures a live SPU local-store/register snapshot and permanently disables
	// the hot probe path for the rest of the process.
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
		const void* registers);

	// Called from the RSX thread. Publishes the one-shot result and writes the
	// captured LS/register files outside the hot SPU path.
	void report(u64 frame_id);
}
