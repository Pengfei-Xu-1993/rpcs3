#pragma once

#include "upscaling.h"

#include <array>
#include <span>
#include <unordered_map>
#include <vector>

namespace vk
{
	// Read-only RSX diagnostics used to discover camera data that was not
	// explicitly authored as a temporal-upscaling resource by the guest.
	// The capture path never modifies guest registers or shader constants.
	class temporal_camera_capture
	{
		struct matrix_candidate
		{
			std::array<float, 16> matrix{};
			std::array<float, 3> camera_position{};
			// RSX viewport transform in normalized clip space:
			// scale xyz followed by offset xyz. Keeping it with the winning draw
			// lets the captured guest matrix reconstruct the depth buffer exactly.
			std::array<float, 6> viewport_transform{1.f, 1.f, 1.f, 0.f, 0.f, 0.f};
			u64 hash = 0;
			u64 total_vertices = 0;
			u32 occurrence_count = 0;
			u32 depth_test_count = 0;
			u32 depth_write_count = 0;
			u32 distinct_program_count = 0;
			u32 last_program_id = umax;
			u32 source_program_id = 0;
			u32 source_vertex_count = 0;
			u16 constant_index = 0;
			bool transposed = false;
			float focal_x = 0.f;
			float focal_y = 0.f;
			float shape_score = 0.f;
			float rank_score = 0.f;
		};

		bool m_enabled = false;
		bool m_verbose = false;
		bool m_announced = false;
		bool m_has_observed_frame = false;
		bool m_has_selected_matrix = false;
		u64 m_observed_frame_id = 0;
		u64 m_finalized_frame_count = 0;
		u32 m_draw_count = 0;
		u32 m_fast_path_draw_count = 0;
		u32 m_matrix_window_count = 0;
		u32 m_valid_matrix_count = 0;
		u32 m_stable_frame_count = 0;
		u32 m_missing_frame_count = 0;
		bool m_fast_path_announced = false;
		std::array<float, 16> m_selected_matrix{};
		std::array<float, 16> m_selected_final_view_projection{};
		std::array<float, 3> m_selected_camera_position{};
		float m_selected_focal_x = 0.f;
		float m_selected_focal_y = 0.f;
		u16 m_selected_constant_index = 0;
		bool m_selected_transposed = false;
		std::vector<matrix_candidate> m_candidates;
		std::unordered_map<u64, u32> m_candidate_lookup;

		void begin_observed_frame(u64 frame_id);
		void clear_observed_frame();
		void add_candidate(
			const std::array<float, 16>& matrix,
			u16 constant_index,
			bool transposed,
			u32 program_id,
			u32 vertex_count,
			bool depth_test,
			bool depth_write,
			const std::array<float, 6>& viewport_transform);

	public:
		temporal_camera_capture();

		bool enabled() const
		{
			return m_enabled;
		}

		void observe_draw(
			u64 frame_id,
			const u32 (*transform_constants)[4],
			u32 transform_constant_count,
			std::span<const u16> referenced_constants,
			bool has_indexed_constants,
			u32 program_id,
			u32 vertex_count,
			bool depth_test,
			bool depth_write,
			const std::array<float, 6>& viewport_transform);

		void finalize_frame(temporal_frame_data& frame);
	};
}
