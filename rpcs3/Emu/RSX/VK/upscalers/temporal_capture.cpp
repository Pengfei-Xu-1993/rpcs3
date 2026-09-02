#include "stdafx.h"
#include "temporal_capture.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace
{
	constexpr u32 max_rsx_transform_constants = 468;
	constexpr u32 max_frame_candidates = 1024;
	constexpr u32 fast_tracking_lock_frames = 120;
	constexpr u64 fast_tracking_rescan_period = 120;

	bool environment_switch_enabled(const char* name)
	{
		const char* value = ::getenv(name);
		if (!value || !value[0])
			return false;

		const std::string_view text{value};
		return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF";
	}

	float vector_length(float x, float y, float z)
	{
		return std::sqrt((x * x) + (y * y) + (z * z));
	}

	float normalized_dot(
		float ax, float ay, float az,
		float bx, float by, float bz,
		float a_length, float b_length)
	{
		return ((ax * bx) + (ay * by) + (az * bz)) / (a_length * b_length);
	}

	struct matrix_shape
	{
		std::array<float, 3> camera_position{};
		float focal_x = 0.f;
		float focal_y = 0.f;
		float score = 0.f;
	};

	bool evaluate_view_projection_shape(const std::array<float, 16>& matrix, matrix_shape& result)
	{
		for (const float value : matrix)
		{
			if (!std::isfinite(value) || std::abs(value) > 1.e8f)
				return false;
		}

		const float n0 = vector_length(matrix[0], matrix[1], matrix[2]);
		const float n1 = vector_length(matrix[4], matrix[5], matrix[6]);
		const float n2 = vector_length(matrix[8], matrix[9], matrix[10]);
		const float n3 = vector_length(matrix[12], matrix[13], matrix[14]);
		if (n0 < 0.02f || n1 < 0.02f || n0 > 100.f || n1 > 100.f || n3 < 0.5f || n3 > 2.f)
			return false;

		const float d01 = std::abs(normalized_dot(
			matrix[0], matrix[1], matrix[2],
			matrix[4], matrix[5], matrix[6], n0, n1));
		const float d03 = std::abs(normalized_dot(
			matrix[0], matrix[1], matrix[2],
			matrix[12], matrix[13], matrix[14], n0, n3));
		const float d13 = std::abs(normalized_dot(
			matrix[4], matrix[5], matrix[6],
			matrix[12], matrix[13], matrix[14], n1, n3));
		if (d01 > 0.15f || d03 > 0.15f || d13 > 0.15f)
			return false;

		float depth_axis_error = 0.f;
		if (n2 > 1.e-5f)
		{
			const float cx = (matrix[9] * matrix[14]) - (matrix[10] * matrix[13]);
			const float cy = (matrix[10] * matrix[12]) - (matrix[8] * matrix[14]);
			const float cz = (matrix[8] * matrix[13]) - (matrix[9] * matrix[12]);
			depth_axis_error = vector_length(cx, cy, cz) / (n2 * n3);
			if (depth_axis_error > 0.2f)
				return false;
		}

		const bool identity_rotation =
			std::abs((matrix[0] / n0) - 1.f) < 1.e-3f && std::abs(matrix[1]) < 1.e-3f && std::abs(matrix[2]) < 1.e-3f &&
			std::abs(matrix[4]) < 1.e-3f && std::abs((matrix[5] / n1) - 1.f) < 1.e-3f && std::abs(matrix[6]) < 1.e-3f &&
			std::abs(matrix[12]) < 1.e-3f && std::abs(matrix[13]) < 1.e-3f;
		const bool no_translation = std::abs(matrix[3]) < 1.e-3f && std::abs(matrix[7]) < 1.e-3f && std::abs(matrix[15]) < 1.e-3f;
		if (identity_rotation && no_translation)
			return false;

		const float r00 = matrix[0] / n0;
		const float r01 = matrix[1] / n0;
		const float r02 = matrix[2] / n0;
		const float r10 = matrix[4] / n1;
		const float r11 = matrix[5] / n1;
		const float r12 = matrix[6] / n1;
		const float r20 = matrix[12] / n3;
		const float r21 = matrix[13] / n3;
		const float r22 = matrix[14] / n3;
		const float tx = matrix[3] / n0;
		const float ty = matrix[7] / n1;
		const float tz = matrix[15] / n3;

		result.camera_position = {
			-((r00 * tx) + (r10 * ty) + (r20 * tz)),
			-((r01 * tx) + (r11 * ty) + (r21 * tz)),
			-((r02 * tx) + (r12 * ty) + (r22 * tz)),
		};
		for (const float value : result.camera_position)
		{
			if (!std::isfinite(value) || std::abs(value) > 1.e8f)
				return false;
		}

		const float scale_error = std::min(std::abs(n3 - 1.f), 1.f);
		const float orthogonality_error = std::min((d01 + d03 + d13) / 0.45f, 1.f);
		const float axis_error = std::min(depth_axis_error / 0.2f, 1.f);
		const float aspect = n1 / n0;
		const float aspect_penalty = (aspect < 0.35f || aspect > 4.f) ? 0.15f : 0.f;
		result.focal_x = n0;
		result.focal_y = n1;
		result.score = std::clamp(1.f - (0.35f * scale_error) - (0.45f * orthogonality_error) - (0.2f * axis_error) - aspect_penalty, 0.f, 1.f);
		return result.score >= 0.55f;
	}

	std::array<float, 16> transpose_matrix(const std::array<float, 16>& source)
	{
		std::array<float, 16> result{};
		for (u32 row = 0; row < 4; ++row)
		{
			for (u32 column = 0; column < 4; ++column)
				result[(row * 4) + column] = source[(column * 4) + row];
		}
		return result;
	}

	std::array<float, 16> apply_viewport_transform(
		const std::array<float, 16>& matrix,
		const std::array<float, 6>& viewport)
	{
		// Vertex programs produce guest clip coordinates first. RPCS3 then applies
		// gl_Position = gl_Position * scale_offset_mat. In column-vector notation
		// that is T * M, with the normalized viewport offsets in T's last column.
		std::array<float, 16> result{};
		for (u32 column = 0; column < 4; ++column)
		{
			result[column] = (viewport[0] * matrix[column]) + (viewport[3] * matrix[12 + column]);
			result[4 + column] = (viewport[1] * matrix[4 + column]) + (viewport[4] * matrix[12 + column]);
			result[8 + column] = (viewport[2] * matrix[8 + column]) + (viewport[5] * matrix[12 + column]);
			result[12 + column] = matrix[12 + column];
		}
		return result;
	}

	u64 hash_matrix(const std::array<float, 16>& matrix)
	{
		constexpr u64 offset_basis = 1469598103934665603ull;
		constexpr u64 prime = 1099511628211ull;
		u64 hash = offset_basis;
		for (const float value : matrix)
		{
			hash ^= std::bit_cast<u32>(value);
			hash *= prime;
		}
		return hash;
	}

	float relative_matrix_delta(const std::array<float, 16>& lhs, const std::array<float, 16>& rhs)
	{
		double squared_delta = 0.;
		double squared_reference = 0.;
		for (u32 index = 0; index < 16; ++index)
		{
			const double delta = static_cast<double>(lhs[index]) - rhs[index];
			squared_delta += delta * delta;
			squared_reference += static_cast<double>(rhs[index]) * rhs[index];
		}
		return static_cast<float>(std::sqrt(squared_delta / std::max(squared_reference, 1.e-12)));
	}

	float orientation_dot(const std::array<float, 16>& lhs, const std::array<float, 16>& rhs, u32 row)
	{
		const u32 offset = row * 4;
		const float lhs_length = vector_length(lhs[offset], lhs[offset + 1], lhs[offset + 2]);
		const float rhs_length = vector_length(rhs[offset], rhs[offset + 1], rhs[offset + 2]);
		if (lhs_length < 1.e-6f || rhs_length < 1.e-6f)
			return -1.f;
		return normalized_dot(
			lhs[offset], lhs[offset + 1], lhs[offset + 2],
			rhs[offset], rhs[offset + 1], rhs[offset + 2], lhs_length, rhs_length);
	}
}

namespace vk
{
	temporal_camera_capture::temporal_camera_capture()
		: m_enabled(environment_switch_enabled("RPCS3_TEMPORAL_CAPTURE"))
		, m_verbose(environment_switch_enabled("RPCS3_TEMPORAL_CAPTURE_VERBOSE"))
	{
		m_candidates.reserve(128);
		m_candidate_lookup.reserve(256);
	}

	void temporal_camera_capture::clear_observed_frame()
	{
		m_draw_count = 0;
		m_fast_path_draw_count = 0;
		m_matrix_window_count = 0;
		m_valid_matrix_count = 0;
		m_candidates.clear();
		m_candidate_lookup.clear();
	}

	void temporal_camera_capture::begin_observed_frame(u64 frame_id)
	{
		if (m_has_observed_frame && m_observed_frame_id == frame_id)
			return;

		clear_observed_frame();
		m_has_observed_frame = true;
		m_observed_frame_id = frame_id;
	}

	void temporal_camera_capture::add_candidate(
		const std::array<float, 16>& matrix,
		u16 constant_index,
		bool transposed,
		u32 program_id,
		u32 vertex_count,
		bool depth_test,
		bool depth_write,
		const std::array<float, 6>& viewport_transform)
	{
		matrix_shape shape{};
		if (!evaluate_view_projection_shape(matrix, shape))
			return;

		m_valid_matrix_count++;
		u64 hash = hash_matrix(matrix);
		auto found = m_candidate_lookup.find(hash);
		while (found != m_candidate_lookup.end() && m_candidates[found->second].matrix != matrix)
		{
			hash = (hash * 1099511628211ull) ^ 0x9e3779b97f4a7c15ull;
			found = m_candidate_lookup.find(hash);
		}

		matrix_candidate* candidate = nullptr;
		if (found == m_candidate_lookup.end())
		{
			if (m_candidates.size() >= max_frame_candidates)
				return;

			const u32 index = ::size32(m_candidates);
			m_candidate_lookup.emplace(hash, index);
			m_candidates.emplace_back();
			candidate = &m_candidates.back();
			candidate->matrix = matrix;
			candidate->camera_position = shape.camera_position;
			candidate->viewport_transform = viewport_transform;
			candidate->hash = hash;
			candidate->constant_index = constant_index;
			candidate->transposed = transposed;
			candidate->source_program_id = program_id;
			candidate->focal_x = shape.focal_x;
			candidate->focal_y = shape.focal_y;
			candidate->shape_score = shape.score;
		}
		else
		{
			candidate = &m_candidates[found->second];
		}

		candidate->occurrence_count++;
		candidate->total_vertices += vertex_count;
		candidate->depth_test_count += depth_test;
		candidate->depth_write_count += depth_write;
		if (candidate->last_program_id != program_id)
		{
			candidate->distinct_program_count++;
			candidate->last_program_id = program_id;
		}
		if (vertex_count > candidate->source_vertex_count)
		{
			candidate->source_vertex_count = vertex_count;
			candidate->source_program_id = program_id;
			candidate->constant_index = constant_index;
			candidate->transposed = transposed;
			candidate->viewport_transform = viewport_transform;
		}
	}

	void temporal_camera_capture::observe_draw(
		u64 frame_id,
		const u32 (*transform_constants)[4],
		u32 transform_constant_count,
		std::span<const u16> referenced_constants,
		bool has_indexed_constants,
		u32 program_id,
		u32 vertex_count,
		bool depth_test,
		bool depth_write,
		const std::array<float, 6>& viewport_transform)
	{
		if (!m_enabled || !transform_constants)
			return;

		if (!m_announced)
		{
			m_announced = true;
			rsx_log.notice("Temporal camera capture enabled (read-only RSX vertex-constant diagnostics).\n"
				"Set RPCS3_TEMPORAL_CAPTURE_VERBOSE=1 for per-frame candidate logs.");
		}

		begin_observed_frame(frame_id);
		m_draw_count++;
		const u32 constant_count = std::min(transform_constant_count, max_rsx_transform_constants);
		if (constant_count < 4)
			return;

		const auto observe_window = [&](u32 first)
		{
			if (first + 3 >= constant_count)
				return;

			std::array<float, 16> matrix{};
			for (u32 row = 0; row < 4; ++row)
			{
				for (u32 column = 0; column < 4; ++column)
					matrix[(row * 4) + column] = std::bit_cast<float>(transform_constants[first + row][column]);
			}

			m_matrix_window_count++;
			add_candidate(matrix, static_cast<u16>(first), false, program_id, vertex_count, depth_test, depth_write, viewport_transform);
			add_candidate(transpose_matrix(matrix), static_cast<u16>(first), true, program_id, vertex_count, depth_test, depth_write, viewport_transform);
		};

		// Once a matrix register has survived two seconds of frame-to-frame
		// qualification, track that four-register window directly. This removes
		// the 468-register search loop from almost every draw while retaining a
		// periodic full scan and automatically unlocking after missing frames.
		const bool periodic_rescan = (frame_id % fast_tracking_rescan_period) == 0;
		if (m_has_selected_matrix && m_stable_frame_count >= fast_tracking_lock_frames && !periodic_rescan &&
			m_selected_constant_index + 3 < constant_count)
		{
			bool selected_window_is_referenced = has_indexed_constants;
			if (!selected_window_is_referenced)
			{
				selected_window_is_referenced = true;
				for (u32 index = m_selected_constant_index; index <= m_selected_constant_index + 3; ++index)
				{
					if (std::find(referenced_constants.cbegin(), referenced_constants.cend(), static_cast<u16>(index)) == referenced_constants.cend())
					{
						selected_window_is_referenced = false;
						break;
					}
				}
			}

			if (selected_window_is_referenced)
			{
				observe_window(m_selected_constant_index);
				m_fast_path_draw_count++;
				if (!m_fast_path_announced)
				{
					m_fast_path_announced = true;
					rsx_log.success("Temporal camera capture locked the stable matrix register; low-overhead tracking is active.");
				}
				return;
			}
		}

		std::array<bool, max_rsx_transform_constants> referenced{};
		// Indexed vertex programs expose the whole 468-register file. Scanning
		// every overlapping matrix on every draw is needlessly expensive for a
		// diagnostic path, so take dense samples at the start of a frame and at
		// regular intervals. Statically referenced constants are still observed
		// on every draw.
		const bool take_dense_indexed_sample = has_indexed_constants && (m_draw_count <= 8 || (m_draw_count % 64) == 0);
		if (take_dense_indexed_sample)
		{
			std::fill_n(referenced.begin(), constant_count, true);
		}
		for (const u16 index : referenced_constants)
		{
			if (index < constant_count)
				referenced[index] = true;
		}

		for (u32 first = 0; first + 3 < constant_count; ++first)
		{
			if (!referenced[first] || !referenced[first + 1] || !referenced[first + 2] || !referenced[first + 3])
				continue;

			observe_window(first);
		}
	}

	void temporal_camera_capture::finalize_frame(temporal_frame_data& frame)
	{
		if (!m_enabled)
			return;

		m_finalized_frame_count++;
		const bool matching_frame = m_has_observed_frame && m_observed_frame_id == frame.frame_id;
		matrix_candidate* selected = nullptr;
		if (matching_frame)
		{
			for (auto& candidate : m_candidates)
			{
				const float occurrence_score = std::log2(static_cast<float>(candidate.occurrence_count) + 1.f) * 8.f;
				const float vertex_score = std::log2(static_cast<float>(candidate.total_vertices) + 1.f) * 1.5f;
				const float depth_test_ratio = static_cast<float>(candidate.depth_test_count) / candidate.occurrence_count;
				const float depth_write_ratio = static_cast<float>(candidate.depth_write_count) / candidate.occurrence_count;
				candidate.rank_score = (candidate.shape_score * 70.f) + occurrence_score + vertex_score +
					(depth_test_ratio * 6.f) + (depth_write_ratio * 4.f) +
					(std::min(candidate.distinct_program_count, 4u) * 2.f);

				if (m_has_selected_matrix)
				{
					const float delta = relative_matrix_delta(candidate.matrix, m_selected_matrix);
					if (delta < 0.005f)
						candidate.rank_score += 30.f;
					else if (delta < 0.02f)
						candidate.rank_score += 25.f;
					else if (delta < 0.1f)
						candidate.rank_score += 15.f;
					else if (delta < 0.35f)
						candidate.rank_score += 5.f;
					else if (delta > 1.f)
						candidate.rank_score -= 15.f;

					if (candidate.constant_index == m_selected_constant_index && candidate.transposed == m_selected_transposed)
						candidate.rank_score += 6.f;
				}

				if (!selected || candidate.rank_score > selected->rank_score)
					selected = &candidate;
			}
		}

		const bool repeated_across_scene = selected &&
			(selected->occurrence_count >= 3 || (selected->occurrence_count >= 2 && selected->distinct_program_count >= 2));
		const float confidence = selected
			? std::clamp(((selected->rank_score - 65.f) / 70.f) * selected->shape_score, 0.f, 1.f)
			: 0.f;
		const bool qualified = selected && repeated_across_scene && selected->shape_score >= 0.65f && confidence >= 0.55f;

		bool camera_cut = false;
		if (qualified)
		{
			const auto final_view_projection = apply_viewport_transform(selected->matrix, selected->viewport_transform);
			if (m_has_selected_matrix)
			{
				const float delta = relative_matrix_delta(selected->matrix, m_selected_matrix);
				const float right_dot = orientation_dot(selected->matrix, m_selected_matrix, 0);
				const float up_dot = orientation_dot(selected->matrix, m_selected_matrix, 1);
				const float forward_dot = orientation_dot(selected->matrix, m_selected_matrix, 3);
				const float focal_x_ratio = selected->focal_x / std::max(m_selected_focal_x, 1.e-6f);
				const float focal_y_ratio = selected->focal_y / std::max(m_selected_focal_y, 1.e-6f);
				camera_cut = delta > 2.f || std::min({right_dot, up_dot, forward_dot}) < 0.5f ||
					focal_x_ratio < 0.625f || focal_x_ratio > 1.6f || focal_y_ratio < 0.625f || focal_y_ratio > 1.6f;

				if (camera_cut)
				{
					m_stable_frame_count = 1;
				}
				else
				{
					m_stable_frame_count++;
					frame.previous_view_projection = m_selected_final_view_projection;
					frame.current_view_projection = final_view_projection;
					frame.has_camera_matrices = m_stable_frame_count >= 3;
					frame.camera_matrix_source = temporal_resource_source::reconstructed_camera;
					frame.camera_matrix_confidence = confidence;
				}
			}
			else
			{
				m_stable_frame_count = 1;
			}

			frame.camera_cut_detected = camera_cut;
			if (camera_cut)
				frame.request_history_reset(temporal_history_reset_reason::camera_cut);
			m_selected_matrix = selected->matrix;
			m_selected_final_view_projection = final_view_projection;
			m_selected_camera_position = selected->camera_position;
			m_selected_focal_x = selected->focal_x;
			m_selected_focal_y = selected->focal_y;
			m_selected_constant_index = selected->constant_index;
			m_selected_transposed = selected->transposed;
			m_has_selected_matrix = true;
			m_missing_frame_count = 0;
		}
		else if (++m_missing_frame_count >= 3)
		{
			m_has_selected_matrix = false;
			m_stable_frame_count = 0;
		}

		const bool should_log = m_verbose || m_finalized_frame_count <= 8 || (m_finalized_frame_count % 60) == 0 || camera_cut;
		if (should_log)
		{
			if (selected)
			{
				rsx_log.notice(
					"Temporal camera frame %llu: draws=%u, fast=%u, windows=%u, valid=%u, unique=%u; "
					"best=vc[%u..%u]%s vp=%u uses=%u programs=%u vertices=%llu shape=%.3f rank=%.1f confidence=%.3f stable=%u%s "
					"camera=(%.3f, %.3f, %.3f) focal=(%.3f, %.3f)",
					frame.frame_id, m_draw_count, m_fast_path_draw_count, m_matrix_window_count, m_valid_matrix_count, ::size32(m_candidates),
					selected->constant_index, selected->constant_index + 3, selected->transposed ? " transpose" : "",
					selected->source_program_id, selected->occurrence_count, selected->distinct_program_count,
					selected->total_vertices, selected->shape_score, selected->rank_score, confidence, m_stable_frame_count,
					qualified ? (camera_cut ? " CUT" : " QUALIFIED") : " diagnostic-only",
					selected->camera_position[0], selected->camera_position[1], selected->camera_position[2],
					selected->focal_x, selected->focal_y);
			}
			else
			{
				rsx_log.notice("Temporal camera frame %llu: no view-projection candidate (draws=%u, fast=%u, windows=%u, valid=%u).",
					frame.frame_id, matching_frame ? m_draw_count : 0, matching_frame ? m_fast_path_draw_count : 0,
					matching_frame ? m_matrix_window_count : 0,
					matching_frame ? m_valid_matrix_count : 0);
			}
		}

		if (matching_frame)
		{
			clear_observed_frame();
			m_has_observed_frame = false;
		}
	}
}
