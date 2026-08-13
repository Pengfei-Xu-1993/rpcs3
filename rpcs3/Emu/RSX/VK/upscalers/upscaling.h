#pragma once

#include "../vkutils/commands.h"
#include "../vkutils/image.h"

#include <array>
#include <chrono>

namespace vk
{
	namespace upscaling_flags_
	{
		enum upscaling_flags
		{
			UPSCALE_DEFAULT_VIEW = (1 << 0),
			UPSCALE_LEFT_VIEW    = (1 << 0),
			UPSCALE_RIGHT_VIEW   = (1 << 1),
			UPSCALE_AND_COMMIT   = (1 << 2)
		};
	}

	using namespace upscaling_flags_;

	enum class temporal_resource_source : u8
	{
		unavailable,
		guest_render_target,
		guest_metadata,
		reconstructed_camera,
		reconstructed_geometry,
		optical_flow,
		heuristic,
		synthetic_fallback,
	};

	enum class temporal_history_reset_reason : u32
	{
		none                  = 0,
		cold_start            = 1u << 0,
		frame_discontinuity   = 1u << 1,
		render_size_changed   = 1u << 2,
		display_size_changed  = 1u << 3,
		color_format_changed  = 1u << 4,
		presentation_stall    = 1u << 5,
		camera_cut            = 1u << 6,
		guest_requested       = 1u << 7,
		backend_changed       = 1u << 8,
		swapchain_recreated   = 1u << 9,
	};

	constexpr u32 temporal_reset_bit(temporal_history_reset_reason reason)
	{
		return static_cast<u32>(reason);
	}

	struct temporal_image_input
	{
		vk::viewable_image* image = nullptr;
		temporal_resource_source source = temporal_resource_source::unavailable;
		size2u active_size{};
		u64 frame_id = 0;
		float confidence = 0.f;

		explicit operator bool() const
		{
			return image != nullptr;
		}
	};

	// Vendor-neutral inputs shared by temporal upscalers, frame generation and
	// latency/pacing integrations. Keep this packet independent of NGX,
	// Streamline and FidelityFX so every backend consumes the same captured
	// RSX frame and the same history-reset decisions.
	struct temporal_frame_data
	{
		temporal_image_input color;
		temporal_image_input depth;
		temporal_image_input motion_vectors;
		temporal_image_input hudless_color;
		temporal_image_input ui_color;
		temporal_image_input reactive_mask;
		temporal_image_input transparency_and_composition_mask;
		temporal_image_input exposure;
		temporal_image_input distortion_field;

		u64 frame_id = 0;
		size2u render_size{};
		size2u display_size{};
		float frame_time_delta_ms = 1000.f / 60.f;
		float pre_exposure = 1.f;

		float jitter_x = 0.f;
		float jitter_y = 0.f;
		float motion_vector_scale_x = 1.f;
		float motion_vector_scale_y = 1.f;

		std::array<float, 16> current_view_projection{};
		std::array<float, 16> previous_view_projection{};
		temporal_resource_source camera_matrix_source = temporal_resource_source::unavailable;
		float camera_matrix_confidence = 0.f;
		float camera_near = 0.f;
		float camera_far = 0.f;
		float camera_fov_y = 0.f;
		float view_space_to_meters = 1.f;

		u32 history_reset_reasons = temporal_reset_bit(temporal_history_reset_reason::cold_start);
		bool reset_accumulation = true;
		bool depth_inverted = false;
		bool has_camera_matrices = false;
		bool camera_cut_detected = false;
		bool camera_motion_included = false;
		bool motion_vectors_jittered = false;
		bool motion_vectors_dilated = false;
		bool hdr = false;

		void request_history_reset(temporal_history_reset_reason reason)
		{
			history_reset_reasons |= temporal_reset_bit(reason);
			reset_accumulation = true;
		}
	};

	using upscaler_frame_data = temporal_frame_data;

	class temporal_frame_tracker
	{
		using clock = std::chrono::steady_clock;

		bool m_has_history = false;
		u64 m_last_frame_id = 0;
		size2u m_last_render_size{};
		size2u m_last_display_size{};
		VkFormat m_last_color_format = VK_FORMAT_UNDEFINED;
		clock::time_point m_last_frame_time{};
		u32 m_pending_reset_reasons = temporal_reset_bit(temporal_history_reset_reason::cold_start);

		static bool sizes_match(const size2u& lhs, const size2u& rhs)
		{
			return lhs.width == rhs.width && lhs.height == rhs.height;
		}

	public:
		void request_reset(temporal_history_reset_reason reason)
		{
			m_pending_reset_reasons |= temporal_reset_bit(reason);
		}

		void begin_frame(
			temporal_frame_data& frame,
			u64 frame_id,
			const size2u& render_size,
			const size2u& display_size,
			VkFormat color_format)
		{
			const auto now = clock::now();
			frame = {};
			frame.frame_id = frame_id;
			frame.render_size = render_size;
			frame.display_size = display_size;
			frame.history_reset_reasons = m_pending_reset_reasons;
			m_pending_reset_reasons = 0;

			if (m_has_history)
			{
				if (frame_id != m_last_frame_id + 1)
					frame.request_history_reset(temporal_history_reset_reason::frame_discontinuity);

				if (!sizes_match(render_size, m_last_render_size))
					frame.request_history_reset(temporal_history_reset_reason::render_size_changed);

				if (!sizes_match(display_size, m_last_display_size))
					frame.request_history_reset(temporal_history_reset_reason::display_size_changed);

				if (color_format != m_last_color_format)
					frame.request_history_reset(temporal_history_reset_reason::color_format_changed);

				frame.frame_time_delta_ms = std::chrono::duration<float, std::milli>(now - m_last_frame_time).count();
				if (frame.frame_time_delta_ms > 250.f)
					frame.request_history_reset(temporal_history_reset_reason::presentation_stall);
			}
			else
			{
				frame.request_history_reset(temporal_history_reset_reason::cold_start);
			}

			frame.reset_accumulation = frame.history_reset_reasons != 0;
			m_has_history = true;
			m_last_frame_id = frame_id;
			m_last_render_size = render_size;
			m_last_display_size = display_size;
			m_last_color_format = color_format;
			m_last_frame_time = now;
		}
	};

	struct upscaler
	{
		virtual ~upscaler() {}

		virtual vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,          // CB
			vk::viewable_image* src,                // Source input
			VkImage present_surface,                // Present target. May be VK_NULL_HANDLE for some passes
			VkImageLayout present_surface_layout,   // Present surface layout, or VK_IMAGE_LAYOUT_UNDEFINED if no present target is provided
			const VkImageBlit& request,             // Scaling request information
			rsx::flags32_t mode,                    // Mode
			const upscaler_frame_data* frame_data = nullptr
		) = 0;
	};
}
