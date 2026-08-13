#pragma once

#include "bilinear_pass.hpp"
#include "Emu/RSX/Core/CharacterVertexProbe.h"

#include <memory>
#include <vector>

struct NVSDK_NGX_Handle;
struct NVSDK_NGX_Parameter;

namespace vk
{
	class dlss_optical_flow_state;
	class camera_motion_vector_pass;
	class depth_history_pass;
	class geometry_motion_vector_pass;

	namespace dlss
	{
		bool append_required_instance_extensions(std::vector<const char*>& extensions);
		bool append_required_device_extensions(VkInstance instance, VkPhysicalDevice physical_device, std::vector<const char*>& extensions);
	}

	class dlss_upscale_pass final : public upscaler
	{
		std::unique_ptr<vk::viewable_image> m_output;
		std::unique_ptr<vk::viewable_image> m_dummy_depth;
		std::unique_ptr<vk::viewable_image> m_dummy_motion_vectors;
		std::unique_ptr<vk::viewable_image> m_camera_motion_vectors;
		std::unique_ptr<vk::viewable_image> m_geometry_motion_vectors;
		std::unique_ptr<vk::viewable_image> m_previous_depth;
		std::unique_ptr<dlss_optical_flow_state> m_optical_flow;
		std::unique_ptr<camera_motion_vector_pass> m_camera_motion_pass;
		std::unique_ptr<depth_history_pass> m_depth_history_pass;
		std::unique_ptr<geometry_motion_vector_pass> m_geometry_motion_pass;
		rsx::character_vertex_probe::motion_raster_frame m_geometry_motion_frame;
		vk::bilinear_upscale_pass m_fallback;

		NVSDK_NGX_Parameter* m_parameters = nullptr;
		NVSDK_NGX_Handle* m_feature = nullptr;

		size2u m_input_size{};
		size2u m_output_size{};
		VkFormat m_input_format = VK_FORMAT_UNDEFINED;
		bool m_feature_depth_inverted = false;
		bool m_ngx_initialized = false;
		bool m_dlss_available = false;
		bool m_permanently_failed = false;
		bool m_synthetic_inputs_reported = false;
		bool m_real_inputs_reported = false;
		bool m_camera_motion_reported = false;
		bool m_geometry_motion_reported = false;
		bool m_previous_depth_valid = false;
		u64 m_previous_depth_frame_id = umax;

		void dispose_images();
		void release_feature();
		void shutdown_ngx();
		bool initialize_ngx();
		bool initialize_images(const vk::command_buffer& cmd, vk::viewable_image* src, const size2u& input_size, const size2u& output_size);
		bool initialize_feature(const vk::command_buffer& cmd, const size2u& input_size, const size2u& output_size, bool depth_inverted);

		vk::viewable_image* run_fallback(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			VkImage present_surface,
			VkImageLayout present_surface_layout,
			const VkImageBlit& request,
			rsx::flags32_t mode,
			const upscaler_frame_data* frame_data);

	public:
		dlss_upscale_pass();
		~dlss_upscale_pass() override;

		bool prepare_optical_flow(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			const size2u& input_size,
			u32 frame_slot,
			u32 frame_slot_count);
		VkSemaphore get_optical_flow_ready_semaphore() const;
		VkSemaphore submit_optical_flow();
		vk::viewable_image* get_optical_flow_motion_vectors() const;

		vk::viewable_image* scale_output(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			VkImage present_surface,
			VkImageLayout present_surface_layout,
			const VkImageBlit& request,
			rsx::flags32_t mode,
			const upscaler_frame_data* frame_data) override;
	};
}
