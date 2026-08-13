#include "stdafx.h"

#include "../vkutils/barriers.h"
#include "../vkutils/sampler.h"
#include "../VKCompute.h"
#include "../VKHelpers.h"
#include "../VKFramebuffer.h"
#include "../VKOverlays.h"
#include "../VKQueryPool.h"
#include "../VKRenderPass.h"
#include "../VKResourceManager.h"

#include "dlss_pass.h"

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <string>
#include <string_view>

namespace
{
	constexpr char g_dlss_project_id[] = "6b7d4bf8-4b29-4f4f-9f65-f12baf6e8150";
	constexpr char g_dlss_engine_version[] = "RPCS3-DLSS-Prototype-1";
	constexpr wchar_t g_dlss_data_path[] = L".";

	NVSDK_NGX_FeatureDiscoveryInfo make_discovery_info()
	{
		NVSDK_NGX_FeatureDiscoveryInfo info{};
		info.SDKVersion = NVSDK_NGX_Version_API;
		info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
		info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
		info.Identifier.v.ProjectDesc.ProjectId = g_dlss_project_id;
		info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
		info.Identifier.v.ProjectDesc.EngineVersion = g_dlss_engine_version;
		info.ApplicationDataPath = g_dlss_data_path;
		return info;
	}

	bool extension_is_requested(const std::vector<const char*>& requested, std::string_view name)
	{
		return std::any_of(requested.cbegin(), requested.cend(), [&](const char* value)
		{
			return value && name == value;
		});
	}

	bool extension_is_supported(const std::vector<VkExtensionProperties>& supported, std::string_view name)
	{
		return std::any_of(supported.cbegin(), supported.cend(), [&](const VkExtensionProperties& value)
		{
			return name == value.extensionName;
		});
	}

	bool append_extensions(
		std::vector<const char*>& requested,
		std::vector<std::string>& storage,
		const std::vector<std::string>& required,
		const std::vector<VkExtensionProperties>& supported,
		std::string_view extension_class)
	{
		for (const std::string& name : required)
		{
			if (!extension_is_supported(supported, name))
			{
				rsx_log.error("NVIDIA DLSS requires unsupported Vulkan %s extension '%s'.", extension_class, name);
				return false;
			}
		}

		storage.clear();
		storage.reserve(required.size());
		for (const std::string& name : required)
		{
			if (!extension_is_requested(requested, name))
			{
				storage.push_back(name);
			}
		}

		for (const std::string& name : storage)
		{
			requested.push_back(name.c_str());
		}

		return true;
	}

	NVSDK_NGX_PerfQuality_Value select_quality_mode(const size2u& input_size, const size2u& output_size)
	{
		const float ratio_x = static_cast<float>(input_size.width) / static_cast<float>(output_size.width);
		const float ratio_y = static_cast<float>(input_size.height) / static_cast<float>(output_size.height);
		const float ratio = std::min(ratio_x, ratio_y);

		// Choose the closest DLSS Super Resolution render ratio. Equal-sized
		// input/output requests are rejected below, so DLAA is intentionally not
		// selected by this pass.
		if (ratio >= 0.623f)
			return NVSDK_NGX_PerfQuality_Value_MaxQuality;
		if (ratio >= 0.54f)
			return NVSDK_NGX_PerfQuality_Value_Balanced;
		if (ratio >= 0.416f)
			return NVSDK_NGX_PerfQuality_Value_MaxPerf;
		return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
	}

	bool motion_vector_format_is_supported(VkFormat format)
	{
		return format == VK_FORMAT_R16G16_SFLOAT || format == VK_FORMAT_R32G32_SFLOAT;
	}

	bool depth_format_is_supported(VkFormat format)
	{
		switch (format)
		{
		case VK_FORMAT_D16_UNORM:
		case VK_FORMAT_D24_UNORM_S8_UINT:
		case VK_FORMAT_D32_SFLOAT:
		case VK_FORMAT_D32_SFLOAT_S8_UINT:
		case VK_FORMAT_R32_SFLOAT:
			return true;
		default:
			return false;
		}
	}

	bool invert_matrix_4x4(const std::array<float, 16>& source, std::array<float, 16>& result)
	{
		// Gauss-Jordan is inexpensive here (once per presented frame), avoids a
		// per-pixel matrix inverse in the reconstruction shader, and lets a
		// singular/garbled capture fall back to optical flow without touching NGX.
		double augmented[4][8]{};
		for (u32 row = 0; row < 4; ++row)
		{
			for (u32 column = 0; column < 4; ++column)
				augmented[row][column] = source[(row * 4) + column];
			augmented[row][4 + row] = 1.;
		}

		for (u32 column = 0; column < 4; ++column)
		{
			u32 pivot_row = column;
			double pivot_size = std::abs(augmented[pivot_row][column]);
			for (u32 row = column + 1; row < 4; ++row)
			{
				const double candidate_size = std::abs(augmented[row][column]);
				if (candidate_size > pivot_size)
				{
					pivot_row = row;
					pivot_size = candidate_size;
				}
			}

			if (!std::isfinite(pivot_size) || pivot_size < 1.e-12)
				return false;

			if (pivot_row != column)
			{
				for (u32 entry = 0; entry < 8; ++entry)
					std::swap(augmented[column][entry], augmented[pivot_row][entry]);
			}

			const double inverse_pivot = 1. / augmented[column][column];
			for (u32 entry = 0; entry < 8; ++entry)
				augmented[column][entry] *= inverse_pivot;

			for (u32 row = 0; row < 4; ++row)
			{
				if (row == column)
					continue;
				const double factor = augmented[row][column];
				for (u32 entry = 0; entry < 8; ++entry)
					augmented[row][entry] -= factor * augmented[column][entry];
			}
		}

		for (u32 row = 0; row < 4; ++row)
		{
			for (u32 column = 0; column < 4; ++column)
			{
				const double value = augmented[row][4 + column];
				if (!std::isfinite(value) || std::abs(value) > 1.e12)
					return false;
				result[(row * 4) + column] = static_cast<float>(value);
			}
		}
		return true;
	}

	VkImageSubresourceRange make_subresource_range(VkImageAspectFlags aspect)
	{
		return {aspect, 0, 1, 0, 1};
	}

	void prepare_ngx_input(const vk::command_buffer& cmd, vk::viewable_image* image, VkImageAspectFlags aspect)
	{
		image->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		// RPCS3's generic layout helper targets graphics shader stages. NGX reads
		// these resources from commands recorded by the SDK, so add a conservative
		// dependency that also covers compute and any future implementation stage.
		vk::insert_image_memory_barrier(
			cmd,
			image->value,
			image->current_layout,
			image->current_layout,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_ACCESS_MEMORY_WRITE_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
			make_subresource_range(aspect));
	}

	void restore_ngx_input(const vk::command_buffer& cmd, vk::viewable_image* image, VkImageAspectFlags aspect)
	{
		// Ensure all SDK reads are finished before pop_layout restores a layout that
		// may be written by the rest of the presentation pipeline.
		vk::insert_image_memory_barrier(
			cmd,
			image->value,
			image->current_layout,
			image->current_layout,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
			make_subresource_range(aspect));
		image->pop_layout(cmd);
	}

	void prepare_ngx_output(const vk::command_buffer& cmd, vk::viewable_image* image)
	{
		const VkImageLayout old_layout = image->current_layout;
		const bool discard_contents = old_layout == VK_IMAGE_LAYOUT_UNDEFINED;

		vk::insert_image_memory_barrier(
			cmd,
			image->value,
			old_layout,
			VK_IMAGE_LAYOUT_GENERAL,
			discard_contents ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			discard_contents ? 0 : (VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT),
			VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			make_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
		image->current_layout = VK_IMAGE_LAYOUT_GENERAL;
	}

	void finish_ngx_output(
		const vk::command_buffer& cmd,
		vk::viewable_image* image,
		VkImageLayout new_layout,
		VkPipelineStageFlags dst_stage,
		VkAccessFlags dst_access)
	{
		vk::insert_image_memory_barrier(
			cmd,
			image->value,
			image->current_layout,
			new_layout,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			dst_stage,
			VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
			dst_access,
			make_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
		image->current_layout = new_layout;
	}

	NVSDK_NGX_Resource_VK wrap_image(vk::viewable_image* image, VkImageAspectFlags aspect, bool read_write)
	{
		const VkImageSubresourceRange range = make_subresource_range(aspect);
		const auto view = image->get_view(rsx::default_remap_vector.with_encoding(vk::VK_REMAP_IDENTITY), aspect);
		return NVSDK_NGX_Create_ImageView_Resource_VK(
			view->value,
			image->value,
			range,
			image->format(),
			image->width(),
			image->height(),
			read_write);
	}
}

namespace vk::dlss
{
	bool append_required_instance_extensions(std::vector<const char*>& extensions)
	{
		std::vector<std::string> required;
		const auto discovery_info = make_discovery_info();
		u32 count = 0;
		VkExtensionProperties* properties = nullptr;
		NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&discovery_info, &count, &properties);

		if (NVSDK_NGX_SUCCEED(result))
		{
			required.reserve(count);
			for (u32 i = 0; i < count; ++i)
			{
				required.emplace_back(properties[i].extensionName);
			}
		}
		else
		{
			unsigned int instance_count = 0;
			unsigned int device_count = 0;
			const char** instance_extensions = nullptr;
			const char** device_extensions = nullptr;
			result = NVSDK_NGX_VULKAN_RequiredExtensions(&instance_count, &instance_extensions, &device_count, &device_extensions);
			if (NVSDK_NGX_FAILED(result))
			{
				rsx_log.error("Failed to query NVIDIA DLSS Vulkan instance extensions (0x%x).", static_cast<u32>(result));
				return false;
			}

			required.reserve(instance_count);
			for (u32 i = 0; i < instance_count; ++i)
			{
				required.emplace_back(instance_extensions[i]);
			}
		}

		u32 supported_count = 0;
		if (vkEnumerateInstanceExtensionProperties(nullptr, &supported_count, nullptr) != VK_SUCCESS)
			return false;

		std::vector<VkExtensionProperties> supported(supported_count);
		if (vkEnumerateInstanceExtensionProperties(nullptr, &supported_count, supported.data()) != VK_SUCCESS)
			return false;

		static std::vector<std::string> extension_storage;
		if (!append_extensions(extensions, extension_storage, required, supported, "instance"))
			return false;

		rsx_log.notice("NVIDIA DLSS Vulkan instance requirements enabled (%u extensions).", static_cast<u32>(required.size()));
		return true;
	}

	bool append_required_device_extensions(VkInstance instance, VkPhysicalDevice physical_device, std::vector<const char*>& extensions)
	{
		std::vector<std::string> required;
		const auto discovery_info = make_discovery_info();
		u32 count = 0;
		VkExtensionProperties* properties = nullptr;
		NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physical_device, &discovery_info, &count, &properties);

		if (NVSDK_NGX_SUCCEED(result))
		{
			required.reserve(count);
			for (u32 i = 0; i < count; ++i)
			{
				required.emplace_back(properties[i].extensionName);
			}
		}
		else
		{
			unsigned int instance_count = 0;
			unsigned int device_count = 0;
			const char** instance_extensions = nullptr;
			const char** device_extensions = nullptr;
			result = NVSDK_NGX_VULKAN_RequiredExtensions(&instance_count, &instance_extensions, &device_count, &device_extensions);
			if (NVSDK_NGX_FAILED(result))
			{
				rsx_log.error("Failed to query NVIDIA DLSS Vulkan device extensions (0x%x).", static_cast<u32>(result));
				return false;
			}

			required.reserve(device_count);
			for (u32 i = 0; i < device_count; ++i)
			{
				required.emplace_back(device_extensions[i]);
			}
		}

		u32 supported_count = 0;
		if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &supported_count, nullptr) != VK_SUCCESS)
			return false;

		std::vector<VkExtensionProperties> supported(supported_count);
		if (vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &supported_count, supported.data()) != VK_SUCCESS)
			return false;

		static std::vector<std::string> extension_storage;
		if (!append_extensions(extensions, extension_storage, required, supported, "device"))
			return false;

		rsx_log.notice("NVIDIA DLSS Vulkan device requirements enabled (%u extensions).", static_cast<u32>(required.size()));
		return true;
	}
}

namespace vk
{
	class optical_flow_convert_pass final : public compute_task
	{
		vk::image_view* m_input = nullptr;
		vk::image_view* m_output = nullptr;
		std::unique_ptr<vk::sampler> m_sampler;

	public:
		optical_flow_convert_pass()
		{
			ssbo_count = 0;
			m_src = R"glsl(
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D InputTexture;
layout(rg16f, set = 0, binding = 1) writeonly uniform image2D OutputTexture;

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 extent = imageSize(OutputTexture);
	if (any(greaterThanEqual(pixel, extent)))
		return;

	// VK_FORMAT_R16G16_SFIXED5_NV samples directly as floating-point
	// pixel displacement. Forward OF maps current-frame pixels to their
	// previous-frame positions, which is the convention DLSS expects.
	vec2 motion = texelFetch(InputTexture, pixel, 0).rg;
	imageStore(OutputTexture, pixel, vec4(motion, 0.0, 0.0));
}
)glsl";
			create();
		}

		std::vector<glsl::program_input> get_inputs() override
		{
			return {
				glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"InputTexture",
					vk::glsl::input_type_texture,
					0,
					0),
				glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"OutputTexture",
					vk::glsl::input_type_storage_texture,
					0,
					1)};
		}

		void bind_resources(const vk::command_buffer& /*cmd*/) override
		{
			if (!m_sampler)
			{
				const auto device = vk::get_current_renderer();
				m_sampler = std::make_unique<vk::sampler>(
					*device,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE,
					0.f,
					1.f,
					0.f,
					0.f,
					VK_FILTER_NEAREST,
					VK_FILTER_NEAREST,
					VK_SAMPLER_MIPMAP_MODE_NEAREST,
					VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
			}

			m_program->bind_uniform({*m_input, *m_sampler}, 0, 0);
			m_program->bind_uniform({*m_output}, 0, 1);
		}

		void run(const vk::command_buffer& cmd, vk::viewable_image* src, vk::viewable_image* dst)
		{
			m_input = src->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			m_output = dst->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			compute_task::run(cmd, utils::aligned_div(dst->width(), 16u), utils::aligned_div(dst->height(), 16u), 1);
		}
	};

	class depth_history_pass final : public compute_task
	{
		vk::image_view* m_input = nullptr;
		vk::image_view* m_output = nullptr;
		std::unique_ptr<vk::sampler> m_sampler;

	public:
		depth_history_pass()
		{
			ssbo_count = 0;
			m_src = R"glsl(
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D DepthTexture;
layout(r32f, set = 0, binding = 1) writeonly uniform image2D HistoryTexture;

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 extent = imageSize(HistoryTexture);
	if (any(greaterThanEqual(pixel, extent)))
		return;

	float depth = texelFetch(DepthTexture, pixel, 0).r;
	if (isnan(depth) || isinf(depth))
		depth = 1.0;
	imageStore(HistoryTexture, pixel, vec4(depth, 0.0, 0.0, 0.0));
}
)glsl";
			create();
		}

		std::vector<glsl::program_input> get_inputs() override
		{
			return {
				glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"DepthTexture",
					vk::glsl::input_type_texture,
					0,
					0),
				glsl::program_input::make(
					::glsl::program_domain::glsl_compute_program,
					"HistoryTexture",
					vk::glsl::input_type_storage_texture,
					0,
					1)};
		}

		void bind_resources(const vk::command_buffer& /*cmd*/) override
		{
			if (!m_sampler)
			{
				const auto device = vk::get_current_renderer();
				m_sampler = std::make_unique<vk::sampler>(
					*device,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE, 0.f, 1.f, 0.f, 0.f,
					VK_FILTER_NEAREST, VK_FILTER_NEAREST,
					VK_SAMPLER_MIPMAP_MODE_NEAREST,
					VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE);
			}

			m_program->bind_uniform({*m_input, *m_sampler}, 0, 0);
			m_program->bind_uniform({*m_output}, 0, 1);
		}

		bool run(
			const vk::command_buffer& cmd,
			vk::viewable_image* source,
			vk::viewable_image* destination)
		{
			const VkImageAspectFlags source_aspect = (source->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT)
				? VK_IMAGE_ASPECT_DEPTH_BIT
				: VK_IMAGE_ASPECT_COLOR_BIT;
			m_input = source->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY), source_aspect);
			m_output = destination->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			if (!m_input || !m_output)
				return false;

			compute_task::run(
				cmd,
				utils::aligned_div(destination->width(), 16u),
				utils::aligned_div(destination->height(), 16u),
				1);
			return true;
		}
	};

	class geometry_motion_vector_pass final : public overlay_pass
	{
		struct pending_coverage_query
		{
			u32 index = umax;
			u64 frame_id = 0;
			u64 render_samples = 0;
			u32 meshes = 0;
			u32 triangle_vertices = 0;
		};

		std::unique_ptr<query_pool_manager> m_coverage_queries;
		std::deque<pending_coverage_query> m_pending_coverage;
		u32 m_active_coverage_query = umax;
		u64 m_last_submitted_frame = 0;
		u64 m_submitted_frames = 0;
		u64 m_measured_frames = 0;
		u64 m_nonzero_frames = 0;
		u64 m_surviving_samples = 0;
		u64 m_render_samples = 0;
		u64 m_max_frame_samples = 0;
		u64 m_skipped_queries = 0;
		bool m_coverage_summary_reported = false;

	public:
		geometry_motion_vector_pass()
		{
			m_num_usable_samplers = 2;
			m_num_uniform_buffers = 0;
			renderpass_config.set_primitive_type(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
			renderpass_config.set_attachment_count(1);
			renderpass_config.set_color_mask(0, true, true, true, true);
			renderpass_config.set_depth_mask(false);
			num_drawable_elements = 0;

			vs_src = R"glsl(
#version 450
layout(location = 0) in vec4 CurrentClip;
layout(location = 1) in vec4 PreviousClip;
layout(location = 0) out vec4 CurrentClipOut;
layout(location = 1) out vec4 PreviousClipOut;
void main()
{
	gl_Position = CurrentClip;
	CurrentClipOut = CurrentClip;
	PreviousClipOut = PreviousClip;
}
)glsl";

			fs_src = R"glsl(
#version 450
layout(set = 0, binding = 0) uniform sampler2D fs0;
layout(set = 0, binding = 1) uniform sampler2D fs1;
layout(location = 0) in vec4 CurrentClipIn;
layout(location = 1) in vec4 PreviousClipIn;
layout(location = 0) out vec4 MotionAndMask;
void main()
{
	ivec2 pixel = ivec2(gl_FragCoord.xy);
	ivec2 extent = textureSize(fs0, 0);
	if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, extent)) ||
		abs(CurrentClipIn.w) < 1.e-7 || PreviousClipIn.w <= 1.e-7)
		discard;

	float scene_depth = texelFetch(fs0, pixel, 0).r;
	float geometry_depth = CurrentClipIn.z / CurrentClipIn.w;
	float depth_tolerance = max(0.00075, 0.0025 * max(abs(scene_depth), abs(geometry_depth)));
	if (isnan(scene_depth) || isinf(scene_depth) || isnan(geometry_depth) || isinf(geometry_depth) ||
		abs(scene_depth - geometry_depth) > depth_tolerance)
		discard;

	vec2 current_ndc = CurrentClipIn.xy / CurrentClipIn.w;
	vec2 previous_ndc = PreviousClipIn.xy / PreviousClipIn.w;
	float previous_geometry_depth = PreviousClipIn.z / PreviousClipIn.w;
	vec2 previous_pixel_position = (previous_ndc * 0.5 + 0.5) * vec2(extent) - vec2(0.5);
	ivec2 previous_pixel = ivec2(floor(previous_pixel_position + vec2(0.5)));
	if (any(lessThan(previous_pixel, ivec2(0))) || any(greaterThanEqual(previous_pixel, extent)) ||
		isnan(previous_geometry_depth) || isinf(previous_geometry_depth))
		discard;

	// A current-frame depth match proves that this triangle is visible now. It
	// does not prove that its reprojected location contained the same surface in
	// the previous frame. Compare against the retained game-depth history and
	// reject disocclusions, occlusion boundaries and bad mesh associations. A
	// small 3x3 search absorbs quantization and sub-pixel raster differences.
	float previous_depth_error = 1.e20;
	for (int y = -1; y <= 1; ++y)
	{
		for (int x = -1; x <= 1; ++x)
		{
			ivec2 history_pixel = clamp(previous_pixel + ivec2(x, y), ivec2(0), extent - ivec2(1));
			float previous_scene_depth = texelFetch(fs1, history_pixel, 0).r;
			if (!isnan(previous_scene_depth) && !isinf(previous_scene_depth))
				previous_depth_error = min(previous_depth_error, abs(previous_scene_depth - previous_geometry_depth));
		}
	}
	float previous_depth_tolerance = max(0.0015, 0.005 * abs(previous_geometry_depth));
	if (previous_depth_error > previous_depth_tolerance)
		discard;

	vec2 motion = (previous_ndc - current_ndc) * (0.5 * vec2(extent));
	if (any(isnan(motion)) || any(isinf(motion)) || length(motion) > 0.5 * length(vec2(extent)))
		discard;

	MotionAndMask = vec4(motion, 1.0, 0.0);
}
)glsl";
		}

		void create(const vk::render_device& dev) override
		{
			overlay_pass::create(dev);
			m_coverage_queries = std::make_unique<query_pool_manager>(
				*const_cast<vk::render_device*>(&dev), VK_QUERY_TYPE_OCCLUSION, 64);
			// Do not accept partial counts: coverage is diagnostic data, not a
			// visibility predicate. Polling remains non-blocking until availability.
			m_coverage_queries->set_control_flags(0, 0);
		}

		void destroy() override
		{
			m_pending_coverage.clear();
			m_coverage_queries.reset();
			overlay_pass::destroy();
		}

		std::vector<VkVertexInputBindingDescription> get_vertex_bindings() override
		{
			return {{0, sizeof(rsx::character_vertex_probe::motion_raster_vertex), VK_VERTEX_INPUT_RATE_VERTEX}};
		}

		std::vector<VkVertexInputAttributeDescription> get_vertex_attributes() override
		{
			return {
				{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(rsx::character_vertex_probe::motion_raster_vertex, current_clip)},
				{1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(rsx::character_vertex_probe::motion_raster_vertex, previous_clip)},
			};
		}

		void emit_geometry(vk::command_buffer& cmd, glsl::program* program) override
		{
			if (m_active_coverage_query != umax)
				m_coverage_queries->begin_query(cmd, m_active_coverage_query);

			overlay_pass::emit_geometry(cmd, program);

			if (m_active_coverage_query != umax)
				m_coverage_queries->end_query(cmd, m_active_coverage_query);
		}

		void poll_coverage(vk::command_buffer& cmd, u64 current_frame)
		{
			while (!m_pending_coverage.empty())
			{
				const auto& pending = m_pending_coverage.front();
				if (!m_coverage_queries->check_query_status(pending.index))
					break;

				const u64 samples = m_coverage_queries->get_query_result(pending.index);
				m_measured_frames++;
				m_nonzero_frames += samples ? 1 : 0;
				m_surviving_samples += samples;
				m_render_samples += pending.render_samples;
				m_max_frame_samples = std::max(m_max_frame_samples, samples);
				const double ratio = pending.render_samples
					? (100. * static_cast<double>(samples) / pending.render_samples)
					: 0.;
				rsx_log.notice(
					"NVIDIA DLSS geometry coverage: frame=%llu samples=%llu ratio=%.6f%% meshes=%u triangle_vertices=%u.",
					pending.frame_id,
					samples,
					ratio,
					pending.meshes,
					pending.triangle_vertices);

				m_coverage_queries->free_query(cmd, pending.index);
				m_pending_coverage.pop_front();
			}

			if (!m_coverage_summary_reported && m_submitted_frames && m_pending_coverage.empty() &&
				current_frame > m_last_submitted_frame + 8)
			{
				const double ratio = m_render_samples
					? (100. * static_cast<double>(m_surviving_samples) / m_render_samples)
					: 0.;
				rsx_log.success(
					"NVIDIA DLSS geometry coverage summary: submitted=%llu measured=%llu nonzero=%llu samples=%llu ratio=%.6f%% max_frame_samples=%llu skipped=%llu.",
					m_submitted_frames,
					m_measured_frames,
					m_nonzero_frames,
					m_surviving_samples,
					ratio,
					m_max_frame_samples,
					m_skipped_queries);
				m_coverage_summary_reported = true;
			}
		}

		bool run(
			vk::command_buffer& cmd,
			vk::viewable_image* target,
			vk::viewable_image* depth,
			vk::viewable_image* previous_depth,
			const rsx::character_vertex_probe::motion_raster_frame& frame)
		{
			if (!target || !depth || !previous_depth || frame.vertices.empty() ||
				frame.render_width != target->width() || frame.render_height != target->height())
			{
				return false;
			}

			num_drawable_elements = ::size32(frame.vertices);
			upload_vertex_data(frame.vertices.data(), num_drawable_elements);
			const VkImageAspectFlags depth_aspect = (depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT)
				? VK_IMAGE_ASPECT_DEPTH_BIT
				: VK_IMAGE_ASPECT_COLOR_BIT;
			auto* depth_view = depth->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY), depth_aspect);
			auto* previous_depth_view = previous_depth->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			if (!depth_view || !previous_depth_view)
				return false;

			m_active_coverage_query = umax;
			if (m_coverage_queries && !(cmd.flags & vk::command_buffer::cb_has_open_query))
				m_active_coverage_query = m_coverage_queries->allocate_query(cmd);

			const VkRenderPass render_pass = vk::get_renderpass(*m_device, vk::get_renderpass_key(target->format()));
			overlay_pass::run(
				cmd,
				{0, 0, frame.render_width, frame.render_height},
				target,
				{depth_view, previous_depth_view},
				render_pass);
			vk::end_renderpass(cmd);
			if (m_active_coverage_query != umax)
			{
				m_pending_coverage.push_back({
					.index = m_active_coverage_query,
					.frame_id = frame.frame_id,
					.render_samples = static_cast<u64>(frame.render_width) * frame.render_height,
					.meshes = frame.matched_meshes,
					.triangle_vertices = num_drawable_elements,
				});
				m_submitted_frames++;
				m_last_submitted_frame = frame.frame_id;
				m_coverage_summary_reported = false;
			}
			else
			{
				m_skipped_queries++;
			}
			m_active_coverage_query = umax;
			return true;
		}
	};

	class camera_motion_vector_pass final : public compute_task
	{
		struct alignas(16) camera_parameters
		{
			std::array<float, 16> inverse_current_view_projection{};
			std::array<float, 16> previous_view_projection{};
		};
		static_assert(sizeof(camera_parameters) == 128);

		vk::image_view* m_depth = nullptr;
		vk::image_view* m_optical_flow = nullptr;
		vk::image_view* m_geometry_motion = nullptr;
		vk::image_view* m_output = nullptr;
		std::unique_ptr<vk::sampler> m_sampler;
		camera_parameters m_parameters{};

	public:
		camera_motion_vector_pass()
		{
			ssbo_count = 0;
			use_push_constants = true;
			push_constants_size = sizeof(camera_parameters);
			m_src = R"glsl(
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler2D DepthTexture;
layout(set = 0, binding = 1) uniform sampler2D OpticalFlowTexture;
layout(set = 0, binding = 2) uniform sampler2D GeometryMotionTexture;
layout(rg16f, set = 0, binding = 3) writeonly uniform image2D OutputTexture;
layout(push_constant) uniform CameraParameters
{
	vec4 inverse_current_view_projection[4];
	vec4 previous_view_projection[4];
} Camera;

vec4 multiply_rows(vec4 rows[4], vec4 value)
{
	return vec4(
		dot(rows[0], value),
		dot(rows[1], value),
		dot(rows[2], value),
		dot(rows[3], value));
}

void main()
{
	ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
	ivec2 extent = imageSize(OutputTexture);
	if (any(greaterThanEqual(pixel, extent)))
		return;

	vec2 optical_motion = texelFetch(OpticalFlowTexture, pixel, 0).rg;
	if (any(isnan(optical_motion)) || any(isinf(optical_motion)))
		optical_motion = vec2(0.0);
	vec4 geometry_motion = texelFetch(GeometryMotionTexture, pixel, 0);
	if (geometry_motion.b > 0.5 && !any(isnan(geometry_motion.rg)) && !any(isinf(geometry_motion.rg)))
	{
		imageStore(OutputTexture, pixel, vec4(geometry_motion.rg, 0.0, 0.0));
		return;
	}
	float depth = texelFetch(DepthTexture, pixel, 0).r;
	if (isnan(depth) || isinf(depth) || depth >= 0.9999999)
	{
		imageStore(OutputTexture, pixel, vec4(optical_motion, 0.0, 0.0));
		return;
	}

	vec2 current_uv = (vec2(pixel) + vec2(0.5)) / vec2(extent);
	vec2 current_ndc = current_uv * 2.0 - 1.0;
	vec4 current_clip = vec4(current_ndc, depth, 1.0);
	vec4 world = multiply_rows(Camera.inverse_current_view_projection, current_clip);
	if (abs(world.w) < 1.e-7 || any(isnan(world)) || any(isinf(world)))
	{
		imageStore(OutputTexture, pixel, vec4(optical_motion, 0.0, 0.0));
		return;
	}

	world /= world.w;
	vec4 previous_clip = multiply_rows(Camera.previous_view_projection, vec4(world.xyz, 1.0));
	if (previous_clip.w <= 1.e-7 || any(isnan(previous_clip)) || any(isinf(previous_clip)))
	{
		imageStore(OutputTexture, pixel, vec4(optical_motion, 0.0, 0.0));
		return;
	}

	vec2 previous_ndc = previous_clip.xy / previous_clip.w;
	// DLSS pixel-space convention: where this current pixel was in the
	// previous frame, minus its current position.
	vec2 camera_motion = (previous_ndc - current_ndc) * (0.5 * vec2(extent));
	float camera_length = length(camera_motion);
	if (any(isnan(camera_motion)) || any(isinf(camera_motion)) || camera_length > 0.5 * length(vec2(extent)))
	{
		imageStore(OutputTexture, pixel, vec4(optical_motion, 0.0, 0.0));
		return;
	}

	// Optical flow retains moving-object motion. The reconstructed camera
	// vector replaces its noisy static-scene component only where both sources
	// agree; disagreement and occlusion edges automatically keep optical flow.
	float disagreement = length(optical_motion - camera_motion);
	// The v5 conservative trial proved the direction and depth convention on
	// real hardware. The balanced gate now gives agreeing static geometry an
	// almost exact camera vector, while residual motion still falls back to OF.
	float agreement_limit = clamp(1.0 + camera_length * 0.20, 1.0, 4.0);
	float camera_weight = 0.97 * (1.0 - smoothstep(agreement_limit, agreement_limit * 2.75, disagreement));
	vec2 fused_motion = mix(optical_motion, camera_motion, camera_weight);
	imageStore(OutputTexture, pixel, vec4(fused_motion, 0.0, 0.0));
}
)glsl";
			create();
		}

		std::vector<glsl::program_input> get_inputs() override
		{
			auto inputs = compute_task::get_inputs();
			inputs.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"DepthTexture", vk::glsl::input_type_texture, 0, 0));
			inputs.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"OpticalFlowTexture", vk::glsl::input_type_texture, 0, 1));
			inputs.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"GeometryMotionTexture", vk::glsl::input_type_texture, 0, 2));
			inputs.push_back(glsl::program_input::make(
				::glsl::program_domain::glsl_compute_program,
				"OutputTexture", vk::glsl::input_type_storage_texture, 0, 3));
			return inputs;
		}

		void bind_resources(const vk::command_buffer& cmd) override
		{
			if (!m_sampler)
			{
				const auto device = vk::get_current_renderer();
				m_sampler = std::make_unique<vk::sampler>(
					*device,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
					VK_FALSE, 0.f, 1.f, 0.f, 0.f,
					VK_FILTER_NEAREST, VK_FILTER_NEAREST,
					VK_SAMPLER_MIPMAP_MODE_NEAREST,
					VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
			}

			vkCmdPushConstants(cmd, m_program->layout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constants_size, &m_parameters);
			m_program->bind_uniform({*m_depth, *m_sampler}, 0, 0);
			m_program->bind_uniform({*m_optical_flow, *m_sampler}, 0, 1);
			m_program->bind_uniform({*m_geometry_motion, *m_sampler}, 0, 2);
			m_program->bind_uniform({*m_output}, 0, 3);
		}

		bool run(
			const vk::command_buffer& cmd,
			vk::viewable_image* depth,
			vk::viewable_image* optical_flow,
			vk::viewable_image* geometry_motion,
			vk::viewable_image* output,
			const temporal_frame_data& frame)
		{
			if (!invert_matrix_4x4(frame.current_view_projection, m_parameters.inverse_current_view_projection))
				return false;

			m_parameters.previous_view_projection = frame.previous_view_projection;
			const VkImageAspectFlags depth_aspect = (depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT)
				? VK_IMAGE_ASPECT_DEPTH_BIT
				: VK_IMAGE_ASPECT_COLOR_BIT;
			m_depth = depth->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY), depth_aspect);
			m_optical_flow = optical_flow->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			m_geometry_motion = geometry_motion->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			m_output = output->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
			if (!m_depth || !m_optical_flow || !m_geometry_motion || !m_output)
				return false;

			compute_task::run(cmd, utils::aligned_div(output->width(), 16u), utils::aligned_div(output->height(), 16u), 1);
			return true;
		}
	};

	struct dlss_optical_flow_slot
	{
		std::unique_ptr<vk::viewable_image> input;
		std::unique_ptr<vk::viewable_image> flow;
		std::unique_ptr<vk::viewable_image> motion_vectors;
		VkOpticalFlowSessionNV session = VK_NULL_HANDLE;
		VkSemaphore graphics_ready = VK_NULL_HANDLE;
		VkSemaphore optical_done = VK_NULL_HANDLE;
		vk::command_pool command_pool;
		vk::command_buffer command_buffer;
		bool command_resources_created = false;
	};

	class dlss_optical_flow_state
	{
		const vk::render_device* m_device = nullptr;
		std::vector<std::unique_ptr<dlss_optical_flow_slot>> m_slots;
		std::unique_ptr<optical_flow_convert_pass> m_converter;
		size2u m_size{};
		u32 m_previous_slot = umax;
		u32 m_pending_slot = umax;
		u32 m_pending_reference_slot = umax;
		bool m_pending_submitted = false;
		bool m_failed = false;

		void destroy()
		{
			if (!m_slots.empty() && m_device)
			{
				vkDeviceWaitIdle(*m_device);
			}

			m_converter.reset();
			for (auto& slot : m_slots)
			{
				if (slot->session && _vkDestroyOpticalFlowSessionNV)
					_vkDestroyOpticalFlowSessionNV(*m_device, slot->session, nullptr);
				if (slot->graphics_ready)
					vkDestroySemaphore(*m_device, slot->graphics_ready, nullptr);
				if (slot->optical_done)
					vkDestroySemaphore(*m_device, slot->optical_done, nullptr);
				if (slot->command_resources_created)
				{
					slot->command_buffer.destroy();
					slot->command_pool.destroy();
				}
			}

			m_slots.clear();
			m_size = {};
			m_previous_slot = umax;
			m_pending_slot = umax;
			m_pending_reference_slot = umax;
			m_pending_submitted = false;
		}

		bool initialize(const size2u& size, u32 slot_count)
		{
			destroy();
			m_device = vk::get_current_renderer();
			if (!m_device || !m_device->get_optical_flow_support() ||
				!_vkCreateOpticalFlowSessionNV || !_vkDestroyOpticalFlowSessionNV ||
				!_vkBindOpticalFlowSessionImageNV || !_vkCmdOpticalFlowExecuteNV ||
				!_vkCmdPipelineBarrier2KHR)
			{
				m_failed = true;
				return false;
			}

			const auto flow_features = m_device->get_format_properties(VK_FORMAT_R16G16_SFIXED5_NV).optimalTilingFeatures;
			const auto motion_features = m_device->get_format_properties(VK_FORMAT_R16G16_SFLOAT).optimalTilingFeatures;
			if (!(flow_features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ||
				!(motion_features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT))
			{
				rsx_log.warning("NVIDIA Optical Flow formats cannot be sampled/converted on this driver.");
				m_failed = true;
				return false;
			}

			slot_count = std::max(2u, slot_count);
			m_slots.reserve(slot_count);
			for (u32 i = 0; i < slot_count; ++i)
			{
				m_slots.emplace_back(std::make_unique<dlss_optical_flow_slot>());
				auto& slot = m_slots.back();
				VkOpticalFlowImageFormatInfoNV input_optical_info{
					.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
					.usage = VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV};
				VkOpticalFlowImageFormatInfoNV output_optical_info{
					.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV,
					.usage = VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV};

				auto create_image = [&](VkFormat format, VkImageUsageFlags usage, const void* creation_pnext)
				{
					return std::make_unique<vk::viewable_image>(
						*m_device,
						m_device->get_memory_mapping().device_local,
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
						VK_IMAGE_TYPE_2D,
						format,
						size.width, size.height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
						VK_IMAGE_LAYOUT_UNDEFINED,
						VK_IMAGE_TILING_OPTIMAL,
						usage,
						VK_IMAGE_CREATE_ALLOW_NULL_RPCS3 | VK_IMAGE_CREATE_OPTICAL_FLOW_SHAREABLE_RPCS3,
						VMM_ALLOCATION_POOL_SWAPCHAIN,
						RSX_FORMAT_CLASS_COLOR,
						creation_pnext);
				};

				slot->input = create_image(
					VK_FORMAT_B8G8R8A8_UNORM,
					VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					&input_optical_info);
				slot->flow = create_image(
					VK_FORMAT_R16G16_SFIXED5_NV,
					VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
					&output_optical_info);
				slot->motion_vectors = create_image(
					VK_FORMAT_R16G16_SFLOAT,
					VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
					nullptr);

				if (!slot->input || !slot->input->value || !slot->flow || !slot->flow->value ||
					!slot->motion_vectors || !slot->motion_vectors->value)
				{
					rsx_log.warning("Failed to allocate NVIDIA Optical Flow images.");
					m_failed = true;
					destroy();
					return false;
				}

				VkOpticalFlowSessionCreateInfoNV session_info{
					.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_SESSION_CREATE_INFO_NV,
					.width = size.width,
					.height = size.height,
					.imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
					.flowVectorFormat = VK_FORMAT_R16G16_SFIXED5_NV,
					.costFormat = VK_FORMAT_UNDEFINED,
					.outputGridSize = VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV,
					.hintGridSize = VK_OPTICAL_FLOW_GRID_SIZE_UNKNOWN_NV,
					.performanceLevel = VK_OPTICAL_FLOW_PERFORMANCE_LEVEL_MEDIUM_NV,
					.flags = 0};
				if (_vkCreateOpticalFlowSessionNV(*m_device, &session_info, nullptr, &slot->session) != VK_SUCCESS)
				{
					rsx_log.warning("Failed to create an NVIDIA Optical Flow session.");
					m_failed = true;
					destroy();
					return false;
				}

				VkSemaphoreCreateInfo semaphore_info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
				if (vkCreateSemaphore(*m_device, &semaphore_info, nullptr, &slot->graphics_ready) != VK_SUCCESS ||
					vkCreateSemaphore(*m_device, &semaphore_info, nullptr, &slot->optical_done) != VK_SUCCESS)
				{
					rsx_log.warning("Failed to create NVIDIA Optical Flow synchronization objects.");
					m_failed = true;
					destroy();
					return false;
				}

				slot->command_pool.create(*const_cast<vk::render_device*>(m_device), m_device->get_optical_flow_queue_family());
				slot->command_buffer.create(slot->command_pool);
				slot->command_buffer.access_hint = vk::command_buffer::access_type_hint::all;
				slot->command_resources_created = true;
			}

			m_converter = std::make_unique<optical_flow_convert_pass>();
			m_size = size;
			m_failed = false;
			rsx_log.success("NVIDIA Optical Flow motion estimation initialized for %ux%u (%u frame slots).", size.width, size.height, slot_count);
			return true;
		}

	public:
		~dlss_optical_flow_state()
		{
			destroy();
		}

		bool prepare(
			const vk::command_buffer& cmd,
			vk::viewable_image* src,
			const size2u& size,
			u32 frame_slot,
			u32 frame_slot_count)
		{
			if (m_failed || !src || !(src->aspect() & VK_IMAGE_ASPECT_COLOR_BIT) ||
				!(src->info.usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
			{
				return false;
			}

			if (m_slots.empty() || m_size.width != size.width || m_size.height != size.height ||
				m_slots.size() != std::max<usz>(2, frame_slot_count))
			{
				if (!initialize(size, frame_slot_count))
					return false;
			}

			frame_slot %= ::size32(m_slots);
			auto& current = *m_slots[frame_slot];
			const VkImageSubresourceLayers layers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
			VkImageBlit region{};
			region.srcSubresource = layers;
			region.dstSubresource = layers;
			region.srcOffsets[1] = {static_cast<s32>(size.width), static_cast<s32>(size.height), 1};
			region.dstOffsets[1] = region.srcOffsets[1];

			src->push_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
			current.input->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdBlitImage(
				cmd,
				src->value,
				src->current_layout,
				current.input->value,
				current.input->current_layout,
				1,
				&region,
				VK_FILTER_NEAREST);
			current.input->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			src->pop_layout(cmd);

			if (m_previous_slot == umax || m_previous_slot == frame_slot)
			{
				m_previous_slot = frame_slot;
				m_pending_slot = umax;
				m_pending_reference_slot = umax;
				m_pending_submitted = false;
				return false;
			}

			auto& reference = *m_slots[m_previous_slot];
			const auto bind = [&](VkOpticalFlowSessionBindingPointNV point, vk::viewable_image* image, VkImageLayout layout)
			{
				const auto view = image->get_view(rsx::default_remap_vector.with_encoding(VK_REMAP_IDENTITY));
				return _vkBindOpticalFlowSessionImageNV(*m_device, current.session, point, view->value, layout) == VK_SUCCESS;
			};

			if (!bind(VK_OPTICAL_FLOW_SESSION_BINDING_POINT_INPUT_NV, current.input.get(), VK_IMAGE_LAYOUT_GENERAL) ||
				!bind(VK_OPTICAL_FLOW_SESSION_BINDING_POINT_REFERENCE_NV, reference.input.get(), VK_IMAGE_LAYOUT_GENERAL) ||
				!bind(VK_OPTICAL_FLOW_SESSION_BINDING_POINT_FLOW_VECTOR_NV, current.flow.get(), VK_IMAGE_LAYOUT_GENERAL))
			{
				rsx_log.warning("Failed to bind NVIDIA Optical Flow frame images. Zero motion vectors will be used.");
				m_failed = true;
				return false;
			}

			m_pending_slot = frame_slot;
			m_pending_reference_slot = m_previous_slot;
			m_previous_slot = frame_slot;
			m_pending_submitted = false;
			return true;
		}

		VkSemaphore ready_semaphore() const
		{
			return m_pending_slot == umax ? VK_NULL_HANDLE : m_slots[m_pending_slot]->graphics_ready;
		}

		VkSemaphore submit()
		{
			if (m_pending_slot == umax || m_pending_reference_slot == umax)
				return VK_NULL_HANDLE;

			auto& current = *m_slots[m_pending_slot];
			auto& reference = *m_slots[m_pending_reference_slot];
			current.command_buffer.begin();

			std::array<VkImageMemoryBarrier2, 3> before{};
			auto initialize_barrier = [](VkImageMemoryBarrier2& barrier, vk::viewable_image* image)
			{
				barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
				barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
				barrier.srcAccessMask = VK_ACCESS_2_NONE;
				barrier.dstStageMask = VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV;
				barrier.oldLayout = image->current_layout;
				barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
				barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				barrier.image = image->value;
				barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			};
			initialize_barrier(before[0], current.input.get());
			before[0].dstAccessMask = VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV;
			initialize_barrier(before[1], reference.input.get());
			before[1].dstAccessMask = VK_ACCESS_2_OPTICAL_FLOW_READ_BIT_NV;
			initialize_barrier(before[2], current.flow.get());
			before[2].dstAccessMask = VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV;

			VkDependencyInfo before_dependency{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = ::size32(before),
				.pImageMemoryBarriers = before.data()};
			_vkCmdPipelineBarrier2KHR(current.command_buffer, &before_dependency);
			current.flow->current_layout = VK_IMAGE_LAYOUT_GENERAL;

			VkOpticalFlowExecuteInfoNV execute_info{
				.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_EXECUTE_INFO_NV,
				.flags = VK_OPTICAL_FLOW_EXECUTE_DISABLE_TEMPORAL_HINTS_BIT_NV};
			_vkCmdOpticalFlowExecuteNV(current.command_buffer, current.session, &execute_info);

			VkImageMemoryBarrier2 after{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_OPTICAL_FLOW_BIT_NV,
				.srcAccessMask = VK_ACCESS_2_OPTICAL_FLOW_WRITE_BIT_NV,
				.dstStageMask = VK_PIPELINE_STAGE_2_NONE,
				.dstAccessMask = VK_ACCESS_2_NONE,
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = current.flow->value,
				.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
			VkDependencyInfo after_dependency{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &after};
			_vkCmdPipelineBarrier2KHR(current.command_buffer, &after_dependency);
			current.flow->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

			current.command_buffer.end();
			vk::queue_submit_t submit_info{m_device->get_optical_flow_queue(), nullptr};
			submit_info.wait_on(current.graphics_ready, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
			submit_info.queue_signal(current.optical_done);
			current.command_buffer.submit(submit_info);
			m_pending_submitted = true;
			return current.optical_done;
		}

		vk::viewable_image* motion_vectors() const
		{
			return (m_pending_slot == umax || !m_pending_submitted)
				? nullptr
				: m_slots[m_pending_slot]->motion_vectors.get();
		}

		void record_conversion(const vk::command_buffer& cmd, vk::viewable_image* expected_output)
		{
			if (m_pending_slot == umax || !m_pending_submitted)
				return;

			auto& current = *m_slots[m_pending_slot];
			if (expected_output != current.motion_vectors.get())
				return;

			VkImageMemoryBarrier2 acquire{
				.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
				.srcStageMask = VK_PIPELINE_STAGE_2_NONE,
				.srcAccessMask = VK_ACCESS_2_NONE,
				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
				.oldLayout = current.flow->current_layout,
				.newLayout = current.flow->current_layout,
				.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
				.image = current.flow->value,
				.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
			VkDependencyInfo acquire_dependency{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1,
				.pImageMemoryBarriers = &acquire};
			_vkCmdPipelineBarrier2KHR(cmd, &acquire_dependency);

			current.motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			m_converter->run(cmd, current.flow.get(), current.motion_vectors.get());
			vk::insert_image_memory_barrier(
				cmd,
				current.motion_vectors->value,
				current.motion_vectors->current_layout,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
				VK_ACCESS_SHADER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
			current.motion_vectors->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			m_pending_submitted = false;
		}
	};

	dlss_upscale_pass::dlss_upscale_pass() = default;

	dlss_upscale_pass::~dlss_upscale_pass()
	{
		release_feature();
		shutdown_ngx();
		m_optical_flow.reset();
		if (m_geometry_motion_pass)
		{
			m_geometry_motion_pass->destroy();
			m_geometry_motion_pass.reset();
		}
		dispose_images();
	}

	bool dlss_upscale_pass::prepare_optical_flow(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		const size2u& input_size,
		u32 frame_slot,
		u32 frame_slot_count)
	{
		if (!m_optical_flow)
			m_optical_flow = std::make_unique<dlss_optical_flow_state>();
		return m_optical_flow->prepare(cmd, src, input_size, frame_slot, frame_slot_count);
	}

	VkSemaphore dlss_upscale_pass::get_optical_flow_ready_semaphore() const
	{
		return m_optical_flow ? m_optical_flow->ready_semaphore() : VK_NULL_HANDLE;
	}

	VkSemaphore dlss_upscale_pass::submit_optical_flow()
	{
		return m_optical_flow ? m_optical_flow->submit() : VK_NULL_HANDLE;
	}

	vk::viewable_image* dlss_upscale_pass::get_optical_flow_motion_vectors() const
	{
		return m_optical_flow ? m_optical_flow->motion_vectors() : nullptr;
	}

	void dlss_upscale_pass::dispose_images()
	{
		auto safe_delete = [](auto& image)
		{
			if (image && image->value)
			{
				vk::get_resource_manager()->dispose(image);
			}
			else
			{
				image.reset();
			}
		};

		safe_delete(m_output);
		safe_delete(m_dummy_depth);
		safe_delete(m_dummy_motion_vectors);
		safe_delete(m_camera_motion_vectors);
		safe_delete(m_geometry_motion_vectors);
		safe_delete(m_previous_depth);
		m_previous_depth_valid = false;
		m_previous_depth_frame_id = umax;
	}

	void dlss_upscale_pass::release_feature()
	{
		if (!m_feature)
			return;

		// NGX feature handles must not be released while an earlier queue
		// submission can still be evaluating them. This is only reached when the
		// output size/depth convention changes or the upscaler is destroyed.
		vkDeviceWaitIdle(*vk::get_current_renderer());

		const auto result = NVSDK_NGX_VULKAN_ReleaseFeature(m_feature);
		if (NVSDK_NGX_FAILED(result))
		{
			rsx_log.error("Failed to release the NVIDIA DLSS feature (0x%x).", static_cast<u32>(result));
		}
		m_feature = nullptr;
	}

	void dlss_upscale_pass::shutdown_ngx()
	{
		if (!m_ngx_initialized)
			return;

		if (m_parameters)
		{
			const auto result = NVSDK_NGX_VULKAN_DestroyParameters(m_parameters);
			if (NVSDK_NGX_FAILED(result))
			{
				rsx_log.error("Failed to destroy NVIDIA NGX parameters (0x%x).", static_cast<u32>(result));
			}
			m_parameters = nullptr;
		}

		const VkDevice device = *vk::get_current_renderer();
		const auto result = NVSDK_NGX_VULKAN_Shutdown1(device);
		if (NVSDK_NGX_FAILED(result))
		{
			rsx_log.error("Failed to shut down NVIDIA NGX for Vulkan (0x%x).", static_cast<u32>(result));
		}
		m_ngx_initialized = false;
		m_dlss_available = false;
	}

	bool dlss_upscale_pass::initialize_ngx()
	{
		if (m_ngx_initialized)
			return m_dlss_available;

		const auto device = vk::get_current_renderer();
		const auto& gpu = device->gpu();
		const VkInstance instance = gpu;
		const VkPhysicalDevice physical_device = gpu;
		const VkDevice logical_device = *device;

		const auto result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
			g_dlss_project_id,
			NVSDK_NGX_ENGINE_TYPE_CUSTOM,
			g_dlss_engine_version,
			g_dlss_data_path,
			instance,
			physical_device,
			logical_device,
			vkGetInstanceProcAddr,
			vkGetDeviceProcAddr);

		if (NVSDK_NGX_FAILED(result))
		{
			rsx_log.error("Failed to initialize NVIDIA NGX for Vulkan (0x%x). DLSS will use bilinear fallback.", static_cast<u32>(result));
			return false;
		}

		m_ngx_initialized = true;
		const auto parameter_result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&m_parameters);
		if (NVSDK_NGX_FAILED(parameter_result) || !m_parameters)
		{
			rsx_log.error("Failed to query NVIDIA NGX capability parameters (0x%x).", static_cast<u32>(parameter_result));
			shutdown_ngx();
			return false;
		}

		int available = 0;
		const auto available_result = NVSDK_NGX_Parameter_GetI(m_parameters, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
		if (NVSDK_NGX_FAILED(available_result) || !available)
		{
			int feature_result = 0;
			NVSDK_NGX_Parameter_GetI(m_parameters, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &feature_result);
			rsx_log.error("NVIDIA DLSS Super Resolution is unavailable (query=0x%x, feature=0x%x).", static_cast<u32>(available_result), static_cast<u32>(feature_result));
			shutdown_ngx();
			return false;
		}

		m_dlss_available = true;
		rsx_log.success("NVIDIA DLSS Super Resolution initialized through Vulkan NGX.");
		return true;
	}

	bool dlss_upscale_pass::initialize_images(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		const size2u& input_size,
		const size2u& output_size)
	{
		dispose_images();

		const auto device = vk::get_current_renderer();
		auto create_image = [device](u32 width, u32 height, VkFormat format, VkImageUsageFlags usage)
		{
			return std::make_unique<vk::viewable_image>(
				*device,
				device->get_memory_mapping().device_local,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				VK_IMAGE_TYPE_2D,
				format,
				width, height, 1, 1, 1, VK_SAMPLE_COUNT_1_BIT,
				VK_IMAGE_LAYOUT_UNDEFINED,
				VK_IMAGE_TILING_OPTIMAL,
				usage,
				VK_IMAGE_CREATE_ALLOW_NULL_RPCS3,
				VMM_ALLOCATION_POOL_SWAPCHAIN,
				RSX_FORMAT_CLASS_COLOR);
		};

		VkFormat output_format = VK_FORMAT_UNDEFINED;
		const std::array output_candidates{
			src->format(),
			VK_FORMAT_R8G8B8A8_UNORM,
			VK_FORMAT_B8G8R8A8_UNORM,
			VK_FORMAT_R16G16B16A16_SFLOAT};
		constexpr VkFormatFeatureFlags required_output_features =
			VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
			VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
			VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
			VK_FORMAT_FEATURE_TRANSFER_DST_BIT;

		for (const VkFormat candidate : output_candidates)
		{
			if ((device->get_format_properties(candidate).optimalTilingFeatures & required_output_features) == required_output_features)
			{
				output_format = candidate;
				break;
			}
		}

		if (output_format == VK_FORMAT_UNDEFINED)
		{
			rsx_log.error("No Vulkan storage image format suitable for NVIDIA DLSS output was found.");
			return false;
		}

		m_output = create_image(
			output_size.width,
			output_size.height,
			output_format,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
			VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		m_dummy_motion_vectors = create_image(
			input_size.width,
			input_size.height,
			VK_FORMAT_R16G16_SFLOAT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		m_dummy_depth = create_image(
			input_size.width,
			input_size.height,
			VK_FORMAT_R32_SFLOAT,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		m_camera_motion_vectors = create_image(
			input_size.width,
			input_size.height,
			VK_FORMAT_R16G16_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		m_geometry_motion_vectors = create_image(
			input_size.width,
			input_size.height,
			VK_FORMAT_R16G16B16A16_SFLOAT,
			VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		m_previous_depth = create_image(
			input_size.width,
			input_size.height,
			VK_FORMAT_R32_SFLOAT,
			VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		if (!m_output || !m_output->value ||
			!m_dummy_motion_vectors || !m_dummy_motion_vectors->value ||
			!m_dummy_depth || !m_dummy_depth->value)
		{
			dispose_images();
			rsx_log.error("Failed to allocate NVIDIA DLSS Vulkan resources. DLSS will use bilinear fallback.");
			return false;
		}

		if (!m_camera_motion_vectors || !m_camera_motion_vectors->value)
		{
			m_camera_motion_vectors.reset();
			rsx_log.warning("Failed to allocate optional camera-reconstructed motion-vector image; optical flow will remain active.");
		}
		if (!m_geometry_motion_vectors || !m_geometry_motion_vectors->value)
		{
			m_geometry_motion_vectors.reset();
			m_camera_motion_vectors.reset();
			rsx_log.warning("Failed to allocate optional geometry motion-vector image; optical flow will remain active.");
		}
		if (!m_previous_depth || !m_previous_depth->value)
		{
			m_previous_depth.reset();
			rsx_log.warning("Failed to allocate optional previous-depth history; geometry motion will remain disabled.");
		}

		const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
		const VkClearColorValue zero{};
		VkClearColorValue far_depth{};
		far_depth.float32[0] = 1.f;

		m_dummy_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vkCmdClearColorImage(cmd, m_dummy_motion_vectors->value, m_dummy_motion_vectors->current_layout, &zero, 1, &range);
		m_dummy_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		m_dummy_depth->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
		vkCmdClearColorImage(cmd, m_dummy_depth->value, m_dummy_depth->current_layout, &far_depth, 1, &range);
		m_dummy_depth->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		if (m_geometry_motion_vectors)
		{
			m_geometry_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdClearColorImage(cmd, m_geometry_motion_vectors->value, m_geometry_motion_vectors->current_layout, &zero, 1, &range);
			m_geometry_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		if (m_previous_depth)
		{
			m_previous_depth->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdClearColorImage(cmd, m_previous_depth->value, m_previous_depth->current_layout, &far_depth, 1, &range);
			m_previous_depth->change_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		}
		m_previous_depth_valid = false;
		m_previous_depth_frame_id = umax;

		m_input_size = input_size;
		m_output_size = output_size;
		m_input_format = src->format();
		return true;
	}

	bool dlss_upscale_pass::initialize_feature(
		const vk::command_buffer& cmd,
		const size2u& input_size,
		const size2u& output_size,
		bool depth_inverted)
	{
		release_feature();

		NVSDK_NGX_DLSS_Create_Params create_params{};
		create_params.Feature.InWidth = input_size.width;
		create_params.Feature.InHeight = input_size.height;
		create_params.Feature.InTargetWidth = output_size.width;
		create_params.Feature.InTargetHeight = output_size.height;
		create_params.Feature.InPerfQualityValue = select_quality_mode(input_size, output_size);
		create_params.InFeatureCreateFlags =
			NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
			NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
			(depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0);

		NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);
		NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);
		NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);
		NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);
		NVSDK_NGX_Parameter_SetUI(m_parameters, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, NVSDK_NGX_DLSS_Hint_Render_Preset_Default);

		const auto result = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &m_feature, m_parameters, &create_params);
		if (NVSDK_NGX_FAILED(result) || !m_feature)
		{
			rsx_log.error(
				"Failed to create NVIDIA DLSS feature for %ux%u -> %ux%u (0x%x).",
				input_size.width,
				input_size.height,
				output_size.width,
				output_size.height,
				static_cast<u32>(result));
			m_feature = nullptr;
			return false;
		}

		m_feature_depth_inverted = depth_inverted;
		rsx_log.success(
			"NVIDIA DLSS feature created for %ux%u -> %ux%u.",
			input_size.width,
			input_size.height,
			output_size.width,
			output_size.height);
		return true;
	}

	vk::viewable_image* dlss_upscale_pass::run_fallback(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		VkImage present_surface,
		VkImageLayout present_surface_layout,
		const VkImageBlit& request,
		rsx::flags32_t mode,
		const upscaler_frame_data* frame_data)
	{
		return m_fallback.scale_output(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
	}

	vk::viewable_image* dlss_upscale_pass::scale_output(
		const vk::command_buffer& cmd,
		vk::viewable_image* src,
		VkImage present_surface,
		VkImageLayout present_surface_layout,
		const VkImageBlit& request,
		rsx::flags32_t mode,
		const upscaler_frame_data* frame_data)
	{
		if (m_optical_flow && frame_data && frame_data->motion_vectors)
		{
			m_optical_flow->record_conversion(cmd, frame_data->motion_vectors.image);
		}

		const size2u input_size{
			static_cast<u32>(std::abs(request.srcOffsets[1].x - request.srcOffsets[0].x)),
			static_cast<u32>(std::abs(request.srcOffsets[1].y - request.srcOffsets[0].y))};
		const size2u output_size{
			static_cast<u32>(std::abs(request.dstOffsets[1].x - request.dstOffsets[0].x)),
			static_cast<u32>(std::abs(request.dstOffsets[1].y - request.dstOffsets[0].y))};

		if (m_permanently_failed ||
			input_size.width < 32 || input_size.height < 32 ||
			output_size.width < 32 || output_size.height < 32 ||
			input_size.width >= output_size.width || input_size.height >= output_size.height ||
			(mode & UPSCALE_RIGHT_VIEW))
		{
			return run_fallback(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
		}

		vk::viewable_image* depth = m_dummy_depth.get();
		vk::viewable_image* motion_vectors = m_dummy_motion_vectors.get();
		bool valid_depth = false;
		bool valid_motion_vectors = false;

		if (frame_data && frame_data->depth)
		{
			const auto candidate = frame_data->depth.image;
			valid_depth = candidate->samples() == 1 &&
				candidate->width() >= input_size.width && candidate->height() >= input_size.height &&
				(candidate->info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
				depth_format_is_supported(candidate->format());
			if (valid_depth)
				depth = candidate;
		}

		if (frame_data && frame_data->motion_vectors)
		{
			const auto candidate = frame_data->motion_vectors.image;
			valid_motion_vectors = candidate->samples() == 1 &&
				candidate->width() >= input_size.width && candidate->height() >= input_size.height &&
				(candidate->info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) &&
				motion_vector_format_is_supported(candidate->format());
			if (valid_motion_vectors)
				motion_vectors = candidate;
		}

		const bool depth_inverted = valid_depth && frame_data->depth_inverted;
		const bool resources_changed =
			!m_output ||
			!m_dummy_depth || !m_dummy_motion_vectors ||
			m_input_size.width != input_size.width || m_input_size.height != input_size.height ||
			m_output_size.width != output_size.width || m_output_size.height != output_size.height ||
			m_input_format != src->format();
		const bool feature_changed = resources_changed || !m_feature || m_feature_depth_inverted != depth_inverted;
		if (m_feature && m_feature_depth_inverted != depth_inverted)
		{
			m_previous_depth_valid = false;
			m_previous_depth_frame_id = umax;
		}

		if (!initialize_ngx())
		{
			m_permanently_failed = true;
			return run_fallback(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
		}

		if (resources_changed)
		{
			// Retire the old feature before replacing resources that may still be
			// referenced by an earlier NGX evaluation.
			release_feature();
			if (!initialize_images(cmd, src, input_size, output_size))
			{
				m_permanently_failed = true;
				return run_fallback(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
			}
		}

		if (feature_changed && !initialize_feature(cmd, input_size, output_size, depth_inverted))
		{
			m_permanently_failed = true;
			return run_fallback(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
		}

		// Image recreation above invalidates the provisional dummy pointers.
		if (!valid_depth)
			depth = m_dummy_depth.get();
		if (!valid_motion_vectors)
			motion_vectors = m_dummy_motion_vectors.get();

		const bool previous_depth_contiguous =
			m_previous_depth_valid && frame_data &&
			m_previous_depth_frame_id != umax &&
			m_previous_depth_frame_id + 1 == frame_data->frame_id;
		bool geometry_motion_active = false;
		if (m_geometry_motion_pass && frame_data)
		{
			m_geometry_motion_pass->poll_coverage(
				const_cast<vk::command_buffer&>(cmd), frame_data->frame_id);
		}
		const bool has_geometry_packet = frame_data &&
			rsx::character_vertex_probe::take_motion_raster_frame(frame_data->frame_id, m_geometry_motion_frame);
		if (m_geometry_motion_vectors)
		{
			const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
			const VkClearColorValue zero{};
			m_geometry_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
			vkCmdClearColorImage(cmd, m_geometry_motion_vectors->value, m_geometry_motion_vectors->current_layout, &zero, 1, &range);

			if (has_geometry_packet && valid_depth && previous_depth_contiguous &&
				m_previous_depth && frame_data && frame_data->has_camera_matrices &&
				!frame_data->camera_cut_detected && !frame_data->reset_accumulation)
			{
				if (!m_geometry_motion_pass)
				{
					m_geometry_motion_pass = std::make_unique<geometry_motion_vector_pass>();
					m_geometry_motion_pass->create(*vk::get_current_renderer());
				}

				depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
				m_geometry_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
				geometry_motion_active = m_geometry_motion_pass->run(
					const_cast<vk::command_buffer&>(cmd),
					m_geometry_motion_vectors.get(),
					depth,
					m_previous_depth.get(),
					m_geometry_motion_frame);
				depth->pop_layout(cmd);
			}

			vk::insert_image_memory_barrier(
				cmd,
				m_geometry_motion_vectors->value,
				m_geometry_motion_vectors->current_layout,
				VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
				geometry_motion_active ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				geometry_motion_active ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_SHADER_READ_BIT,
				range);
			m_geometry_motion_vectors->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}

		bool camera_motion_active = false;
		if (valid_depth && valid_motion_vectors && frame_data &&
			frame_data->has_camera_matrices && !frame_data->camera_cut_detected && !frame_data->reset_accumulation &&
			m_camera_motion_vectors && m_geometry_motion_vectors)
		{
			if (!m_camera_motion_pass)
				m_camera_motion_pass = std::make_unique<camera_motion_vector_pass>();

			auto* optical_motion_vectors = motion_vectors;
			depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			optical_motion_vectors->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_camera_motion_vectors->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);

			if (m_camera_motion_pass->run(cmd, depth, optical_motion_vectors, m_geometry_motion_vectors.get(), m_camera_motion_vectors.get(), *frame_data))
			{
				vk::insert_image_memory_barrier(
					cmd,
					m_camera_motion_vectors->value,
					m_camera_motion_vectors->current_layout,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,
					make_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
				m_camera_motion_vectors->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				camera_motion_active = true;
			}

			optical_motion_vectors->pop_layout(cmd);
			depth->pop_layout(cmd);
			if (camera_motion_active)
				motion_vectors = m_camera_motion_vectors.get();
		}

		// Retain the exact game depth after all previous-depth consumers have
		// finished. On the next consecutive frame the geometry pass uses it to
		// reject motion vectors that reproject onto another surface. This is kept
		// internal instead of using NGX's legacy BiasCurrentColor input, which the
		// current DLSS integration guide explicitly advises against for new models.
		if (valid_depth && frame_data && m_previous_depth)
		{
			if (!m_depth_history_pass)
				m_depth_history_pass = std::make_unique<depth_history_pass>();

			depth->push_layout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
			m_previous_depth->change_layout(cmd, VK_IMAGE_LAYOUT_GENERAL);
			const bool depth_history_written = m_depth_history_pass->run(cmd, depth, m_previous_depth.get());
			depth->pop_layout(cmd);
			if (depth_history_written)
			{
				vk::insert_image_memory_barrier(
					cmd,
					m_previous_depth->value,
					m_previous_depth->current_layout,
					VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
					VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
					VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
					VK_ACCESS_SHADER_WRITE_BIT,
					VK_ACCESS_SHADER_READ_BIT,
					make_subresource_range(VK_IMAGE_ASPECT_COLOR_BIT));
				m_previous_depth->current_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
				m_previous_depth_valid = true;
				m_previous_depth_frame_id = frame_data->frame_id;
			}
			else
			{
				m_previous_depth_valid = false;
				m_previous_depth_frame_id = umax;
			}
		}
		else
		{
			m_previous_depth_valid = false;
			m_previous_depth_frame_id = umax;
		}

		if ((!valid_depth || !valid_motion_vectors) && !m_synthetic_inputs_reported)
		{
			rsx_log.warning(
				"NVIDIA DLSS is using synthetic %s%s input for this frame; temporal accumulation was reset.",
				valid_depth ? "" : "depth",
				(!valid_depth && !valid_motion_vectors) ? " and motion-vector" : (valid_motion_vectors ? "" : "motion-vector"));
			m_synthetic_inputs_reported = true;
		}
		else if (valid_depth && valid_motion_vectors && !m_real_inputs_reported)
		{
			rsx_log.success("NVIDIA DLSS is using the bound game depth buffer and NVIDIA Optical Flow motion vectors.");
			m_real_inputs_reported = true;
		}

		if (camera_motion_active && !m_camera_motion_reported)
		{
			rsx_log.success("NVIDIA DLSS camera motion reconstruction is active; balanced depth-reprojection fusion is using Optical Flow for residual motion.");
			m_camera_motion_reported = true;
		}
		if (geometry_motion_active && !m_geometry_motion_reported)
		{
			rsx_log.success(
				"NVIDIA DLSS history-validated geometry motion rasterization is active (%u meshes, %llu triangle vertices); current/previous-depth-tested character pixels override Optical Flow.",
				m_geometry_motion_frame.matched_meshes,
				m_geometry_motion_frame.vertices.size());
			m_geometry_motion_reported = true;
		}

		const VkImageAspectFlags depth_aspect = (depth->aspect() & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
		prepare_ngx_input(cmd, src, VK_IMAGE_ASPECT_COLOR_BIT);
		prepare_ngx_input(cmd, depth, depth_aspect);
		prepare_ngx_input(cmd, motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT);
		prepare_ngx_output(cmd, m_output.get());

		auto input_resource = wrap_image(src, VK_IMAGE_ASPECT_COLOR_BIT, false);
		auto output_resource = wrap_image(m_output.get(), VK_IMAGE_ASPECT_COLOR_BIT, true);
		auto depth_resource = wrap_image(depth, depth_aspect, false);
		auto motion_resource = wrap_image(motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT, false);

		NVSDK_NGX_VK_DLSS_Eval_Params eval_params{};
		eval_params.Feature.pInColor = &input_resource;
		eval_params.Feature.pInOutput = &output_resource;
		eval_params.pInDepth = &depth_resource;
		eval_params.pInMotionVectors = &motion_resource;
		eval_params.InJitterOffsetX = frame_data ? frame_data->jitter_x : 0.f;
		eval_params.InJitterOffsetY = frame_data ? frame_data->jitter_y : 0.f;
		eval_params.InReset = (!valid_depth || !valid_motion_vectors || !frame_data ||
			frame_data->reset_accumulation || frame_data->camera_cut_detected) ? 1 : 0;
		eval_params.InMVScaleX = frame_data ? frame_data->motion_vector_scale_x : 1.f;
		eval_params.InMVScaleY = frame_data ? frame_data->motion_vector_scale_y : 1.f;
		eval_params.InColorSubrectBase = {
			static_cast<u32>(std::max(0, std::min(request.srcOffsets[0].x, request.srcOffsets[1].x))),
			static_cast<u32>(std::max(0, std::min(request.srcOffsets[0].y, request.srcOffsets[1].y)))};
		eval_params.InDepthSubrectBase = valid_depth ? eval_params.InColorSubrectBase : NVSDK_NGX_Coordinates{};
		eval_params.InMVSubrectBase = valid_motion_vectors ? eval_params.InColorSubrectBase : NVSDK_NGX_Coordinates{};
		eval_params.InRenderSubrectDimensions = {input_size.width, input_size.height};
		eval_params.InPreExposure = 1.f;
		eval_params.InExposureScale = 1.f;

		const auto result = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, m_feature, m_parameters, &eval_params);

		restore_ngx_input(cmd, motion_vectors, VK_IMAGE_ASPECT_COLOR_BIT);
		restore_ngx_input(cmd, depth, depth_aspect);
		restore_ngx_input(cmd, src, VK_IMAGE_ASPECT_COLOR_BIT);

		if (NVSDK_NGX_FAILED(result))
		{
			rsx_log.error("NVIDIA DLSS evaluation failed (0x%x). Future frames will use bilinear fallback.", static_cast<u32>(result));
			m_permanently_failed = true;
			return run_fallback(cmd, src, present_surface, present_surface_layout, request, mode, frame_data);
		}

		if (mode & UPSCALE_AND_COMMIT)
		{
			ensure(present_surface);
			finish_ngx_output(
				cmd,
				m_output.get(),
				VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_ACCESS_TRANSFER_READ_BIT);

			VkImageBlit output_request = request;
			output_request.srcOffsets[0] = {0, 0, 0};
			output_request.srcOffsets[1] = {
				static_cast<s32>(output_size.width),
				static_cast<s32>(output_size.height),
				1};
			if (request.srcOffsets[0].x > request.srcOffsets[1].x)
				std::swap(output_request.srcOffsets[0].x, output_request.srcOffsets[1].x);
			if (request.srcOffsets[0].y > request.srcOffsets[1].y)
				std::swap(output_request.srcOffsets[0].y, output_request.srcOffsets[1].y);

			vkCmdBlitImage(
				cmd,
				m_output->value,
				m_output->current_layout,
				present_surface,
				present_surface_layout,
				1,
				&output_request,
				VK_FILTER_LINEAR);
			return nullptr;
		}

		finish_ngx_output(
			cmd,
			m_output.get(),
			VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
		return m_output.get();
	}
}
