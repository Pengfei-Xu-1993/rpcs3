#pragma once

#include "TextureUtils.h"

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rsx::texture_replacements
{
	struct texture_descriptor
	{
		u32 gcm_format = 0;
		u32 format_bits = 0;
		u32 format_features = 0;
		u32 texel_remap_control = 0;
		u32 encoded_remap = 0;
		u16 width = 0;
		u16 height = 0;
		u16 mipmaps = 0;
		bool swizzled = false;
	};

	struct image
	{
		u32 width = 0;
		u32 height = 0;
		std::vector<u8> rgba;
	};

	bool is_supported_format(u32 gcm_format);
	bool is_valid_replacement_size(u32 original_width, u32 original_height, u32 replacement_width, u32 replacement_height);

	std::optional<std::vector<image>> decode_texture(
		u32 gcm_format,
		bool swizzled,
		const std::vector<rsx::subresource_layout>& subresources);

	std::string make_key(const texture_descriptor& descriptor, std::span<const image> levels);
	std::string make_filename(const texture_descriptor& descriptor, std::string_view key);

	// Decodes and identifies a guest texture, queues an optional PNG dump, and
	// returns a validated external replacement when one is present.
	std::optional<image> process_texture(
		const texture_descriptor& descriptor,
		const std::vector<rsx::subresource_layout>& subresources,
		bool dump_enabled,
		bool replacement_enabled);
} // namespace rsx::texture_replacements
