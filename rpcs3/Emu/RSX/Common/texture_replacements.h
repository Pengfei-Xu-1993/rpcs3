#pragma once

#include "TextureUtils.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
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

	enum class replacement_encoding : u8
	{
		rgba8,
		bc1,
		bc2,
		bc3,
	};

	struct replacement_level
	{
		u32 width = 0;
		u32 height = 0;
		std::vector<u8> data;
	};

	struct replacement_texture
	{
		replacement_encoding encoding = replacement_encoding::rgba8;
		u32 gcm_format = CELL_GCM_TEXTURE_A8R8G8B8;
		std::vector<replacement_level> levels;

		u32 width() const
		{
			return levels.empty() ? 0 : levels.front().width;
		}

		u32 height() const
		{
			return levels.empty() ? 0 : levels.front().height;
		}
	};

	struct pack_entry
	{
		std::string key;
		std::string path;
		u32 semantic = 0;
	};

	struct runtime_statistics
	{
		u32 textures_seen = 0;
		u32 eligible_textures = 0;
		u32 content_scans = 0;
		u32 canonicalized_textures = 0;
		u32 pack_matches = 0;
		u32 dds_loads = 0;
		u32 upload_successes = 0;
		u32 upload_failures = 0;
	};

	constexpr bool is_early_scan_candidate(
		bool backend_supported,
		bool feature_enabled,
		bool compressed,
		bool texture_2d,
		bool single_layer,
		bool borderless,
		bool format_supported)
	{
		return backend_supported && feature_enabled && compressed && texture_2d &&
			single_layer && borderless && format_supported;
	}

	bool is_supported_format(u32 gcm_format);
	bool is_valid_replacement_size(u32 original_width, u32 original_height, u32 replacement_width, u32 replacement_height);

	std::optional<std::vector<image>> decode_texture(
		u32 gcm_format,
		bool swizzled,
		const std::vector<rsx::subresource_layout>& subresources);

	std::optional<replacement_texture> canonicalize_compressed_texture(
		u32 gcm_format,
		bool swizzled,
		const std::vector<rsx::subresource_layout>& subresources);

	std::optional<replacement_texture> parse_dds(std::span<const u8> encoded);
	bool is_valid_compressed_replacement(
		u32 original_width,
		u32 original_height,
		u32 original_mipmaps,
		u32 original_gcm_format,
		const replacement_texture& replacement);

	std::string make_compressed_key(const replacement_texture& texture);
	std::optional<std::vector<pack_entry>> parse_pack_index(std::string_view contents, std::string_view index_directory);
	bool is_pack_entry_enabled(const pack_entry& entry, bool normal_replacement_enabled);
	void record_runtime_texture(bool eligible);
	void record_runtime_upload(bool success);
	runtime_statistics get_runtime_statistics();

	std::string make_key(const texture_descriptor& descriptor, std::span<const image> levels);
	std::string make_filename(const texture_descriptor& descriptor, std::string_view key);

	// Decodes and identifies a guest texture, queues an optional PNG dump, and
	// returns a validated external replacement when one is present.
	std::optional<replacement_texture> process_texture(
		const texture_descriptor& descriptor,
		const std::vector<rsx::subresource_layout>& subresources,
		bool dump_enabled,
		bool replacement_enabled,
		bool normal_replacement_enabled);
} // namespace rsx::texture_replacements
