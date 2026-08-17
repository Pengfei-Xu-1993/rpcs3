#include "stdafx.h"
#include "texture_replacements.h"

#include "Crypto/sha1.h"
#include "Emu/System.h"

#include <stb_image.h>
#include <png.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <charconv>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

LOG_CHANNEL(texture_replacement_log, "TEXREPLACE");

namespace rsx::texture_replacements
{
	namespace
	{
		struct atomic_runtime_statistics
		{
			std::atomic<u32> textures_seen{0};
			std::atomic<u32> eligible_textures{0};
			std::atomic<u32> content_scans{0};
			std::atomic<u32> canonicalized_textures{0};
			std::atomic<u32> pack_matches{0};
			std::atomic<u32> dds_loads{0};
			std::atomic<u32> upload_successes{0};
			std::atomic<u32> upload_failures{0};
		};

		atomic_runtime_statistics g_runtime_statistics;

		runtime_statistics snapshot_runtime_statistics()
		{
			return {
				.textures_seen = g_runtime_statistics.textures_seen.load(std::memory_order_relaxed),
				.eligible_textures = g_runtime_statistics.eligible_textures.load(std::memory_order_relaxed),
				.content_scans = g_runtime_statistics.content_scans.load(std::memory_order_relaxed),
				.canonicalized_textures = g_runtime_statistics.canonicalized_textures.load(std::memory_order_relaxed),
				.pack_matches = g_runtime_statistics.pack_matches.load(std::memory_order_relaxed),
				.dds_loads = g_runtime_statistics.dds_loads.load(std::memory_order_relaxed),
				.upload_successes = g_runtime_statistics.upload_successes.load(std::memory_order_relaxed),
				.upload_failures = g_runtime_statistics.upload_failures.load(std::memory_order_relaxed),
			};
		}

		void log_runtime_statistics(std::string_view reason)
		{
			const auto stats = snapshot_runtime_statistics();
			texture_replacement_log.notice(
				"Runtime %s: seen=%u eligible=%u scanned=%u canonical=%u matches=%u dds=%u uploads=%u upload_failures=%u",
				reason, stats.textures_seen, stats.eligible_textures, stats.content_scans,
				stats.canonicalized_textures, stats.pack_matches, stats.dds_loads,
				stats.upload_successes, stats.upload_failures);
		}

		constexpr u32 max_scale = 8;
		constexpr u32 max_dimension = 16384;
		constexpr u64 max_pixels = 64ull * 1024 * 1024;
		constexpr u64 max_compressed_pixels = 128ull * 1024 * 1024;
		constexpr u64 max_file_size = 256ull * 1024 * 1024;
		constexpr u64 max_index_file_size = 16ull * 1024 * 1024;
		constexpr u64 max_queued_dump_bytes = 256ull * 1024 * 1024;
		constexpr u32 dds_magic = 0x20534444;
		constexpr u32 dds_fourcc = 0x4;
		constexpr u32 dds_dxt1 = 0x31545844;
		constexpr u32 dds_dxt3 = 0x33545844;
		constexpr u32 dds_dxt5 = 0x35545844;
		constexpr std::string_view pack_index_header = "RPCS3_TEXTURE_PACK_V1";

		void hash_bytes(sha1_context& context, const void* data, usz size)
		{
			sha1_update(&context, static_cast<const unsigned char*>(data), size);
		}

		void hash_u32(sha1_context& context, u32 value)
		{
			const std::array<u8, 4> bytes =
				{
					static_cast<u8>(value),
					static_cast<u8>(value >> 8),
					static_cast<u8>(value >> 16),
					static_cast<u8>(value >> 24),
				};
			hash_bytes(context, bytes.data(), bytes.size());
		}

		std::string finish_hash(sha1_context& context)
		{
			std::array<u8, 20> digest{};
			sha1_finish(&context, digest.data());
			constexpr char hex[] = "0123456789abcdef";
			std::string result(40, '0');
			for (usz i = 0; i < digest.size(); ++i)
			{
				result[i * 2] = hex[digest[i] >> 4];
				result[i * 2 + 1] = hex[digest[i] & 0xf];
			}
			return result;
		}

		u32 read_le32(std::span<const u8> data, usz offset)
		{
			return static_cast<u32>(data[offset]) |
				(static_cast<u32>(data[offset + 1]) << 8) |
				(static_cast<u32>(data[offset + 2]) << 16) |
				(static_cast<u32>(data[offset + 3]) << 24);
		}

		bool is_compressed_format(u32 gcm_format)
		{
			return gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT1 ||
				gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT23 ||
				gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT45;
		}

		u32 block_size_for_format(u32 gcm_format)
		{
			return gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT1 ? 8 : 16;
		}

		std::string title_root()
		{
			const std::string& title_id = Emu.GetTitleID();
			if (title_id.empty())
			{
				return {};
			}

			return fs::get_config_dir() + "textures/" + title_id + "/";
		}

		bool write_png(const std::string& path, const image& source)
		{
			if (!fs::create_path(fs::get_parent_dir(path)) && fs::g_tls_error != fs::error::exist)
			{
				return false;
			}

			fs::pending_file pending(path);
			if (!pending.file)
			{
				return false;
			}

			struct png_sink
			{
				fs::file* file = nullptr;
				bool valid = true;
			} sink{&pending.file};

			std::vector<png_bytep> rows(source.height);
			for (u32 y = 0; y < source.height; ++y)
			{
				rows[y] = const_cast<png_bytep>(source.rgba.data() + static_cast<usz>(y) * source.width * 4);
			}

			png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
			if (!png)
			{
				return false;
			}

			png_infop info = png_create_info_struct(png);
			if (!info)
			{
				png_destroy_write_struct(&png, nullptr);
				return false;
			}

			if (setjmp(png_jmpbuf(png)))
			{
				png_destroy_write_struct(&png, &info);
				return false;
			}

			png_set_write_fn(png, &sink, [](png_structp png_ptr, png_bytep data, png_size_t length)
				{
					auto* output = static_cast<png_sink*>(png_get_io_ptr(png_ptr));
					if (!output->file || output->file->write(data, length) != length)
					{
						output->valid = false;
						png_error(png_ptr, "Texture dump write failed");
					}
				},
				[](png_structp) {});

			png_set_IHDR(png, info, source.width, source.height, 8, PNG_COLOR_TYPE_RGBA,
				PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
			png_set_compression_level(png, 3);
			png_write_info(png, info);
			png_write_image(png, rows.data());
			png_write_end(png, info);
			png_destroy_write_struct(&png, &info);

			return sink.valid && (pending.commit(false) || fs::is_file(path));
		}

		class dump_queue
		{
			struct job
			{
				std::string path;
				image source;
			};

			std::mutex m_mutex;
			std::condition_variable m_cv;
			std::deque<job> m_jobs;
			std::unordered_set<std::string> m_seen;
			std::thread m_worker;
			u64 m_queued_bytes = 0;
			bool m_stopping = false;

			void run()
			{
				for (;;)
				{
					job next;
					{
						std::unique_lock lock(m_mutex);
						m_cv.wait(lock, [&]
							{
								return m_stopping || !m_jobs.empty();
							});
						if (m_jobs.empty())
						{
							return;
						}

						next = std::move(m_jobs.front());
						m_jobs.pop_front();
						m_queued_bytes -= next.source.rgba.size();
					}

					if (!write_png(next.path, next.source))
					{
						texture_replacement_log.error("Failed to dump texture to %s", next.path);
					}
				}
			}

		public:
			~dump_queue()
			{
				{
					std::lock_guard lock(m_mutex);
					m_stopping = true;
				}
				m_cv.notify_one();
				if (m_worker.joinable())
				{
					m_worker.join();
				}
			}

			void enqueue(std::string path, image source)
			{
				if (fs::is_file(path))
				{
					return;
				}

				std::lock_guard lock(m_mutex);
				if (m_seen.contains(path) || m_queued_bytes + source.rgba.size() > max_queued_dump_bytes)
				{
					return;
				}

				m_seen.emplace(path);
				m_queued_bytes += source.rgba.size();
				m_jobs.push_back({std::move(path), std::move(source)});
				if (!m_worker.joinable())
				{
					m_worker = std::thread([this]
						{
							run();
						});
				}
				m_cv.notify_one();
			}
		};

		dump_queue& get_dump_queue()
		{
			static dump_queue queue;
			return queue;
		}

		std::optional<replacement_texture> load_png_replacement(const std::string& path, u32 original_width, u32 original_height)
		{
			fs::file file(path);
			if (!file || file.size() == 0 || file.size() > max_file_size)
			{
				return {};
			}

			const std::vector<u8> encoded = file.to_vector<u8>();
			int width = 0;
			int height = 0;
			int components = 0;
			if (!stbi_info_from_memory(encoded.data(), ::narrow<int>(encoded.size()), &width, &height, &components) ||
				width <= 0 || height <= 0 ||
				!is_valid_replacement_size(original_width, original_height, width, height))
			{
				texture_replacement_log.error("Ignoring texture replacement with invalid dimensions: %s", path);
				return {};
			}

			u8* pixels = stbi_load_from_memory(encoded.data(), ::narrow<int>(encoded.size()), &width, &height, &components, 4);
			if (!pixels)
			{
				texture_replacement_log.error("Failed to decode texture replacement: %s", path);
				return {};
			}

			replacement_texture result;
			result.encoding = replacement_encoding::rgba8;
			result.gcm_format = CELL_GCM_TEXTURE_A8R8G8B8;
			replacement_level level;
			level.width = width;
			level.height = height;
			level.data.assign(pixels, pixels + static_cast<usz>(width) * height * 4);
			result.levels.push_back(std::move(level));
			stbi_image_free(pixels);
			return result;
		}

		std::optional<replacement_texture> load_dds_replacement(const std::string& path, const texture_descriptor& descriptor)
		{
			fs::file file(path);
			if (!file || file.size() < 128 || file.size() > max_file_size)
			{
				texture_replacement_log.error("Ignoring missing or oversized DDS replacement: %s", path);
				return {};
			}

			const std::vector<u8> encoded = file.to_vector<u8>();
			auto result = parse_dds(encoded);
			if (!result || !is_valid_compressed_replacement(
				descriptor.width, descriptor.height, descriptor.mipmaps, descriptor.gcm_format, *result))
			{
				texture_replacement_log.error("Ignoring incompatible DDS replacement: %s", path);
				return {};
			}
			// A pack may retain archive-only tail mips that the game does not bind.
			// The content key has already proven that the runtime-visible prefix is
			// the intended source texture, so discard only the unused tail here.
			result->levels.resize(descriptor.mipmaps);
			return result;
		}

		class pack_registry
		{
			std::string m_root;
			std::unordered_map<std::string, pack_entry> m_entries;

			void load(const std::string& root)
			{
				m_root = root;
				m_entries.clear();

				const std::string pack_directory = root + "packs/";
				std::vector<std::string> indexes;
				for (const auto& entry : fs::dir(pack_directory))
				{
					if (!entry.is_directory && entry.name.ends_with(".tsv"))
					{
						indexes.push_back(pack_directory + entry.name);
					}
				}
				std::sort(indexes.begin(), indexes.end());

				u32 normal_count = 0;
				for (const std::string& index_path : indexes)
				{
					fs::file file(index_path);
					if (!file || file.size() == 0 || file.size() > max_index_file_size)
					{
						texture_replacement_log.error("Ignoring invalid texture pack index: %s", index_path);
						continue;
					}

					auto parsed = parse_pack_index(file.to_string(), fs::get_parent_dir(index_path));
					if (!parsed)
					{
						texture_replacement_log.error("Ignoring malformed texture pack index: %s", index_path);
						continue;
					}

					for (pack_entry& item : *parsed)
					{
						if (item.semantic == 1)
						{
							++normal_count;
						}

						if (auto [it, inserted] = m_entries.emplace(item.key, std::move(item)); !inserted)
						{
							texture_replacement_log.warning("Duplicate texture pack key %s; keeping the first mounted entry", it->first);
						}
					}
				}

				if (!indexes.empty())
				{
					texture_replacement_log.notice("Mounted %u texture pack indexes with %u unique entries (%u normal-map candidates disabled by default)",
						::size32(indexes), ::size32(m_entries), normal_count);
				}
			}

		public:
			std::optional<pack_entry> find(const std::string& root, std::string_view key, bool normal_replacement_enabled)
			{
				if (root != m_root)
				{
					load(root);
				}

				const auto found = m_entries.find(std::string(key));
				if (found == m_entries.end() || !is_pack_entry_enabled(found->second, normal_replacement_enabled))
				{
					return {};
				}
				return found->second;
			}
		};

		pack_registry& get_pack_registry()
		{
			static pack_registry registry;
			return registry;
		}
	} // namespace

	void record_runtime_texture(bool eligible)
	{
		const u32 seen = g_runtime_statistics.textures_seen.fetch_add(1, std::memory_order_relaxed) + 1;
		const u32 eligible_count = eligible
			? g_runtime_statistics.eligible_textures.fetch_add(1, std::memory_order_relaxed) + 1
			: g_runtime_statistics.eligible_textures.load(std::memory_order_relaxed);

		if (seen == 1)
		{
			log_runtime_statistics("hook-active");
		}
		else if (eligible_count == 1 && eligible)
		{
			log_runtime_statistics("first-eligible");
		}
		else if ((seen % 262144) == 0)
		{
			log_runtime_statistics("progress");
		}
	}

	void record_runtime_upload(bool success)
	{
		const u32 count = success
			? g_runtime_statistics.upload_successes.fetch_add(1, std::memory_order_relaxed) + 1
			: g_runtime_statistics.upload_failures.fetch_add(1, std::memory_order_relaxed) + 1;

		if (count == 1 || (count % 32) == 0)
		{
			log_runtime_statistics(success ? "upload-success" : "upload-failure");
		}
	}

	runtime_statistics get_runtime_statistics()
	{
		return snapshot_runtime_statistics();
	}

	bool is_supported_format(u32 gcm_format)
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_A8R8G8B8:
		case CELL_GCM_TEXTURE_D8R8G8B8:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
			return true;
		default:
			return false;
		}
	}

	bool is_valid_replacement_size(u32 original_width, u32 original_height, u32 replacement_width, u32 replacement_height)
	{
		if (!original_width || !original_height ||
			!replacement_width || !replacement_height ||
			replacement_width > max_dimension || replacement_height > max_dimension ||
			static_cast<u64>(replacement_width) * replacement_height > max_pixels ||
			replacement_width < original_width || replacement_height < original_height ||
			replacement_width % original_width || replacement_height % original_height)
		{
			return false;
		}

		const u32 scale_x = replacement_width / original_width;
		const u32 scale_y = replacement_height / original_height;
		return scale_x == scale_y && scale_x >= 1 && scale_x <= max_scale;
	}

	std::optional<std::vector<image>> decode_texture(
		u32 gcm_format,
		bool swizzled,
		const std::vector<rsx::subresource_layout>& subresources)
	{
		if (!is_supported_format(gcm_format) || subresources.empty())
		{
			return {};
		}

		std::vector<image> result;
		result.reserve(subresources.size());

		for (const auto& layout : subresources)
		{
			if (layout.layer != 0 || layout.depth != 1 ||
				!is_valid_replacement_size(layout.width_in_texel, layout.height_in_texel, layout.width_in_texel, layout.height_in_texel))
			{
				return {};
			}

			const bool dxt = gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT1 ||
			                 gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT23 ||
			                 gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT45;
			const u32 decoded_width = dxt ? layout.width_in_block * 4 : layout.width_in_texel;
			const u32 decoded_height = dxt ? layout.height_in_block * 4 : layout.height_in_texel;
			std::vector<u8> bgra(static_cast<usz>(decoded_width) * decoded_height * 4);

			rsx::io_buffer output(bgra.data(), bgra.size());
			rsx::texture_uploader_capabilities caps{
				.supports_byteswap = false,
				.supports_vtc_decoding = false,
				.supports_hw_deswizzle = false,
				.supports_zero_copy = false,
				.supports_dxt = false,
				// Software BC decoding writes a full four-texel block row at a
			    // time, so its destination pitch must describe the decoded row.
				.alignment = dxt ? decoded_width * 4 : 4,
			};
			rsx::upload_texture_subresource(output, layout, gcm_format, swizzled, caps);

			image level;
			level.width = layout.width_in_texel;
			level.height = layout.height_in_texel;
			level.rgba.resize(static_cast<usz>(level.width) * level.height * 4);

			for (u32 y = 0; y < level.height; ++y)
			{
				for (u32 x = 0; x < level.width; ++x)
				{
					const usz src = (static_cast<usz>(y) * decoded_width + x) * 4;
					const usz dst = (static_cast<usz>(y) * level.width + x) * 4;
					level.rgba[dst + 0] = bgra[src + 2];
					level.rgba[dst + 1] = bgra[src + 1];
					level.rgba[dst + 2] = bgra[src + 0];
					level.rgba[dst + 3] = gcm_format == CELL_GCM_TEXTURE_D8R8G8B8 ? 0xff : bgra[src + 3];
				}
			}

			result.push_back(std::move(level));
		}

		return result;
	}

	std::optional<replacement_texture> canonicalize_compressed_texture(
		u32 gcm_format,
		bool swizzled,
		const std::vector<rsx::subresource_layout>& subresources)
	{
		if (!is_compressed_format(gcm_format) || subresources.empty() || subresources.size() > 16)
		{
			return {};
		}

		replacement_texture result;
		result.gcm_format = gcm_format;
		result.encoding = gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT1
			? replacement_encoding::bc1
			: (gcm_format == CELL_GCM_TEXTURE_COMPRESSED_DXT23 ? replacement_encoding::bc2 : replacement_encoding::bc3);
		result.levels.reserve(subresources.size());
		const u32 block_size = block_size_for_format(gcm_format);

		for (usz index = 0; index < subresources.size(); ++index)
		{
			const auto& layout = subresources[index];
			const u32 expected_blocks_x = std::max<u32>(1, (layout.width_in_texel + 3) / 4);
			const u32 expected_blocks_y = std::max<u32>(1, (layout.height_in_texel + 3) / 4);
			const u64 source_bytes = static_cast<u64>(layout.pitch_in_block) * layout.height_in_block * block_size;
			const u64 output_bytes = static_cast<u64>(expected_blocks_x) * expected_blocks_y * block_size;
			if (!layout.width_in_texel || !layout.height_in_texel ||
				layout.width_in_texel > max_dimension || layout.height_in_texel > max_dimension ||
				layout.layer != 0 || layout.depth != 1 || layout.level != index ||
				layout.width_in_block != expected_blocks_x || layout.height_in_block != expected_blocks_y ||
				layout.pitch_in_block < expected_blocks_x || source_bytes > layout.data.size() ||
				output_bytes > max_file_size)
			{
				return {};
			}

			replacement_level level;
			level.width = layout.width_in_texel;
			level.height = layout.height_in_texel;
			level.data.resize(static_cast<usz>(output_bytes));

			rsx::io_buffer output(level.data.data(), level.data.size());
			rsx::texture_uploader_capabilities caps{
				.supports_byteswap = false,
				.supports_vtc_decoding = false,
				.supports_hw_deswizzle = false,
				.supports_zero_copy = false,
				.supports_dxt = true,
				.alignment = block_size,
			};
			rsx::upload_texture_subresource(output, layout, gcm_format, swizzled, caps);
			result.levels.push_back(std::move(level));
		}

		return result;
	}

	std::optional<replacement_texture> parse_dds(std::span<const u8> encoded)
	{
		if (encoded.size() < 128 || read_le32(encoded, 0) != dds_magic ||
			read_le32(encoded, 4) != 124 || read_le32(encoded, 76) != 32 ||
			!(read_le32(encoded, 80) & dds_fourcc) || read_le32(encoded, 112) != 0)
		{
			return {};
		}

		const u32 width = read_le32(encoded, 16);
		const u32 height = read_le32(encoded, 12);
		const u32 mipmaps = std::max<u32>(1, read_le32(encoded, 28));
		const u32 fourcc = read_le32(encoded, 84);
		if (!width || !height || width > max_dimension || height > max_dimension ||
			static_cast<u64>(width) * height > max_compressed_pixels || mipmaps > 16)
		{
			return {};
		}

		replacement_texture result;
		switch (fourcc)
		{
		case dds_dxt1:
			result.encoding = replacement_encoding::bc1;
			result.gcm_format = CELL_GCM_TEXTURE_COMPRESSED_DXT1;
			break;
		case dds_dxt3:
			result.encoding = replacement_encoding::bc2;
			result.gcm_format = CELL_GCM_TEXTURE_COMPRESSED_DXT23;
			break;
		case dds_dxt5:
			result.encoding = replacement_encoding::bc3;
			result.gcm_format = CELL_GCM_TEXTURE_COMPRESSED_DXT45;
			break;
		default:
			return {};
		}

		const u32 block_size = block_size_for_format(result.gcm_format);
		usz offset = 128;
		result.levels.reserve(mipmaps);
		for (u32 level_index = 0; level_index < mipmaps; ++level_index)
		{
			const u32 level_width = std::max<u32>(1, width >> level_index);
			const u32 level_height = std::max<u32>(1, height >> level_index);
			const u32 blocks_x = std::max<u32>(1, (level_width + 3) / 4);
			const u32 blocks_y = std::max<u32>(1, (level_height + 3) / 4);
			const u64 level_size_64 = static_cast<u64>(blocks_x) * blocks_y * block_size;
			if (level_size_64 > max_file_size || offset > encoded.size() || level_size_64 > encoded.size() - offset)
			{
				return {};
			}

			const usz level_size = static_cast<usz>(level_size_64);
			replacement_level level;
			level.width = level_width;
			level.height = level_height;
			level.data.assign(encoded.begin() + offset, encoded.begin() + offset + level_size);
			result.levels.push_back(std::move(level));
			offset += level_size;
		}

		if (offset != encoded.size())
		{
			return {};
		}
		return result;
	}

	bool is_valid_compressed_replacement(
		u32 original_width,
		u32 original_height,
		u32 original_mipmaps,
		u32 original_gcm_format,
		const replacement_texture& replacement)
	{
		if (!is_compressed_format(original_gcm_format) || !original_width || !original_height || !original_mipmaps ||
			replacement.gcm_format != original_gcm_format || replacement.levels.size() < original_mipmaps ||
			replacement.levels.empty() || !replacement.width() || !replacement.height() ||
			replacement.width() > max_dimension || replacement.height() > max_dimension ||
			static_cast<u64>(replacement.width()) * replacement.height() > max_compressed_pixels ||
			replacement.width() < original_width || replacement.height() < original_height ||
			replacement.width() % original_width || replacement.height() % original_height)
		{
			return false;
		}

		const u32 scale_x = replacement.width() / original_width;
		const u32 scale_y = replacement.height() / original_height;
		if (scale_x != scale_y || scale_x < 1 || scale_x > max_scale)
		{
			return false;
		}

		for (usz index = 0; index < original_mipmaps; ++index)
		{
			const auto& level = replacement.levels[index];
			const u32 expected_width = std::max<u32>(1, replacement.width() >> index);
			const u32 expected_height = std::max<u32>(1, replacement.height() >> index);
			const u32 blocks_x = std::max<u32>(1, (expected_width + 3) / 4);
			const u32 blocks_y = std::max<u32>(1, (expected_height + 3) / 4);
			const u64 expected_size = static_cast<u64>(blocks_x) * blocks_y * block_size_for_format(replacement.gcm_format);
			if (level.width != expected_width || level.height != expected_height || level.data.size() != expected_size)
			{
				return false;
			}
		}
		return true;
	}

	std::string make_compressed_key(const replacement_texture& texture)
	{
		if (!is_compressed_format(texture.gcm_format) || texture.levels.empty())
		{
			return {};
		}

		sha1_context context{};
		sha1_starts(&context);
		constexpr std::string_view domain = "RPCS3_TEXTURE_PACK_BC_V1";
		hash_bytes(context, domain.data(), domain.size());
		hash_u32(context, texture.gcm_format);
		hash_u32(context, texture.width());
		hash_u32(context, texture.height());
		hash_u32(context, ::size32(texture.levels));
		for (const replacement_level& level : texture.levels)
		{
			hash_u32(context, level.width);
			hash_u32(context, level.height);
			hash_u32(context, ::size32(level.data));
			hash_bytes(context, level.data.data(), level.data.size());
		}
		return finish_hash(context);
	}

	std::optional<std::vector<pack_entry>> parse_pack_index(std::string_view contents, std::string_view index_directory)
	{
		std::vector<pack_entry> result;
		usz cursor = 0;
		bool saw_header = false;
		while (cursor <= contents.size())
		{
			const usz end = contents.find('\n', cursor);
			std::string_view line = contents.substr(cursor, end == std::string_view::npos ? contents.size() - cursor : end - cursor);
			if (!line.empty() && line.back() == '\r')
			{
				line.remove_suffix(1);
			}
			cursor = end == std::string_view::npos ? contents.size() + 1 : end + 1;

			if (!saw_header)
			{
				if (line != pack_index_header)
				{
					return {};
				}
				saw_header = true;
				continue;
			}
			if (line.empty() || line.starts_with('#'))
			{
				continue;
			}

			const usz first_tab = line.find('\t');
			const usz second_tab = first_tab == std::string_view::npos ? first_tab : line.find('\t', first_tab + 1);
			if (first_tab != 40 || second_tab == std::string_view::npos || line.find('\t', second_tab + 1) != std::string_view::npos)
			{
				return {};
			}

			pack_entry item;
			item.key.assign(line.substr(0, first_tab));
			for (char& value : item.key)
			{
				const unsigned char ch = static_cast<unsigned char>(value);
				if (!std::isxdigit(ch))
				{
					return {};
				}
				value = static_cast<char>(std::tolower(ch));
			}

			const std::string_view semantic = line.substr(first_tab + 1, second_tab - first_tab - 1);
			const auto [ptr, error] = std::from_chars(semantic.data(), semantic.data() + semantic.size(), item.semantic);
			if (error != std::errc{} || ptr != semantic.data() + semantic.size() || item.semantic > 1)
			{
				return {};
			}

			item.path.assign(line.substr(second_tab + 1));
			if (item.path.empty())
			{
				return {};
			}
			const bool absolute = item.path.starts_with('/') || item.path.starts_with('\\') ||
				(item.path.size() >= 3 && std::isalpha(static_cast<unsigned char>(item.path[0])) && item.path[1] == ':' &&
					(item.path[2] == '/' || item.path[2] == '\\'));
			if (!absolute)
			{
				item.path = std::string(index_directory) + "/" + item.path;
			}
			result.push_back(std::move(item));
		}

		return saw_header ? std::optional<std::vector<pack_entry>>(std::move(result)) : std::nullopt;
	}

	bool is_pack_entry_enabled(const pack_entry& entry, bool normal_replacement_enabled)
	{
		return entry.semantic != 1 || normal_replacement_enabled;
	}

	std::string make_key(const texture_descriptor& descriptor, std::span<const image> levels)
	{
		sha1_context context{};
		sha1_starts(&context);
		constexpr std::string_view domain = "RPCS3_TEXTURE_REPLACEMENT_V1";
		hash_bytes(context, domain.data(), domain.size());
		hash_u32(context, descriptor.gcm_format);
		hash_u32(context, descriptor.format_bits);
		hash_u32(context, descriptor.format_features);
		hash_u32(context, descriptor.texel_remap_control);
		hash_u32(context, descriptor.encoded_remap);
		hash_u32(context, descriptor.width);
		hash_u32(context, descriptor.height);
		hash_u32(context, descriptor.mipmaps);
		hash_u32(context, descriptor.swizzled);
		hash_u32(context, ::size32(levels));

		for (const image& level : levels)
		{
			hash_u32(context, level.width);
			hash_u32(context, level.height);
			hash_bytes(context, level.rgba.data(), level.rgba.size());
		}

		return finish_hash(context);
	}

	std::string make_filename(const texture_descriptor& descriptor, std::string_view key)
	{
		return fmt::format("v1_%s_%dx%d_f%02x_m%d_r%08x.png",
			key, descriptor.width, descriptor.height, descriptor.gcm_format,
			descriptor.mipmaps, descriptor.encoded_remap);
	}

	std::optional<replacement_texture> process_texture(
		const texture_descriptor& descriptor,
		const std::vector<rsx::subresource_layout>& subresources,
		bool dump_enabled,
		bool replacement_enabled,
		bool normal_replacement_enabled)
	{
		g_runtime_statistics.content_scans.fetch_add(1, std::memory_order_relaxed);

		if ((!dump_enabled && !replacement_enabled) || descriptor.width < 8 || descriptor.height < 8 ||
			!is_valid_replacement_size(descriptor.width, descriptor.height, descriptor.width, descriptor.height))
		{
			return {};
		}

		const std::string root = title_root();
		if (root.empty())
		{
			return {};
		}

		if (replacement_enabled && is_compressed_format(descriptor.gcm_format))
		{
			if (auto canonical = canonicalize_compressed_texture(descriptor.gcm_format, descriptor.swizzled, subresources))
			{
				g_runtime_statistics.canonicalized_textures.fetch_add(1, std::memory_order_relaxed);
				const std::string compressed_key = make_compressed_key(*canonical);
				if (const auto entry = get_pack_registry().find(root, compressed_key, normal_replacement_enabled))
				{
					g_runtime_statistics.pack_matches.fetch_add(1, std::memory_order_relaxed);
					if (auto replacement = load_dds_replacement(entry->path, descriptor))
					{
						g_runtime_statistics.dds_loads.fetch_add(1, std::memory_order_relaxed);
						texture_replacement_log.notice("Loaded %dx%d DDS replacement with %u mips for %dx%d texture %s",
							replacement->width(), replacement->height(), ::size32(replacement->levels),
							descriptor.width, descriptor.height, compressed_key);
						return replacement;
					}
				}
			}
		}

		const bool legacy_replacement_available = replacement_enabled && fs::is_dir(root + "replacements/");
		if (!dump_enabled && !legacy_replacement_available)
		{
			return {};
		}

		auto decoded = decode_texture(descriptor.gcm_format, descriptor.swizzled, subresources);
		if (!decoded || decoded->empty())
		{
			return {};
		}

		const std::string key = make_key(descriptor, *decoded);
		const std::string filename = make_filename(descriptor, key);

		if (dump_enabled)
		{
			get_dump_queue().enqueue(root + "dumps/" + filename, std::move(decoded->front()));
		}

		if (!legacy_replacement_available)
		{
			return {};
		}

		const std::string replacement_path = root + "replacements/" + filename;
		if (!fs::is_file(replacement_path))
		{
			return {};
		}

		auto replacement = load_png_replacement(replacement_path, descriptor.width, descriptor.height);
		if (replacement)
		{
			texture_replacement_log.notice("Loaded %dx%d replacement for %dx%d texture %s",
				replacement->width(), replacement->height(), descriptor.width, descriptor.height, key);
		}
		return replacement;
	}
} // namespace rsx::texture_replacements
