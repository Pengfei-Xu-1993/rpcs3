#include "stdafx.h"
#include "texture_replacements.h"

#include "Crypto/sha1.h"
#include "Emu/System.h"

#include <stb_image.h>
#include <png.h>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_set>

LOG_CHANNEL(texture_replacement_log, "TEXREPLACE");

namespace rsx::texture_replacements
{
	namespace
	{
		constexpr u32 max_scale = 8;
		constexpr u32 max_dimension = 16384;
		constexpr u64 max_pixels = 64ull * 1024 * 1024;
		constexpr u64 max_file_size = 256ull * 1024 * 1024;
		constexpr u64 max_queued_dump_bytes = 256ull * 1024 * 1024;

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

		std::optional<image> load_replacement(const std::string& path, u32 original_width, u32 original_height)
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

			image result;
			result.width = width;
			result.height = height;
			result.rgba.assign(pixels, pixels + static_cast<usz>(width) * height * 4);
			stbi_image_free(pixels);
			return result;
		}
	} // namespace

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

	std::string make_filename(const texture_descriptor& descriptor, std::string_view key)
	{
		return fmt::format("v1_%s_%dx%d_f%02x_m%d_r%08x.png",
			key, descriptor.width, descriptor.height, descriptor.gcm_format,
			descriptor.mipmaps, descriptor.encoded_remap);
	}

	std::optional<image> process_texture(
		const texture_descriptor& descriptor,
		const std::vector<rsx::subresource_layout>& subresources,
		bool dump_enabled,
		bool replacement_enabled)
	{
		if ((!dump_enabled && !replacement_enabled) || descriptor.width < 8 || descriptor.height < 8 ||
			!is_valid_replacement_size(descriptor.width, descriptor.height, descriptor.width, descriptor.height))
		{
			return {};
		}

		auto decoded = decode_texture(descriptor.gcm_format, descriptor.swizzled, subresources);
		if (!decoded || decoded->empty())
		{
			return {};
		}

		const std::string root = title_root();
		if (root.empty())
		{
			return {};
		}

		const std::string key = make_key(descriptor, *decoded);
		const std::string filename = make_filename(descriptor, key);

		if (dump_enabled)
		{
			get_dump_queue().enqueue(root + "dumps/" + filename, std::move(decoded->front()));
		}

		if (!replacement_enabled)
		{
			return {};
		}

		const std::string replacement_path = root + "replacements/" + filename;
		if (!fs::is_file(replacement_path))
		{
			return {};
		}

		auto replacement = load_replacement(replacement_path, descriptor.width, descriptor.height);
		if (replacement)
		{
			texture_replacement_log.notice("Loaded %dx%d replacement for %dx%d texture %s",
				replacement->width, replacement->height, descriptor.width, descriptor.height, key);
		}
		return replacement;
	}
} // namespace rsx::texture_replacements
