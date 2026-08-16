#include <gtest/gtest.h>

#include "Emu/RSX/Common/texture_replacements.h"

namespace rsx::texture_replacements
{
	namespace
	{
		void write_le32(std::vector<u8>& output, usz offset, u32 value)
		{
			output[offset + 0] = static_cast<u8>(value);
			output[offset + 1] = static_cast<u8>(value >> 8);
			output[offset + 2] = static_cast<u8>(value >> 16);
			output[offset + 3] = static_cast<u8>(value >> 24);
		}

		std::vector<u8> make_dxt1_dds(u32 width, u32 height, u32 mipmaps)
		{
			usz payload_size = 0;
			for (u32 level = 0; level < mipmaps; ++level)
			{
				const u32 level_width = std::max<u32>(1, width >> level);
				const u32 level_height = std::max<u32>(1, height >> level);
				payload_size += std::max<u32>(1, (level_width + 3) / 4) *
					std::max<u32>(1, (level_height + 3) / 4) * 8;
			}

			std::vector<u8> result(128 + payload_size);
			write_le32(result, 0, 0x20534444);
			write_le32(result, 4, 124);
			write_le32(result, 12, height);
			write_le32(result, 16, width);
			write_le32(result, 28, mipmaps);
			write_le32(result, 76, 32);
			write_le32(result, 80, 4);
			write_le32(result, 84, 0x31545844);
			for (usz index = 128; index < result.size(); ++index)
			{
				result[index] = static_cast<u8>(index * 29);
			}
			return result;
		}
	} // namespace

	TEST(TextureReplacements, ValidatesIntegerUpscale)
	{
		EXPECT_TRUE(is_valid_replacement_size(256, 128, 256, 128));
		EXPECT_TRUE(is_valid_replacement_size(256, 128, 1024, 512));
		EXPECT_FALSE(is_valid_replacement_size(256, 128, 512, 512));
		EXPECT_FALSE(is_valid_replacement_size(256, 128, 384, 192));
		EXPECT_FALSE(is_valid_replacement_size(256, 128, 128, 64));
		EXPECT_FALSE(is_valid_replacement_size(256, 128, 4096, 2048));
		EXPECT_FALSE(is_valid_replacement_size(0, 128, 512, 512));
	}

	TEST(TextureReplacements, KeyIncludesDescriptorAndMipContents)
	{
		texture_descriptor descriptor{
			.gcm_format = CELL_GCM_TEXTURE_A8R8G8B8,
			.format_bits = CELL_GCM_TEXTURE_A8R8G8B8 | CELL_GCM_TEXTURE_LN,
			.encoded_remap = 0x1234,
			.width = 2,
			.height = 2,
			.mipmaps = 1,
		};

		image level{.width = 2, .height = 2, .rgba = std::vector<u8>(16, 0x7f)};
		const std::array levels{level};
		const std::string key = make_key(descriptor, levels);
		EXPECT_EQ(key.size(), 40);
		EXPECT_EQ(key, make_key(descriptor, levels));

		level.rgba[0] ^= 1;
		const std::array changed_levels{level};
		EXPECT_NE(key, make_key(descriptor, changed_levels));

		descriptor.encoded_remap ^= 1;
		EXPECT_NE(key, make_key(descriptor, levels));
	}

	TEST(TextureReplacements, DecodesLinearArgbToCanonicalRgba)
	{
		// Little-endian storage below is byte-for-byte PS3 ARGB: A, R, G, B.
		std::array<u32, 4> guest_pixels{
			0x332211ff,
			0x66554480,
			0x99887740,
			0xccbbaa00,
		};

		rsx::subresource_layout layout{
			.data = rsx::io_buffer(guest_pixels.data(), sizeof(guest_pixels)),
			.width_in_texel = 2,
			.height_in_texel = 2,
			.width_in_block = 2,
			.height_in_block = 2,
			.depth = 1,
			.level = 0,
			.layer = 0,
			.pitch_in_block = 2,
		};

		const auto decoded = decode_texture(CELL_GCM_TEXTURE_A8R8G8B8, false, {layout});
		ASSERT_TRUE(decoded);
		ASSERT_EQ(decoded->size(), 1);
		EXPECT_EQ(decoded->front().width, 2);
		EXPECT_EQ(decoded->front().height, 2);
		const std::vector<u8> expected{
			0x11,
			0x22,
			0x33,
			0xff,
			0x44,
			0x55,
			0x66,
			0x80,
			0x77,
			0x88,
			0x99,
			0x40,
			0xaa,
			0xbb,
			0xcc,
			0x00,
		};
		EXPECT_EQ(decoded->front().rgba, expected);
	}

	TEST(TextureReplacements, D8ForcesOpaqueAlpha)
	{
		std::array<u32, 1> guest_pixel{0x33221100};
		rsx::subresource_layout layout{
			.data = rsx::io_buffer(guest_pixel.data(), sizeof(guest_pixel)),
			.width_in_texel = 1,
			.height_in_texel = 1,
			.width_in_block = 1,
			.height_in_block = 1,
			.depth = 1,
			.pitch_in_block = 1,
		};

		const auto decoded = decode_texture(CELL_GCM_TEXTURE_D8R8G8B8, false, {layout});
		ASSERT_TRUE(decoded);
		ASSERT_EQ(decoded->front().rgba.size(), 4);
		EXPECT_EQ(decoded->front().rgba[3], 0xff);
	}

	TEST(TextureReplacements, DecodesDxt1ToCanonicalRgba)
	{
		// A single BC1 block whose first endpoint and all indices select red.
		std::array<u8, 8> guest_block{
			0x00,
			0xf8, // RGB565 red
			0x00,
			0x00,
			0x00,
			0x00,
			0x00,
			0x00,
		};

		rsx::subresource_layout layout{
			.data = rsx::io_buffer(guest_block.data(), guest_block.size()),
			.width_in_texel = 4,
			.height_in_texel = 4,
			.width_in_block = 1,
			.height_in_block = 1,
			.depth = 1,
			.pitch_in_block = 1,
		};

		const auto decoded = decode_texture(CELL_GCM_TEXTURE_COMPRESSED_DXT1, false, {layout});
		ASSERT_TRUE(decoded);
		ASSERT_EQ(decoded->front().rgba.size(), 4 * 4 * 4);
		for (usz i = 0; i < decoded->front().rgba.size(); i += 4)
		{
			EXPECT_EQ(decoded->front().rgba[i + 0], 0xff);
			EXPECT_EQ(decoded->front().rgba[i + 1], 0x00);
			EXPECT_EQ(decoded->front().rgba[i + 2], 0x00);
			EXPECT_EQ(decoded->front().rgba[i + 3], 0xff);
		}
	}

	TEST(TextureReplacements, DdsAndGuestBlocksProduceSameContentKey)
	{
		const std::vector<u8> dds = make_dxt1_dds(8, 4, 2);
		const auto parsed = parse_dds(dds);
		ASSERT_TRUE(parsed);
		ASSERT_EQ(parsed->encoding, replacement_encoding::bc1);
		ASSERT_EQ(parsed->gcm_format, CELL_GCM_TEXTURE_COMPRESSED_DXT1);
		ASSERT_EQ(parsed->levels.size(), 2);
		EXPECT_EQ(parsed->levels[0].data.size(), 16);
		EXPECT_EQ(parsed->levels[1].data.size(), 8);

		const rsx::subresource_layout base{
			.data = rsx::io_buffer(dds.data() + 128, 16),
			.width_in_texel = 8,
			.height_in_texel = 4,
			.width_in_block = 2,
			.height_in_block = 1,
			.depth = 1,
			.level = 0,
			.layer = 0,
			.pitch_in_block = 2,
		};
		const rsx::subresource_layout mip{
			.data = rsx::io_buffer(dds.data() + 144, 8),
			.width_in_texel = 4,
			.height_in_texel = 2,
			.width_in_block = 1,
			.height_in_block = 1,
			.depth = 1,
			.level = 1,
			.layer = 0,
			.pitch_in_block = 1,
		};

		const auto canonical = canonicalize_compressed_texture(
			CELL_GCM_TEXTURE_COMPRESSED_DXT1, false, {base, mip});
		ASSERT_TRUE(canonical);
		EXPECT_EQ(canonical->levels[0].data, parsed->levels[0].data);
		EXPECT_EQ(canonical->levels[1].data, parsed->levels[1].data);
		EXPECT_EQ(make_compressed_key(*canonical), make_compressed_key(*parsed));
		EXPECT_EQ(make_compressed_key(*parsed), "4907125e991f57cf5e5d8615fb37c54c38d5cdb7");
	}

	TEST(TextureReplacements, CompressedIdentityStripsGuestRowPadding)
	{
		std::array<u8, 32> guest{};
		for (u32 index = 0; index < guest.size(); ++index)
		{
			guest[index] = static_cast<u8>(index);
		}

		const rsx::subresource_layout layout{
			.data = rsx::io_buffer(guest.data(), guest.size()),
			.width_in_texel = 4,
			.height_in_texel = 8,
			.width_in_block = 1,
			.height_in_block = 2,
			.depth = 1,
			.level = 0,
			.layer = 0,
			.pitch_in_block = 2,
		};
		const auto canonical = canonicalize_compressed_texture(
			CELL_GCM_TEXTURE_COMPRESSED_DXT1, false, {layout});
		ASSERT_TRUE(canonical);
		const std::vector<u8> expected{
			0, 1, 2, 3, 4, 5, 6, 7,
			16, 17, 18, 19, 20, 21, 22, 23,
		};
		EXPECT_EQ(canonical->levels.front().data, expected);
	}

	TEST(TextureReplacements, DdsParserRejectsTrailingOrIncompatibleData)
	{
		auto dds = make_dxt1_dds(8, 4, 2);
		auto parsed = parse_dds(dds);
		ASSERT_TRUE(parsed);
		EXPECT_TRUE(is_valid_compressed_replacement(
			2, 1, 2, CELL_GCM_TEXTURE_COMPRESSED_DXT1, *parsed));
		EXPECT_FALSE(is_valid_compressed_replacement(
			2, 1, 1, CELL_GCM_TEXTURE_COMPRESSED_DXT1, *parsed));
		EXPECT_FALSE(is_valid_compressed_replacement(
			2, 1, 2, CELL_GCM_TEXTURE_COMPRESSED_DXT45, *parsed));

		dds.push_back(0);
		EXPECT_FALSE(parse_dds(dds));
	}

	TEST(TextureReplacements, ParsesStrictMountedPackIndex)
	{
		const std::string key(40, 'A');
		const std::string valid =
			"RPCS3_TEXTURE_PACK_V1\r\n"
			"# generated index\r\n" + key + "\t1\treplacements/example.dds\r\n";
		const auto parsed = parse_pack_index(valid, "C:/pack");
		ASSERT_TRUE(parsed);
		ASSERT_EQ(parsed->size(), 1);
		EXPECT_EQ(parsed->front().key, std::string(40, 'a'));
		EXPECT_EQ(parsed->front().semantic, 1);
		EXPECT_EQ(parsed->front().path, "C:/pack/replacements/example.dds");
		EXPECT_FALSE(is_pack_entry_enabled(parsed->front(), false));
		EXPECT_TRUE(is_pack_entry_enabled(parsed->front(), true));

		pack_entry color_entry = parsed->front();
		color_entry.semantic = 0;
		EXPECT_TRUE(is_pack_entry_enabled(color_entry, false));

		EXPECT_FALSE(parse_pack_index("RPCS3_TEXTURE_PACK_V1\n" + key + "\t2\tbad.dds\n", "C:/pack"));
		EXPECT_FALSE(parse_pack_index("WRONG_HEADER\n", "C:/pack"));
	}
} // namespace rsx::texture_replacements
