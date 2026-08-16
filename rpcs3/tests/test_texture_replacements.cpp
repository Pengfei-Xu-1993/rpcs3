#include <gtest/gtest.h>

#include "Emu/RSX/Common/texture_replacements.h"

namespace rsx::texture_replacements
{
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
} // namespace rsx::texture_replacements
