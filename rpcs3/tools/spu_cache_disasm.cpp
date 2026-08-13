#include "Emu/Cell/SPUDisAsm.h"
#include "Emu/Cell/SPUThread.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
	u16 read_be16(const std::vector<u8>& data, usz offset)
	{
		return static_cast<u16>((data[offset] << 8) | data[offset + 1]);
	}

	u32 read_be32(const std::vector<u8>& data, usz offset)
	{
		return (static_cast<u32>(data[offset]) << 24) |
			(static_cast<u32>(data[offset + 1]) << 16) |
			(static_cast<u32>(data[offset + 2]) << 8) |
			static_cast<u32>(data[offset + 3]);
	}

	void disassemble_range(const std::vector<u8>& local_store, u32 start, u32 end, u32 target_pc = umax)
	{
		start &= ~3u;
		end = std::min<u32>(end & ~3u, SPU_LS_SIZE);
		if (start >= end || local_store.size() < SPU_LS_SIZE)
			return;

		SPUDisAsm disassembler(cpu_disasm_mode::dump, local_store.data());
		for (u32 pc = start; pc < end; pc += 4)
		{
			disassembler.disasm(pc);
			std::cout << (pc == target_pc ? ">>> " : "    ") << disassembler.last_opcode;
		}
	}
}

int main(int argc, char** argv)
{
	if (argc >= 4 && std::string{argv[1]} == "--ls")
	{
		std::ifstream input(argv[2], std::ios::binary);
		if (!input)
		{
			std::cerr << "cannot open local-store image\n";
			return 3;
		}

		std::vector<u8> local_store((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
		if (local_store.size() != SPU_LS_SIZE)
		{
			std::cerr << "local-store image must be exactly 0x40000 bytes\n";
			return 5;
		}

		const u32 start = static_cast<u32>(std::stoul(argv[3], nullptr, 16));
		const u32 end = argc >= 5
			? static_cast<u32>(std::stoul(argv[4], nullptr, 16))
			: std::min<u32>(SPU_LS_SIZE, start + 0x100);
		disassemble_range(local_store, start, end, start);
		return 0;
	}

	if (argc < 3)
	{
		std::cerr << "usage:\n"
			"  spu_cache_disasm <spu-cache.dat> <pc-hex>\n"
			"  spu_cache_disasm --ls <256KiB-ls.bin> <start-hex> [end-hex]\n";
		return 2;
	}

	std::ifstream input(argv[1], std::ios::binary);
	if (!input)
	{
		std::cerr << "cannot open cache\n";
		return 3;
	}

	const u32 target_pc = static_cast<u32>(std::stoul(argv[2], nullptr, 16));
	std::vector<u8> cache((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
	usz cursor = 0;
	u32 record_index = 0;
	u32 match_count = 0;

	while (cursor + 8 <= cache.size())
	{
		const u16 crc = read_be16(cache, cursor);
		const u16 word_count = read_be16(cache, cursor + 2);
		const u32 address = read_be32(cache, cursor + 4);
		const usz byte_count = static_cast<usz>(word_count) * 4;
		const usz data_offset = cursor + 8;
		if (data_offset + byte_count > cache.size())
			break;

		if (word_count && target_pc >= address && static_cast<u64>(target_pc) < static_cast<u64>(address) + byte_count)
		{
			++match_count;
			std::vector<u8> local_store(SPU_LS_SIZE);
			std::copy_n(cache.data() + data_offset, byte_count, local_store.data() + address);

			std::cout << "record=" << record_index
				<< " crc=0x" << std::hex << crc
				<< " entry=0x" << address
				<< " end=0x" << (address + byte_count)
				<< " target=0x" << target_pc << std::dec << "\n";

			disassemble_range(local_store, address, address + static_cast<u32>(byte_count), target_pc);
		}

		cursor = data_offset + byte_count;
		++record_index;
	}

	std::cout << "matches=" << match_count << "\n";
	return match_count ? 0 : 4;
}
