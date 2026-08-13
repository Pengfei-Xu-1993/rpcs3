#include "Emu/Cell/AscensionMfcProvenance.h"

#include <iostream>

namespace
{
	using table_type = ascension::live_probe::mfc_get_provenance_table<0x400, 0x10000>;
	static_assert(ascension::live_probe::mfc_get_provenance_table<0x40000, 0xe0000000>::storage_size == 196608);

	int failures = 0;

	void check(bool condition, int line)
	{
		if (!condition)
		{
			std::cerr << "check failed at line " << line << '\n';
			++failures;
		}
	}

#define CHECK(condition) check((condition), __LINE__)

	void test_basic_range_and_no_unrelated_eviction()
	{
		table_type table;
		CHECK(!table.resolve(0x100, 1, 1).found);
		CHECK(table.record(0x100, 0x1000, 0x10, 1));
		CHECK(table.resolve(0x100, 0x10, 1).guest_ea == 0x1000);
		CHECK(table.resolve(0x108, 1, 1).guest_ea == 0x1008);
		CHECK(table.resolve(0x10f, 1, 1).guest_ea == 0x100f);
		CHECK(!table.resolve(0x110, 1, 1).found);

		for (u32 index = 0; index < 129; ++index)
		{
			const u32 lsa = 0x180 + (index % 32) * 16;
			CHECK(table.record(lsa, 0x4000 + index * 16, 16, 1));
		}

		const auto original = table.resolve(0x100, 0x10, 1);
		CHECK(original.found);
		CHECK(original.guest_ea == 0x1000);
		CHECK(original.age == 129);
	}

	void test_partial_and_cross_granule_updates()
	{
		table_type table;
		CHECK(table.record(0x200, 0x2000, 8, 7));
		CHECK(table.record(0x208, 0x2008, 8, 7));
		const auto merged = table.resolve(0x200, 16, 7);
		CHECK(merged.found);
		CHECK(merged.guest_ea == 0x2000);
		CHECK(merged.age == 1);
		CHECK(table.record(0x200, 0x2000, 16, 7));
		CHECK(table.resolve(0x200, 16, 7).age == 0);

		CHECK(table.record(0x20c, 0x5000, 4, 7));
		const auto replacement = table.resolve(0x20c, 4, 7);
		CHECK(replacement.found);
		CHECK(replacement.guest_ea == 0x5000);
		CHECK(replacement.age == 0);
		CHECK(!table.resolve(0x200, 4, 7).found);
		CHECK(!table.resolve(0x200, 16, 7).found);

		CHECK(table.record(0x30c, 0x700c, 24, 7));
		const auto spanning = table.resolve(0x30c, 24, 7);
		CHECK(spanning.found);
		CHECK(spanning.guest_ea == 0x700c);
	}

	void test_boundaries_epoch_and_zero_ea()
	{
		table_type table;
		CHECK(!table.record(0, 0, 0, 1));
		CHECK(table.order() == 0);
		CHECK(table.record(0, 0, 1, 1));
		const auto zero = table.resolve(0, 1, 1);
		CHECK(zero.found);
		CHECK(zero.guest_ea == 0);
		CHECK(table.record(0x100, 0, 16, 1));
		CHECK(table.resolve(0x100, 16, 1).guest_ea == 0);

		CHECK(table.record(0x3ff, 0xffff, 1, 1));
		CHECK(table.resolve(0x3ff, 1, 1).guest_ea == 0xffff);
		const u32 order = table.order();
		CHECK(!table.record(0x400, 0x1000, 1, 1));
		CHECK(!table.record(0x3ff, 0xffff, 2, 1));
		CHECK(!table.record(0x100, 0x10000, 1, 1));
		CHECK(!table.record(0x100, 0xffff, 2, 1));
		CHECK(table.order() == order);

		CHECK(!table.resolve(0, 1, 2).found);
		CHECK(table.epoch() == 2);
		CHECK(table.order() == 0);
		CHECK(table.record(0x40, 0x1040, 16, 2));
		CHECK(table.resolve(0x40, 16, 2).age == 0);
	}
} // namespace

int main()
{
	test_basic_range_and_no_unrelated_eviction();
	test_partial_and_cross_granule_updates();
	test_boundaries_epoch_and_zero_ea();
	if (failures)
		return 1;
	std::cout << "ascension MFC provenance table tests passed\n";
	return 0;
}
