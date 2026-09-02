#pragma once

#include "util/types.hpp"

#include <array>

namespace ascension::live_probe
{
	template <u32 LocalStoreSize, u32 GuestAddressLimit>
	class mfc_get_provenance_table
	{
		static constexpr u32 granule_size = 16;
		static_assert(LocalStoreSize && LocalStoreSize % granule_size == 0);

		struct entry
		{
			u32 ea_bias = 0;
			u32 order = 0;
			u16 valid_mask = 0;
			u16 reserved = 0;
		};
		static_assert(sizeof(entry) == 12);

		std::array<entry, LocalStoreSize / granule_size> m_entries{};
		u64 m_epoch = 0;
		u32 m_order = 0;

		void ensure_epoch(u64 epoch) noexcept
		{
			if (m_epoch == epoch)
				return;

			m_entries.fill({});
			m_epoch = epoch;
			m_order = 0;
		}

		static constexpr u16 mask_for_range(u32 begin, u32 end) noexcept
		{
			const u32 first_bit = begin % granule_size;
			const u32 bit_count = end - begin;
			return static_cast<u16>(((1u << bit_count) - 1) << first_bit);
		}

	public:
		struct match
		{
			u32 guest_ea = 0;
			u32 age = 0;
			bool found = false;
		};

		static constexpr u32 slot_count = LocalStoreSize / granule_size;
		static constexpr usz storage_size = sizeof(entry) * slot_count;

		bool record(u32 lsa, u32 eal, u32 size, u64 epoch) noexcept
		{
			if (!size || lsa >= LocalStoreSize || size > LocalStoreSize - lsa ||
				eal >= GuestAddressLimit || size > GuestAddressLimit - eal)
			{
				return false;
			}

			ensure_epoch(epoch);
			if (m_order == ~u32{})
			{
				// Prefer a conservative loss of history to ambiguous wrapped ages.
				m_entries.fill({});
				m_order = 0;
			}
			const u32 transfer_order = ++m_order;
			const u32 end = lsa + size;
			const u32 ea_bias = eal - lsa;

			for (u32 position = lsa; position < end;)
			{
				const u32 granule_begin = position & ~(granule_size - 1);
				const u32 granule_end = granule_begin + granule_size;
				const u32 chunk_end = end < granule_end ? end : granule_end;
				const u16 new_mask = mask_for_range(position, chunk_end);
				auto& slot = m_entries[granule_begin / granule_size];

				if (!slot.valid_mask || slot.ea_bias != ea_bias)
				{
					// A slot can describe only one linear EA mapping. Replacing a
					// different-bias partial write may lose old provenance, but it
					// cannot fabricate a mixed mapping.
					slot = {ea_bias, transfer_order, new_mask, 0};
				}
				else
				{
					// Preserve the oldest surviving contribution unless this GET
					// overwrote every byte that was previously known in the slot.
					if ((new_mask & slot.valid_mask) == slot.valid_mask)
						slot.order = transfer_order;
					slot.valid_mask |= new_mask;
				}

				position = chunk_end;
			}

			return true;
		}

		match resolve(u32 lsa, u32 size, u64 epoch) noexcept
		{
			if (!size || lsa >= LocalStoreSize || size > LocalStoreSize - lsa)
				return {};

			ensure_epoch(epoch);
			const u32 end = lsa + size;
			u32 expected_bias = 0;
			u32 oldest_age = 0;
			bool have_bias = false;

			for (u32 position = lsa; position < end;)
			{
				const u32 granule_begin = position & ~(granule_size - 1);
				const u32 granule_end = granule_begin + granule_size;
				const u32 chunk_end = end < granule_end ? end : granule_end;
				const u16 required_mask = mask_for_range(position, chunk_end);
				const auto& slot = m_entries[granule_begin / granule_size];

				if ((slot.valid_mask & required_mask) != required_mask ||
					(have_bias && slot.ea_bias != expected_bias))
				{
					return {};
				}

				if (!have_bias)
				{
					expected_bias = slot.ea_bias;
					have_bias = true;
				}

				const u32 age = m_order - slot.order;
				if (age > oldest_age)
					oldest_age = age;
				position = chunk_end;
			}

			const u32 guest_ea = expected_bias + lsa;
			if (guest_ea >= GuestAddressLimit || size > GuestAddressLimit - guest_ea)
				return {};

			return {guest_ea, oldest_age, true};
		}

		u64 epoch() const noexcept
		{
			return m_epoch;
		}

		u32 order() const noexcept
		{
			return m_order;
		}
	};
} // namespace ascension::live_probe
