#pragma once

#include "util/types.hpp"

#include <atomic>

namespace ascension::live_probe
{
	inline bool try_reserve_event(std::atomic<u64>& accepted_events, u64 maximum) noexcept
	{
		if (!maximum)
		{
			accepted_events.fetch_add(1, std::memory_order_relaxed);
			return true;
		}

		u64 accepted = accepted_events.load(std::memory_order_relaxed);
		while (accepted < maximum)
		{
			if (accepted_events.compare_exchange_weak(accepted, accepted + 1, std::memory_order_relaxed))
				return true;
		}
		return false;
	}
} // namespace ascension::live_probe
