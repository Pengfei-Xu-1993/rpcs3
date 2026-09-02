#pragma once

#include "MFC.h"

namespace ascension::live_probe
{
	constexpr bool force_cpp_mfc_get(u8 command, bool probe_bootstrap_enabled) noexcept
	{
		constexpr u8 modifiers = MFC_BARRIER_MASK | MFC_FENCE_MASK | MFC_RESULT_MASK;
		return probe_bootstrap_enabled && (command & ~modifiers) == MFC_GET_CMD;
	}
} // namespace ascension::live_probe
