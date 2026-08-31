#pragma once

#include <util/types.hpp>
#include "Emu/RSX/RSXThread.h"

#include "context_accessors.define.h"

namespace rsx
{
	void mm_flush_lazy();
	void mm_flush();

	namespace util
	{
		template <bool FlushDMA, bool FlushPipe>
		static void write_gcm_label(context* ctx, u32 type, u32 address, u32 data)
		{
			const bool is_flip_sema = (address == (RSX(ctx)->label_addr + 0x10) || address == (RSX(ctx)->device_addr + 0x30));
			if (!is_flip_sema)
			{
				// First, queue the GPU work. If it flushes the queue for us, the following routines will be faster.
				auto& dma = g_fxo->get<rsx::dma_manager>();
				const bool supports_host_gpu_labels = RSX(ctx)->get_backend_config().supports_host_gpu_labels;
				if (!supports_host_gpu_labels)
				{
					dma.probe_record_gcm_label_branch(dma_manager::probe_gcm_label_branch::unsupported);
				}
				const bool handled = supports_host_gpu_labels && RSX(ctx)->release_GCM_label(type, address, data);

				if (vm::_ref<RsxSemaphore>(address) == data)
				{
					dma.probe_record_gcm_label_same_value_early_return();
					// It's a no-op to write the same value (although there is a delay in real-hw so it's more accurate to allow GPU label in this case)
					return;
				}

				if constexpr (FlushDMA || FlushPipe)
				{
					// Release op must be acoompanied by MM flush.
					// FlushPipe implicitly does a MM flush but FlushDMA does not. Trigger the flush here
					rsx::mm_flush();

					if constexpr (FlushDMA)
					{
						if constexpr (!FlushPipe)
						{
							if (type == NV4097_TEXTURE_READ_SEMAPHORE_RELEASE && !handled &&
								dma.try_enqueue_ordered_gcm_label_write(address, data))
							{
								return;
							}
						}

						// If the backend handled the request, this call will basically be a NOP
						const auto context = type == NV4097_TEXTURE_READ_SEMAPHORE_RELEASE
							? (handled ? dma_manager::probe_sync_context::gcm_texture_handled : dma_manager::probe_sync_context::gcm_texture_fallback)
							: (handled ? dma_manager::probe_sync_context::gcm_backend_handled : dma_manager::probe_sync_context::gcm_backend_fallback);
						dma.sync(context);
					}

					if constexpr (FlushPipe)
					{
						// Syncronization point, may be associated with memory changes without actually changing addresses
						RSX(ctx)->m_graphics_state |= rsx::pipeline_state::fragment_program_needs_rehash;

						// Manually flush the pipeline.
						// It is possible to stream report writes using the host GPU, but that generates too much submit traffic.
						RSX(ctx)->sync();
					}
				}

				if (handled)
				{
					// Backend will handle it, nothing to write.
					return;
				}
			}

			vm::write<atomic_t<RsxSemaphore>>(address, data);
		}
	}
}

#include "context_accessors.undef.h"
