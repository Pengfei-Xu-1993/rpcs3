#include "stdafx.h"

#include "Emu/Cell/lv2/sys_rsx.h"
#include "Emu/Memory/vm.h"
#include "Common/BufferUtils.h"
#include "Core/RSXReservationLock.hpp"
#include "RSXOffload.h"
#include "RSXThread.h"

#include "Utilities/lockless.h"

#include <algorithm>
#include <atomic>
#include <thread>
#include "util/asm.hpp"

namespace rsx
{
	namespace
	{
		constexpr std::array<u64, dma_manager::probe_residence_count> s_probe_residence_us{2, 4, 8, 16, 32};

		usz get_copy_size_bucket(u32 size)
		{
			if (size == 0) return 0;
			if (size <= 256) return 1;
			if (size <= 512) return 2;
			if (size <= 1024) return 3;
			if (size <= 2048) return 4;
			if (size <= 3072) return 5;
			if (size <= 3584) return 6;
			if (size <= 4096) return 7;
			if (size <= 6144) return 8;
			if (size <= 8192) return 9;
			if (size <= 12288) return 10;
			if (size <= 16384) return 11;
			if (size <= 32768) return 12;
			if (size <= 65536) return 13;
			return 14;
		}

		usz get_draw_bucket(u64 value)
		{
			if (value <= 4) return static_cast<usz>(value);
			if (value <= 8) return 5;
			return 6;
		}

		usz get_slice_bucket(u64 value)
		{
			if (!value) return umax;
			if (value <= 4) return static_cast<usz>(value - 1);
			if (value <= 8) return 4;
			if (value <= 16) return 5;
			if (value <= 32) return 6;
			if (value <= 64) return 7;
			return 8;
		}

		u64 probe_cycles_to_us(u64 cycles)
		{
			static const u64 frequency = utils::get_tsc_freq();
			return frequency ? (cycles * 1'000'000) / frequency : 0;
		}

		usz get_duration_bucket(u64 elapsed_us)
		{
			if (elapsed_us < 1) return 0;
			if (elapsed_us < 2) return 1;
			if (elapsed_us < 4) return 2;
			if (elapsed_us < 8) return 3;
			if (elapsed_us < 16) return 4;
			if (elapsed_us < 32) return 5;
			if (elapsed_us < 64) return 6;
			if (elapsed_us < 128) return 7;
			if (elapsed_us < 256) return 8;
			if (elapsed_us < 512) return 9;
			if (elapsed_us < 1'000) return 10;
			if (elapsed_us < 2'000) return 11;
			if (elapsed_us < 4'000) return 12;
			if (elapsed_us < 8'000) return 13;
			if (elapsed_us < 16'000) return 14;
			return 15;
		}
	}

	struct dma_manager::offload_thread
	{
		lf_queue<transport_packet> m_work_queue;
		atomic_t<u64> m_enqueued_count = 0;
		atomic_t<u64> m_processed_count = 0;
		transport_packet* m_current_job = nullptr;

		alignas(64) probe_producer_stats m_probe_producer{};
		alignas(64) probe_worker_stats m_probe_worker{};
		alignas(64) atomic_t<u32> m_probe_worker_phase = static_cast<u32>(probe_worker_phase::wait_intent);
		struct alignas(64) probe_ack_state
		{
			atomic_t<u64> epoch = 0;
			atomic_t<u32> status = 0;
		};
		probe_ack_state m_probe_begin_ack{};
		probe_ack_state m_probe_end_ack{};
		bool m_probe_active = false;
		u64 m_probe_epoch = 0;
		u64 m_probe_idle_start_tsc = 0;
		bool m_probe_idle_interval_open = false;

		thread_base* current_thread_ = nullptr;

		void set_probe_worker_phase(probe_worker_phase phase)
		{
			std::atomic_ref<u32>(m_probe_worker_phase.raw()).store(static_cast<u32>(phase), std::memory_order_relaxed);
		}

		void record_probe_idle_gap(u64 elapsed_cycles)
		{
			const u64 elapsed_us = probe_cycles_to_us(elapsed_cycles);
			m_probe_worker.idle_gap[get_duration_bucket(elapsed_us)]++;

			for (usz index = 0; index < s_probe_residence_us.size(); index++)
			{
				const u64 residence_us = s_probe_residence_us[index];
				auto& prediction = m_probe_worker.residence[index];
				prediction.intervals++;
				prediction.arrivals_within_r += (elapsed_us <= residence_us);
				prediction.projected_extra_busy_us +=
					std::min(elapsed_us, residence_us) - std::min(elapsed_us, u64{1});
			}
		}

		void operator ()()
		{
			if (!g_cfg.video.multithreaded_rsx)
			{
				// Abort if disabled
				return;
			}

			current_thread_ = thread_ctrl::get_current();
			ensure(current_thread_);

			if (g_cfg.core.thread_scheduler != thread_scheduler_mode::os)
			{
				thread_ctrl::set_thread_affinity_mask(thread_ctrl::get_affinity_mask(thread_class::rsx));
			}

			while (thread_ctrl::state() != thread_state::aborting)
			{
				u64 measured_jobs_in_slice = 0;
				bool probe_phase_is_processing = false;
				bool probe_began_in_slice = false;

				for (auto&& job : m_work_queue.pop_all())
				{
					m_current_job = &job;

					const auto begin_measured_job = [&]()
					{
						if (!m_probe_active)
						{
							return false;
						}

						if (!probe_phase_is_processing)
						{
							set_probe_worker_phase(probe_worker_phase::processing);
							probe_phase_is_processing = true;
						}
						if (m_probe_idle_interval_open)
						{
							record_probe_idle_gap(utils::get_tsc() - m_probe_idle_start_tsc);
							m_probe_idle_interval_open = false;
						}

						m_probe_worker.processed_jobs++;
						measured_jobs_in_slice++;
						return true;
					};

					switch (job.type)
					{
					case raw_copy:
					{
						if (begin_measured_job())
						{
							m_probe_worker.processed_raw_calls++;
							m_probe_worker.processed_raw_bytes += job.length;
							m_probe_worker.jobs_by_type[raw_copy]++;
							m_probe_worker.bytes_by_type[raw_copy] += job.length;
						}

						const u32 vm_addr = vm::try_get_addr(job.src).first;
						rsx::reservation_lock<true, 1> rsx_lock(vm_addr, job.length, g_cfg.video.strict_rendering_mode && vm_addr);
						std::memcpy(job.dst, job.src, job.length);
						break;
					}
					case vector_copy:
					{
						if (begin_measured_job())
						{
							m_probe_worker.jobs_by_type[vector_copy]++;
							m_probe_worker.bytes_by_type[vector_copy] += job.length;
						}

						std::memcpy(job.dst, job.opt_storage.data(), job.length);
						break;
					}
					case index_emulate:
					{
						if (begin_measured_job())
						{
							m_probe_worker.jobs_by_type[index_emulate]++;
							m_probe_worker.bytes_by_type[index_emulate] +=
								static_cast<u64>(get_index_count(static_cast<rsx::primitive_type>(job.aux_param0), job.length)) * sizeof(u16);
						}

						write_index_array_for_non_indexed_non_native_primitive_to_buffer(static_cast<char*>(job.dst), static_cast<rsx::primitive_type>(job.aux_param0), job.length);
						break;
					}
					case callback:
					{
						if (begin_measured_job())
						{
							m_probe_worker.jobs_by_type[callback]++;
						}

						rsx::get_current_renderer()->renderctl(job.aux_param0, job.src);
						break;
					}
					case gcm_label_write:
					{
						if (begin_measured_job())
						{
							// Keep the existing four-bucket probe transport invariant.
							// This is callback-like control work, not copied payload data.
							m_probe_worker.jobs_by_type[callback]++;
						}

						vm::write<atomic_t<RsxSemaphore>>(job.aux_param0, job.aux_param1);
						break;
					}
					case probe_begin_marker:
					{
						const u64 marker_epoch = job.get_probe_epoch();
						const bool accepted = !m_probe_active && marker_epoch;
						if (accepted)
						{
							m_probe_worker = {};
							m_probe_worker.begin_marker_tsc = utils::get_tsc();
							m_probe_epoch = marker_epoch;
							m_probe_active = true;
							m_probe_idle_interval_open = false;
							m_probe_worker.probe_begin_controls++;
							set_probe_worker_phase(probe_worker_phase::processing);
							probe_phase_is_processing = true;
							probe_began_in_slice = true;
						}
						else
						{
							// A second begin without a completed end invalidates the
							// sequence. The matching end marker will be rejected.
							m_probe_active = false;
							m_probe_epoch = 0;
							m_probe_idle_interval_open = false;
						}

						m_probe_begin_ack.status.store(accepted ? 1u : 2u);
						m_probe_begin_ack.epoch.release(marker_epoch);
						m_probe_begin_ack.epoch.notify_all();
						break;
					}
					case probe_end_marker:
					{
						const u64 marker_epoch = job.get_probe_epoch();
						const bool accepted = m_probe_active && marker_epoch && marker_epoch == m_probe_epoch;

						if (accepted)
						{
							m_probe_worker.end_marker_tsc = utils::get_tsc();
							m_probe_worker.probe_end_controls++;
							const bool ended_measured_slice = measured_jobs_in_slice != 0;
							if (ended_measured_slice)
							{
								m_probe_worker.pop_slices++;
								m_probe_worker.jobs_per_slice[get_slice_bucket(measured_jobs_in_slice)]++;
								measured_jobs_in_slice = 0;
							}

							if (ended_measured_slice && m_enqueued_count.load() == m_processed_count.load() + 1)
							{
								m_probe_worker.drain_equal_events++;
							}
						}

						m_probe_active = false;
						m_probe_epoch = 0;
						m_probe_idle_interval_open = false;

						// Publish the frozen worker counters and acknowledgement before
						// the transport completion that releases probe_end().
						m_probe_end_ack.status.store(accepted ? 1u : 2u);
						m_probe_end_ack.epoch.release(marker_epoch);
						m_probe_end_ack.epoch.notify_all();
						break;
					}
					default: fmt::throw_exception("Unreachable");
					}

					m_processed_count.release(m_processed_count + 1);
				}

				m_current_job = nullptr;

				const bool measured_slice = measured_jobs_in_slice != 0;
				if (measured_slice)
				{
					m_probe_worker.pop_slices++;
					m_probe_worker.jobs_per_slice[get_slice_bucket(measured_jobs_in_slice)]++;
				}

				if (m_enqueued_count.load() == m_processed_count.load())
				{
					if (m_probe_active)
					{
						m_probe_worker.drain_equal_events += measured_slice;
						set_probe_worker_phase(probe_worker_phase::prepark);
						if (!probe_began_in_slice)
						{
							m_probe_idle_start_tsc = utils::get_tsc();
							m_probe_idle_interval_open = true;
						}
					}

					m_processed_count.notify_all();
					utils::spin_on_cacheline_once(m_work_queue.get_wait_atomic(), 0u, 1);

					if (m_probe_active)
					{
						m_probe_worker.spin_hits += static_cast<bool>(m_work_queue);
						m_probe_worker.wait_calls++;
						set_probe_worker_phase(probe_worker_phase::wait_intent);
						const u64 wait_start_tsc = utils::get_tsc();
						thread_ctrl::wait_on(m_work_queue);
						m_probe_worker.wait_returned_immediately +=
							(probe_cycles_to_us(utils::get_tsc() - wait_start_tsc) == 0);
						continue;
					}

					thread_ctrl::wait_on(m_work_queue);
				}
			}

			m_processed_count = -1;
			m_processed_count.notify_all();
		}

		static constexpr auto thread_name = "RSX Offloader"sv;
	};

	// initialization
	void dma_manager::init()
	{
		m_thread = std::make_shared<named_thread<offload_thread>>();
	}

	bool dma_manager::is_probe_owner_thread() const
	{
		if (auto* renderer = get_current_renderer())
		{
			return renderer->is_current_thread();
		}

		return false;
	}

	void dma_manager::record_probe_raw_copy(u32 length, bool queued, bool queue_was_empty, u32 empty_push_phase) const
	{
		auto& producer = m_thread->m_probe_producer;
		auto& bucket = producer.size_buckets[get_copy_size_bucket(length)];

		producer.raw_calls++;
		producer.raw_bytes += length;

		if (!queued)
		{
			producer.raw_inline_calls++;
			producer.raw_inline_bytes += length;
			bucket.inline_calls++;
			bucket.inline_bytes += length;
			return;
		}

		producer.raw_queued_calls++;
		producer.raw_queued_bytes += length;
		bucket.queued_calls++;
		bucket.queued_bytes += length;

		if (queue_was_empty)
		{
			producer.queue_was_empty++;
			bucket.empty_pushes++;
			if (empty_push_phase < probe_worker_phase_count)
			{
				producer.empty_push_by_worker_phase[empty_push_phase]++;
			}
			else
			{
				producer.empty_push_phase_unknown++;
			}
		}
	}

	void dma_manager::record_probe_sync(u64 elapsed_cycles, bool fast, probe_sync_context context) const
	{
		auto& producer = m_thread->m_probe_producer;
		const usz context_index = static_cast<usz>(context);
		ensure(context_index < producer.sync_by_context.size());
		auto& context_stats = producer.sync_by_context[context_index];
		producer.sync_calls_rsx++;
		context_stats.calls++;

		if (fast)
		{
			producer.sync_fast_rsx++;
			context_stats.fast++;
			return;
		}

		const u64 elapsed_us = probe_cycles_to_us(elapsed_cycles);
		producer.sync_slow_rsx++;
		producer.sync_slow_cycles_total += elapsed_cycles;
		producer.sync_slow_cycles_max = std::max(producer.sync_slow_cycles_max, elapsed_cycles);
		producer.sync_slow_duration[get_duration_bucket(elapsed_us)]++;
		context_stats.slow++;
		context_stats.slow_cycles_total += elapsed_cycles;
		context_stats.slow_cycles_max = std::max(context_stats.slow_cycles_max, elapsed_cycles);
		context_stats.slow_us_total += elapsed_us;
		context_stats.slow_duration[get_duration_bucket(elapsed_us)]++;
	}

	void dma_manager::probe_record_gcm_label_branch(probe_gcm_label_branch branch) const
	{
		if (!m_probe_enabled.observe() || !is_probe_owner_thread()) [[likely]]
		{
			return;
		}

		const usz index = static_cast<usz>(branch);
		ensure(index < m_thread->m_probe_producer.gcm_label_branches.size());
		m_thread->m_probe_producer.gcm_label_branches[index]++;
	}

	void dma_manager::probe_record_gcm_label_same_value_early_return() const
	{
		if (m_probe_enabled.observe() && is_probe_owner_thread()) [[unlikely]]
		{
			m_thread->m_probe_producer.gcm_label_same_value_early_returns++;
		}
	}

	// General transport
	void dma_manager::copy(void *dst, std::vector<u8>& src, u32 length) const
	{
		if (length <= max_immediate_transfer_size || !g_cfg.video.multithreaded_rsx)
		{
			std::memcpy(dst, src.data(), length);
		}
		else
		{
			m_thread->m_enqueued_count++;
			m_thread->m_work_queue.push(dst, src, length);
		}
	}

	void dma_manager::copy(void *dst, void *src, u32 length) const
	{
		if (length <= max_immediate_transfer_size || !g_cfg.video.multithreaded_rsx)
		{
			if (m_probe_enabled.observe()) [[unlikely]]
			{
				record_probe_raw_copy(length, false, false, static_cast<u32>(umax));
			}

			const u32 vm_addr = vm::try_get_addr(src).first;
			rsx::reservation_lock<true, 1> rsx_lock(vm_addr, length, g_cfg.video.strict_rendering_mode && vm_addr);
			std::memcpy(dst, src, length);
		}
		else
		{
			const bool probe_enabled = m_probe_enabled.observe();
			m_thread->m_enqueued_count++;

			if (probe_enabled) [[unlikely]]
			{
				const bool queue_was_empty = m_thread->m_work_queue.push<false>(dst, src, length);
				const u32 empty_push_phase = queue_was_empty ? m_thread->m_probe_worker_phase.observe() : static_cast<u32>(umax);
				record_probe_raw_copy(length, true, queue_was_empty, empty_push_phase);

				if (queue_was_empty)
				{
					m_thread->m_work_queue.notify(true);
				}
			}
			else
			{
				// Preserve the original publish+notify primitive when the
				// measurement gate is disabled.
				m_thread->m_work_queue.push(dst, src, length);
			}
		}
	}

	// Vertex utilities
	void dma_manager::emulate_as_indexed(void *dst, rsx::primitive_type primitive, u32 count)
	{
		if (!g_cfg.video.multithreaded_rsx)
		{
			write_index_array_for_non_indexed_non_native_primitive_to_buffer(
				static_cast<char*>(dst), primitive, count);
		}
		else
		{
			m_thread->m_enqueued_count++;
			m_thread->m_work_queue.push(dst, primitive, count);
		}
	}

	// Backend callback
	void dma_manager::backend_ctrl(u32 request_code, void* args)
	{
		ensure(g_cfg.video.multithreaded_rsx);

		m_thread->m_enqueued_count++;
		m_thread->m_work_queue.push(request_code, args);
	}

	// Synchronization
	bool dma_manager::is_current_thread() const
	{
		if (auto cpu = thread_ctrl::get_current())
		{
			return m_thread->current_thread_ == cpu;
		}

		return false;
	}

	bool dma_manager::try_enqueue_ordered_gcm_label_write(u32 address, u32 data) const
	{
		auto& _thr = *m_thread;

		// This experiment is deliberately limited to an already-active FIFO. Strict
		// rendering and fault recovery retain the original producer-side sync/write.
		if (!g_cfg.video.multithreaded_rsx || g_cfg.video.strict_rendering_mode || m_mem_fault_flag ||
			_thr.m_enqueued_count.load() <= _thr.m_processed_count.load())
		{
			return false;
		}

		// A single RSX producer appends this after the raw-copy work it observed.
		// The worker executes the same atomic guest-memory write after earlier FIFO jobs.
		_thr.m_enqueued_count++;
		_thr.m_work_queue.push(op::gcm_label_write, address, data);
		return true;
	}

	bool dma_manager::sync(probe_sync_context context) const
	{
		auto& _thr = *m_thread;
		const u64 probe_epoch = m_probe_enabled.observe() && is_probe_owner_thread() ? m_probe_epoch : 0;

		if (_thr.m_enqueued_count.load() <= _thr.m_processed_count.load()) [[likely]]
		{
			// Nothing to do
			if (probe_epoch) [[unlikely]]
			{
				record_probe_sync(0, true, context);
			}
			return true;
		}

		if (auto rsxthr = get_current_renderer(); rsxthr->is_current_thread())
		{
			if (m_mem_fault_flag)
			{
				// Abort if offloader is in recovery mode
				return false;
			}

			const u64 probe_start_tsc = probe_epoch ? utils::get_tsc() : 0;

			while (_thr.m_enqueued_count.load() > _thr.m_processed_count.load())
			{
				rsxthr->on_semaphore_acquire_wait();
				utils::pause();
			}

			// on_semaphore_acquire_wait() can service a re-entrant flip. If that
			// flip closed this epoch, the interval crosses the measurement boundary
			// and is censored instead of writing producer counters after end ack.
			if (probe_epoch && m_probe_enabled.observe() && m_probe_epoch == probe_epoch) [[unlikely]]
			{
				record_probe_sync(utils::get_tsc() - probe_start_tsc, false, context);
			}
		}
		else
		{
			while (_thr.m_enqueued_count.load() > _thr.m_processed_count.load())
				utils::pause();
		}

		return true;
	}

	void dma_manager::join()
	{
		// An abnormal shutdown must not leave the producer writing counters after
		// the worker has stopped. It cannot create a complete snapshot: callers that
		// need one must execute probe_end() before join().
		m_probe_enabled.release(false);
		if (m_probe_begin_pending || m_probe_end_pending || m_probe_sequence_open)
		{
			m_probe_sequence_failed = true;
			m_probe_sequence_open = false;
		}
		sync();
		*m_thread = thread_state::aborting;
	}

	void dma_manager::set_mem_fault_flag()
	{
		ensure(is_current_thread()); // "Access denied"
		m_mem_fault_flag.release(true);
	}

	void dma_manager::clear_mem_fault_flag()
	{
		ensure(is_current_thread()); // "Access denied"
		m_mem_fault_flag.release(false);
	}

	bool dma_manager::probe_begin(u64 epoch)
	{
		if (!epoch || !g_cfg.video.multithreaded_rsx || !m_thread ||
			m_probe_begin_pending || m_probe_end_pending || m_probe_sequence_open ||
			m_probe_sequence_failed || !is_probe_owner_thread())
		{
			return false;
		}

		m_probe_begin_pending = true;
		m_probe_begin_publish_enqueued = m_thread->m_enqueued_count.load();
		m_probe_begin_publish_processed = m_thread->m_processed_count.load();
		m_thread->m_probe_begin_ack.epoch.store(0);
		m_thread->m_probe_begin_ack.status.store(0);
		m_thread->m_enqueued_count++;
		m_thread->m_work_queue.push(probe_begin_marker, epoch);

		// This wait deliberately uses sync(): on the RSX owner it services
		// on_semaphore_acquire_wait(), so callbacks/local tasks cannot deadlock.
		const bool transport_complete = sync();
		const u64 ack_epoch = m_thread->m_probe_begin_ack.epoch.load();
		const u32 ack = ack_epoch == epoch ? m_thread->m_probe_begin_ack.status.load() : 0;
		m_probe_begin_ack_enqueued = m_thread->m_enqueued_count.load();
		m_probe_begin_ack_processed = m_thread->m_processed_count.load();
		m_probe_begin_pending = false;

		if (!transport_complete || ack != 1)
		{
			// A cleanup marker follows the begin marker in FIFO order.
			// Even if sync() had to return for fault recovery, the worker cannot be
			// left measurement-active when it resumes. Acknowledgements live in the
			// offload thread, and failed sequences never reuse them.
			m_thread->m_probe_end_ack.epoch.store(0);
			m_thread->m_probe_end_ack.status.store(0);
			m_thread->m_enqueued_count++;
			m_thread->m_work_queue.push(probe_end_marker, epoch);
			m_probe_sequence_failed = true;
			return false;
		}

		m_thread->m_probe_producer = {};
		m_probe_epoch = epoch;
		m_probe_sequence_open = true;
		m_probe_enabled.release(true);
		return true;
	}

	bool dma_manager::probe_end(probe_snapshot& snapshot)
	{
		snapshot = {};
		snapshot.probe_epoch = m_probe_epoch;
		snapshot.multithreaded_rsx = g_cfg.video.multithreaded_rsx.get();
		snapshot.immediate_transfer_threshold = max_immediate_transfer_size;
		snapshot.begin_publish = {m_probe_begin_publish_enqueued, m_probe_begin_publish_processed};
		snapshot.begin_ack = {m_probe_begin_ack_enqueued, m_probe_begin_ack_processed};

		if (m_probe_begin_pending || m_probe_end_pending || !m_probe_sequence_open ||
			m_probe_sequence_failed || !m_thread || !is_probe_owner_thread())
		{
			return false;
		}

		// Close the producer gate before publishing the FIFO end marker. Every
		// counted producer packet is therefore ordered before this marker.
		m_probe_enabled.release(false);
		m_probe_sequence_open = false;
		m_probe_end_pending = true;
		snapshot.end_publish = {m_thread->m_enqueued_count.load(), m_thread->m_processed_count.load()};
		m_thread->m_probe_end_ack.epoch.store(0);
		m_thread->m_probe_end_ack.status.store(0);
		m_thread->m_enqueued_count++;
		m_thread->m_work_queue.push(probe_end_marker, m_probe_epoch);

		const bool transport_complete = sync();
		const u64 ack_epoch = m_thread->m_probe_end_ack.epoch.load();
		const u32 ack = ack_epoch == m_probe_epoch ? m_thread->m_probe_end_ack.status.load() : 0;
		snapshot.end_ack = {m_thread->m_enqueued_count.load(), m_thread->m_processed_count.load()};
		snapshot.transport_complete = transport_complete;
		snapshot.end_ack_accepted = ack == 1;
		snapshot.snapshot_available = ack != 0;
		m_probe_end_pending = false;

		if (!snapshot.snapshot_available)
		{
			// Marker packets contain no pointers. A late acknowledgement stays in
			// offload-thread storage and cannot reference returned stack memory.
			m_probe_sequence_failed = true;
			m_probe_epoch = 0;
			return false;
		}

		snapshot.producer = m_thread->m_probe_producer;
		snapshot.worker = m_thread->m_probe_worker;
		if (snapshot.worker.end_marker_tsc >= snapshot.worker.begin_marker_tsc)
		{
			snapshot.marker_window_us = probe_cycles_to_us(snapshot.worker.end_marker_tsc - snapshot.worker.begin_marker_tsc);
		}

		const u64 producer_raw = snapshot.producer.raw_queued_calls;
		const u64 worker_raw = snapshot.worker.processed_raw_calls;
		snapshot.transport_window_jobs = snapshot.end_publish.enqueued >= snapshot.begin_ack.enqueued ?
			snapshot.end_publish.enqueued - snapshot.begin_ack.enqueued : 0;
		for (const u64 count : snapshot.worker.jobs_by_type)
		{
			snapshot.worker_type_sum += count;
		}
		snapshot.worker_type_sum_matches = snapshot.worker_type_sum == snapshot.worker.processed_jobs;
		snapshot.raw_integrity_matches = producer_raw == worker_raw &&
			snapshot.producer.raw_queued_bytes == snapshot.worker.processed_raw_bytes;
		for (const auto& context : snapshot.producer.sync_by_context)
		{
			snapshot.sync_context_sum += context.calls;
		}
		snapshot.sync_context_sum_matches = snapshot.sync_context_sum == snapshot.producer.sync_calls_rsx;
		snapshot.transport_boundaries_match =
			snapshot.begin_ack.enqueued > snapshot.begin_publish.enqueued &&
			snapshot.begin_ack.processed > snapshot.begin_publish.processed &&
			snapshot.end_ack.enqueued > snapshot.end_publish.enqueued &&
			snapshot.end_ack.processed > snapshot.end_publish.processed &&
			snapshot.begin_ack.enqueued == snapshot.begin_ack.processed &&
			snapshot.end_ack.enqueued == snapshot.end_ack.processed &&
			snapshot.transport_window_jobs == snapshot.worker.processed_jobs;

		const auto difference = [](u64 lhs, u64 rhs)
		{
			return lhs > rhs ? lhs - rhs : rhs - lhs;
		};
		snapshot.dropped_or_unattributed =
			difference(producer_raw, worker_raw) +
			difference(snapshot.transport_window_jobs, snapshot.worker.processed_jobs) +
			difference(snapshot.worker_type_sum, snapshot.worker.processed_jobs) +
			difference(snapshot.sync_context_sum, snapshot.producer.sync_calls_rsx) +
			snapshot.producer.empty_push_phase_unknown +
			snapshot.producer.incomplete_draws;

		snapshot.probe_complete = transport_complete && ack == 1 &&
			snapshot.worker.probe_begin_controls == 1 &&
			snapshot.worker.probe_end_controls == 1 &&
			snapshot.transport_boundaries_match &&
			snapshot.worker_type_sum_matches &&
			snapshot.raw_integrity_matches &&
			snapshot.sync_context_sum_matches &&
			snapshot.producer.empty_push_phase_unknown == 0 &&
			snapshot.producer.incomplete_draws == 0;

		m_probe_sequence_failed = !snapshot.probe_complete;
		m_probe_epoch = 0;
		return snapshot.probe_complete;
	}

	dma_manager::probe_draw_token dma_manager::probe_begin_persistent_draw() const
	{
		if (m_probe_enabled.observe()) [[unlikely]]
		{
			return {true, m_thread->m_probe_producer.queue_was_empty};
		}

		return {};
	}

	void dma_manager::probe_end_persistent_draw(probe_draw_token token, u32 blocks, u32 eligible_blocks, u64 eligible_bytes, bool complete) const
	{
		if (!token.active) [[likely]]
		{
			return;
		}

		if (!m_probe_enabled.observe())
		{
			return;
		}

		auto& producer = m_thread->m_probe_producer;
		if (!complete)
		{
			producer.incomplete_draws++;
			return;
		}

		const u64 empty_pushes = producer.queue_was_empty >= token.empty_pushes_before ?
			producer.queue_was_empty - token.empty_pushes_before : 0;

		producer.persistent_draws++;
		producer.interleaved_blocks += blocks;
		producer.offload_eligible_blocks += eligible_blocks;
		producer.offload_eligible_bytes += eligible_bytes;
		producer.multi_eligible_draws += (eligible_blocks > 1);
		producer.zero_or_one_eligible_draws += (eligible_blocks <= 1);
		producer.predicted_batch_jobs_saved += eligible_blocks > 1 ? eligible_blocks - 1 : 0;
		producer.predicted_batch_notifies_saved += empty_pushes > 1 ? empty_pushes - 1 : 0;
		producer.blocks_per_draw[get_draw_bucket(blocks)]++;
		producer.eligible_blocks_per_draw[get_draw_bucket(eligible_blocks)]++;
		producer.empty_pushes_per_draw[get_draw_bucket(empty_pushes)]++;
	}

	// Fault recovery
	utils::address_range32 dma_manager::get_fault_range(bool writing) const
	{
		const auto m_current_job = ensure(m_thread->m_current_job);

		void *address = nullptr;
		u32 range = m_current_job->length;

		switch (m_current_job->type)
		{
		case raw_copy:
			address = (writing) ? m_current_job->dst : m_current_job->src;
			break;
		case vector_copy:
			ensure(writing);
			address = m_current_job->dst;
			break;
		case index_emulate:
			ensure(writing);
			address = m_current_job->dst;
			range = get_index_count(static_cast<rsx::primitive_type>(m_current_job->aux_param0), m_current_job->length);
			break;
		default:
			fmt::throw_exception("Unreachable");
		}

		return utils::address_range32::start_length(vm::get_addr(address), range);
	}
}
