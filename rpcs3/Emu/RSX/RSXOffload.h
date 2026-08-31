#pragma once

#include "util/types.hpp"
#include "Utilities/address_range.h"
#include "gcm_enums.h"

#include <array>
#include <vector>

template <typename T>
class named_thread;

namespace rsx
{
	class dma_manager
	{
		enum op
		{
			raw_copy = 0,
			vector_copy = 1,
			index_emulate = 2,
			callback = 3,
			probe_begin_marker = 4,
			probe_end_marker = 5
		};

		struct transport_packet
		{
			op type{};
			std::vector<u8> opt_storage{};
			void* src{};
			void* dst{};
			u32 length{};
			u32 aux_param0{};
			u32 aux_param1{};

			transport_packet(void *_dst, void *_src, u32 len)
				: type(op::raw_copy), src(_src), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, std::vector<u8>& _src, u32 len)
				: type(op::vector_copy), opt_storage(std::move(_src)), dst(_dst), length(len)
			{}

			transport_packet(void *_dst, rsx::primitive_type prim, u32 len)
				: type(op::index_emulate), dst(_dst), length(len), aux_param0(static_cast<u8>(prim))
			{}

			transport_packet(u32 command, void* args)
				: type(op::callback), src(args), aux_param0(command)
			{}

			transport_packet(op marker, u64 epoch)
				: type(marker), aux_param0(static_cast<u32>(epoch)), aux_param1(static_cast<u32>(epoch >> 32))
			{}

			u64 get_probe_epoch() const
			{
				return aux_param0 | (static_cast<u64>(aux_param1) << 32);
			}

			transport_packet(const transport_packet&) = delete;
			transport_packet& operator=(const transport_packet&) = delete;
		};

		struct transport_packet_baseline_layout
		{
			op type{};
			std::vector<u8> opt_storage{};
			void* src{};
			void* dst{};
			u32 length{};
			u32 aux_param0{};
			u32 aux_param1{};
		};

		// Keep probe control metadata inside the existing payload fields so ordinary
		// queue nodes retain the exact pre-probe layout on every supported STL ABI.
		static_assert(sizeof(transport_packet) == sizeof(transport_packet_baseline_layout));

		atomic_t<bool> m_mem_fault_flag = false;

		struct offload_thread;
		std::shared_ptr<named_thread<offload_thread>> m_thread;

		// TODO: Improved benchmarks here; value determined by profiling on a Ryzen CPU, rounded to the nearest 512 bytes
		const u32 max_immediate_transfer_size = 3584;

		mutable atomic_t<bool> m_probe_enabled = false;
		mutable bool m_probe_begin_pending = false;
		mutable bool m_probe_end_pending = false;
		mutable bool m_probe_sequence_open = false;
		mutable bool m_probe_sequence_failed = false;
		mutable u64 m_probe_epoch = 0;
		mutable u64 m_probe_begin_publish_enqueued = 0;
		mutable u64 m_probe_begin_publish_processed = 0;
		mutable u64 m_probe_begin_ack_enqueued = 0;
		mutable u64 m_probe_begin_ack_processed = 0;

		bool is_probe_owner_thread() const;
		void record_probe_raw_copy(u32 length, bool queued, bool queue_was_empty, u32 empty_push_phase) const;
		void record_probe_sync(u64 elapsed_cycles, bool fast) const;

	public:
		enum class probe_worker_phase : u32
		{
			processing = 0,
			prepark = 1,
			wait_intent = 2,
		};

		static constexpr usz probe_worker_phase_count = 3;
		static constexpr usz probe_copy_size_bucket_count = 15;
		static constexpr usz probe_draw_bucket_count = 7;
		static constexpr usz probe_slice_bucket_count = 9;
		static constexpr usz probe_duration_bucket_count = 16;
		static constexpr usz probe_job_type_count = 4;
		static constexpr usz probe_residence_count = 5;

		struct probe_size_bucket
		{
			u64 inline_calls{};
			u64 inline_bytes{};
			u64 queued_calls{};
			u64 queued_bytes{};
			u64 empty_pushes{};
		};

		struct probe_residence_prediction
		{
			u64 intervals{};
			u64 arrivals_within_r{};
			u64 projected_extra_busy_us{};
		};

		struct alignas(64) probe_producer_stats
		{
			u64 raw_calls{};
			u64 raw_bytes{};
			u64 raw_inline_calls{};
			u64 raw_inline_bytes{};
			u64 raw_queued_calls{};
			u64 raw_queued_bytes{};
			u64 queue_was_empty{};
			std::array<u64, probe_worker_phase_count> empty_push_by_worker_phase{};
			u64 empty_push_phase_unknown{};
			std::array<probe_size_bucket, probe_copy_size_bucket_count> size_buckets{};

			u64 persistent_draws{};
			u64 interleaved_blocks{};
			u64 offload_eligible_blocks{};
			u64 offload_eligible_bytes{};
			u64 multi_eligible_draws{};
			u64 zero_or_one_eligible_draws{};
			u64 predicted_batch_jobs_saved{};
			u64 predicted_batch_notifies_saved{};
			u64 incomplete_draws{};
			std::array<u64, probe_draw_bucket_count> blocks_per_draw{};
			std::array<u64, probe_draw_bucket_count> eligible_blocks_per_draw{};
			std::array<u64, probe_draw_bucket_count> empty_pushes_per_draw{};

			// First version is deliberately single-writer: non-RSX-owner sync calls
			// are not sampled rather than contending on atomic statistics.
			u64 sync_calls_rsx{};
			u64 sync_fast_rsx{};
			u64 sync_slow_rsx{};
			u64 sync_slow_cycles_total{};
			u64 sync_slow_cycles_max{};
			std::array<u64, probe_duration_bucket_count> sync_slow_duration{};
		};

		struct alignas(64) probe_worker_stats
		{
			u64 begin_marker_tsc{};
			u64 end_marker_tsc{};
			u64 pop_slices{};
			u64 processed_jobs{};
			u64 processed_raw_calls{};
			u64 processed_raw_bytes{};
			u64 drain_equal_events{};
			u64 spin_hits{};
			u64 wait_calls{};
			u64 wait_returned_immediately{};
			u64 probe_begin_controls{};
			u64 probe_end_controls{};
			std::array<u64, probe_job_type_count> jobs_by_type{};
			std::array<u64, probe_job_type_count> bytes_by_type{};
			std::array<u64, probe_slice_bucket_count> jobs_per_slice{};
			std::array<u64, probe_duration_bucket_count> idle_gap{};
			std::array<probe_residence_prediction, probe_residence_count> residence{};
		};

		struct probe_transport_boundary
		{
			u64 enqueued{};
			u64 processed{};
		};

		struct probe_snapshot
		{
			u64 probe_epoch{};
			bool multithreaded_rsx{};
			u32 immediate_transfer_threshold{};
			u64 marker_window_us{};
			bool snapshot_available{};
			bool transport_complete{};
			bool end_ack_accepted{};
			bool probe_complete{};
			u64 dropped_or_unattributed{};
			u64 transport_window_jobs{};
			u64 worker_type_sum{};
			bool worker_type_sum_matches{};
			bool raw_integrity_matches{};
			bool transport_boundaries_match{};
			probe_transport_boundary begin_publish{};
			probe_transport_boundary begin_ack{};
			probe_transport_boundary end_publish{};
			probe_transport_boundary end_ack{};
			probe_producer_stats producer{};
			probe_worker_stats worker{};
		};

		struct probe_draw_token
		{
			bool active{};
			u64 empty_pushes_before{};
		};

		static_assert(sizeof(probe_producer_stats) + sizeof(probe_worker_stats) < 4096);

		dma_manager() = default;

		// initialization
		void init();

		// General tranport
		void copy(void *dst, std::vector<u8>& src, u32 length) const;
		void copy(void *dst, void *src, u32 length) const;

		// Vertex utilities
		void emulate_as_indexed(void *dst, rsx::primitive_type primitive, u32 count);

		// Renderer callback
		void backend_ctrl(u32 request_code, void* args);

		// Synchronization
		bool is_current_thread() const;
		bool sync() const;
		void join();
		void set_mem_fault_flag();
		void clear_mem_fault_flag();

		// Measurement-only RSX Offloader probe. Begin and end markers are FIFO
		// barriers. The producer gate opens only after begin acknowledgement and
		// closes before end is published. Normal packets intentionally carry no epoch.
		bool probe_begin(u64 epoch);
		bool probe_end(probe_snapshot& snapshot);
		bool probe_is_enabled() const { return m_probe_enabled.observe(); }
		probe_draw_token probe_begin_persistent_draw() const;
		void probe_end_persistent_draw(probe_draw_token token, u32 blocks, u32 eligible_blocks, u64 eligible_bytes, bool complete = true) const;
		u32 probe_immediate_transfer_threshold() const { return max_immediate_transfer_size; }

		// Fault recovery
		utils::address_range32 get_fault_range(bool writing) const;
	};
}
