#pragma once
#include "pch.h"
#include "ntm_reader.h"
#include "ntm_writer.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;
			
		public:
			tensor mx;
			tensor mx_grad;
			tensor my;
			tensor my_grad;
			tensor hty_grad;

		public:
			tensor reader_y_grad;
			tensor accum_my_grad;

		public:
			tensor read_wx;
			tensor read_wx_grad;
			tensor read_wy;
			tensor read_wy_grad;
			tensor write_wx;
			tensor write_wx_grad;
			tensor write_wy;
			tensor write_wy_grad;

		public:
			vector<Ntm_reader> internal_readers;
			vector<Ntm_writer> internal_writers;

		public:
			MODEL_FIELDS
			virtual ~ntm_ts();
			ntm_ts();
			ntm_ts(
				size_t a_memory_height,
				size_t a_memory_width,
				size_t a_num_readers,
				size_t a_num_writers,
				vector<int> a_valid_shifts,
				vector<size_t> a_head_hidden_dims,
				function<void(ptr<param>&)> a_func);

		};
		typedef ptr<ntm_ts> Ntm_ts;
	}
}