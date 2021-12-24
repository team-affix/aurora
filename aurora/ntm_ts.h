#pragma once
#include "affix-base/pch.h"
#include "ntm_reader.h"
#include "ntm_writer.h"

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;
			
		public:
			aurora::maths::tensor mx;
			aurora::maths::tensor mx_grad;
			aurora::maths::tensor my;
			aurora::maths::tensor my_grad;
			aurora::maths::tensor hty_grad;

		public:
			aurora::maths::tensor reader_y_grad;
			aurora::maths::tensor accum_my_grad;

		public:
			aurora::maths::tensor read_wx;
			aurora::maths::tensor read_wx_grad;
			aurora::maths::tensor read_wy;
			aurora::maths::tensor read_wy_grad;
			aurora::maths::tensor write_wx;
			aurora::maths::tensor write_wx_grad;
			aurora::maths::tensor write_wy;
			aurora::maths::tensor write_wy_grad;

		public:
			std::vector<Ntm_reader> internal_readers;
			std::vector<Ntm_writer> internal_writers;

		public:
			MODEL_FIELDS
			virtual ~ntm_ts();
			ntm_ts();
			ntm_ts(
				size_t a_memory_height,
				size_t a_memory_width,
				size_t a_num_readers,
				size_t a_num_writers,
				std::vector<int> a_valid_shifts,
				std::vector<size_t> a_head_hidden_dims);

		};
		typedef affix_base::data::ptr<ntm_ts> Ntm_ts;
	}
}
