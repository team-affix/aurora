#pragma once
#include "affix-base/pch.h"
#include "lstm.h"
#include "ntm_ts.h"

namespace aurora {
	namespace models {
		class ntm : public recurrent {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			aurora::maths::tensor mx;
			aurora::maths::tensor mx_grad;
			aurora::maths::tensor my;
			aurora::maths::tensor my_grad;

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
			Lstm internal_lstm;
			Ntm_ts ntm_ts_template;
			vector<Ntm_ts> prepared;
			vector<Ntm_ts> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~ntm();
			ntm();
			ntm(
				size_t a_memory_height,
				size_t a_memory_width,
				size_t a_num_readers,
				size_t a_num_writers,
				vector<int> a_valid_shifts,
				vector<size_t> a_head_hidden_dims);

		};
		typedef affix_base::data::ptr<ntm> Ntm;
	}
}
