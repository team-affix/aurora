#pragma once
#include "affix-base/pch.h"
#include "lstm.h"
#include "ntm_ts.h"

namespace aurora {
	namespace models {
		class ntm : public recurrent {
		public:
			size_t m_memory_height = 0;
			size_t m_memory_width = 0;

		public:
			aurora::maths::tensor m_mx;
			aurora::maths::tensor m_mx_grad;
			aurora::maths::tensor m_my;
			aurora::maths::tensor m_my_grad;

		public:
			aurora::maths::tensor m_read_wx;
			aurora::maths::tensor m_read_wx_grad;
			aurora::maths::tensor m_read_wy;
			aurora::maths::tensor m_read_wy_grad;
			aurora::maths::tensor m_write_wx;
			aurora::maths::tensor m_write_wx_grad;
			aurora::maths::tensor m_write_wy;
			aurora::maths::tensor m_write_wy_grad;

		public:
			Lstm m_internal_lstm;
			Ntm_ts m_ntm_ts_template;
			std::vector<Ntm_ts> m_prepared;
			std::vector<Ntm_ts> m_unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~ntm();
			ntm();
			ntm(
				size_t a_memory_height,
				size_t a_memory_width,
				size_t a_num_readers,
				size_t a_num_writers,
				std::vector<int> a_valid_shifts,
				std::vector<size_t> a_head_hidden_dims
			);

		};
		typedef affix_base::data::ptr<ntm> Ntm;
	}
}
