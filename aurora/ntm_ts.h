#pragma once
#include "affix-base/pch.h"
#include "ntm_reader.h"
#include "ntm_writer.h"

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t m_memory_height = 0;
			size_t m_memory_width = 0;
			
		public:
			aurora::maths::tensor m_mx;
			aurora::maths::tensor m_mx_grad;
			aurora::maths::tensor m_my;
			aurora::maths::tensor m_my_grad;
			aurora::maths::tensor m_hty_grad;

		public:
			aurora::maths::tensor m_reader_y_grad;
			aurora::maths::tensor m_accum_my_grad;

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
			std::vector<Ntm_reader> m_internal_readers;
			std::vector<Ntm_writer> m_internal_writers;

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
				std::vector<size_t> a_head_hidden_dims
			);

		};
		typedef affix_base::data::ptr<ntm_ts> Ntm_ts;
	}
}
