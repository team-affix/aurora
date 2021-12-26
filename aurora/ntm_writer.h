#pragma once
#include "affix-base/pch.h"
#include "ntm_wh.h"
#include "ntm_addresser.h"

namespace aurora {
	namespace models {
		class ntm_writer : public model {
		public:
			size_t m_memory_height = 0;
			size_t m_memory_width = 0;

		public:
			aurora::maths::tensor m_weighted_erase_compliment;

		public:
			aurora::maths::tensor m_mx;
			aurora::maths::tensor m_mx_grad;
			aurora::maths::tensor m_wx;
			aurora::maths::tensor m_wx_grad;
			aurora::maths::tensor m_wy;
			aurora::maths::tensor m_wy_grad;

		public:
			Ntm_wh m_internal_head;
			Ntm_addresser m_internal_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_writer();
			ntm_writer();
			ntm_writer(
				size_t a_memory_height,
				size_t a_memory_width,
				std::vector<int> a_valid_shifts,
				std::vector<size_t> a_head_hidden_dims
			);

		};
		typedef affix_base::data::ptr<ntm_writer> Ntm_writer;
	}
}
