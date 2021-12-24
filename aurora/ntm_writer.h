#pragma once
#include "affix-base/pch.h"
#include "ntm_wh.h"
#include "ntm_addresser.h"

namespace aurora {
	namespace models {
		class ntm_writer : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			aurora::maths::tensor weighted_erase_compliment;

		public:
			aurora::maths::tensor mx;
			aurora::maths::tensor mx_grad;
			aurora::maths::tensor wx;
			aurora::maths::tensor wx_grad;
			aurora::maths::tensor wy;
			aurora::maths::tensor wy_grad;

		public:
			Ntm_wh internal_head;
			Ntm_addresser internal_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_writer();
			ntm_writer();
			ntm_writer(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts, vector<size_t> a_head_hidden_dims);

		};
		typedef affix_base::data::ptr<ntm_writer> Ntm_writer;
	}
}
