#pragma once
#include "ntm_wh.h"
#include "ntm_addresser.h"

using aurora::models::ntm_wh;
using aurora::models::ntm_addresser;

namespace aurora {
	namespace models {
		class ntm_writer : public model {
		public:
			size_t memory_height;
			size_t memory_width;

		public:
			tensor mx;
			tensor mx_grad;

		public:
			ptr<ntm_wh> head;
			ptr<ntm_addresser> addresser;
			
		public:
			MODEL_FIELDS
			virtual ~ntm_writer();
			ntm_writer();
			ntm_writer(
				size_t a_memory_height,
				size_t a_memory_width,
				vector<size_t> a_head_h_dims,
				vector<int> a_valid_shifts,
				function<void(ptr<param>&)> a_func);

		};
	}
}
