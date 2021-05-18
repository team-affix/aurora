#pragma once
#include "pch.h"
#include "ntm_rh.h"
#include "ntm_addresser.h"

using aurora::models::ntm_rh;
using aurora::models::ntm_addresser;

namespace aurora {
	namespace models {
		class ntm_reader : public model {
		public:
			size_t memory_height;
			size_t memory_width;

		public:
			tensor mx;
			tensor mx_grad;
			tensor mx_weighted;

		public:
			ptr<ntm_rh> head;
			ptr<ntm_addresser> addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_reader();
			ntm_reader();
			ntm_reader(
				size_t a_memory_height,
				size_t a_memory_width,
				vector<size_t> a_head_h_dims,
				vector<int> a_valid_shifts,
				function<void(ptr<param>&)> a_func);

		};
	}
}