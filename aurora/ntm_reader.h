#pragma once
#include "pch.h"
#include "ntm_rh.h"
#include "ntm_addresser.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_reader : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			tensor mx;
			tensor mx_grad;
			tensor wx;
			tensor wx_grad;
			tensor wy;
			tensor wy_grad;

		public:
			ptr<ntm_rh> internal_head;
			ptr<ntm_addresser> internal_addresser;

		public:
			MODEL_FIELDS
			virtual ~ntm_reader();
			ntm_reader();
			ntm_reader(size_t a_memory_height, size_t a_memory_width, vector<int> a_valid_shifts, vector<size_t> a_head_hidden_dims, function<void(ptr<param>&)> a_func);

		};
	}
}