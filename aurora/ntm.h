#pragma once
#include "pch.h"
#include "lstm.h"
#include "ntm_ts.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			tensor mx;
			tensor mx_grad;
			tensor my;
			tensor my_grad;

		public:
			ptr<lstm> internal_lstm;
			ptr<ntm_ts> ntm_ts_template;
			vector<ptr<ntm_ts>> prepared;
			vector<ptr<ntm_ts>> unrolled;

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
				vector<size_t> a_head_hidden_dims,
				function<void(ptr<param>&)> a_func);

		};
	}
}