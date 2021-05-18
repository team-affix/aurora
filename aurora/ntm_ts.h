#pragma once
#include "pch.h"
#include "model.h"
#include "ntm_reader.h"
#include "ntm_writer.h"
#include "ntm_addresser.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			tensor mx;
			tensor mx_grad;
			tensor my;
			tensor my_grad;

		public:
			vector<ptr<ntm_reader>> readers;
			vector<ptr<ntm_writer>> writers;

		public:
			MODEL_FIELDS
			virtual ~ntm_ts();
			ntm_ts();
			ntm_ts(
				size_t a_memory_height,
				size_t a_memory_width,
				size_t a_num_reads,
				size_t a_num_writes,
				vector<size_t> a_head_h_dims,
				vector<int> a_valid_shifts,
				function<void(ptr<param>&)> a_func);

		};
	}
}