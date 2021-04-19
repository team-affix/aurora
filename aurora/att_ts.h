#pragma once
#include "pch.h"
#include "model.h"
#include "sync.h"
#include "pseudo.h"

using aurora::models::model;
using aurora::models::sync;

namespace aurora {
	namespace models {
		class att_ts : public model {
		public:
			size_t xt_units;
			size_t ht_units;
			tensor htx;
			tensor htx_grad;

		public:
			ptr<model> model_template;
			ptr<sync> models;

		public:
			RECURRENT_FIELDS
			virtual ~att_ts();
			att_ts();
			att_ts(size_t a_xt_units, size_t a_ht_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func);

		};
	}
}