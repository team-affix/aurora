#pragma once
#include "pch.h"
#include "model.h"
#include "sync.h"
#include "att_ts.h"
#include "lstm.h"

using aurora::models::model;
using aurora::models::sync;
using aurora::models::att_ts;
using aurora::models::lstm;

namespace aurora {
	namespace models {
		class att : public model {
		public:
			size_t units;

		public:
			ptr<att_ts> att_ts_template;
			ptr<sync> models;
			ptr<lstm> internal_lstm;

		public:
			ATTENTION_FIELDS
			~att();
			att();
			att(size_t a_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func);


		};
	}
}
