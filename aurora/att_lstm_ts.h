#pragma once
#include "pch.h"
#include "model.h"
#include "sync.h"
#include "pseudo.h"

using aurora::models::model;
using aurora::models::sync;

namespace aurora {
	namespace models {
		class att_lstm_ts : public model {
		public:
			size_t units;

		public:
			tensor htx;
			tensor htx_grad;

		public:
			Model model_template;
			Sync models;

		public:
			RECURRENT_FIELDS
			virtual ~att_lstm_ts();
			att_lstm_ts();
			att_lstm_ts(size_t a_units, vector<size_t> a_h_dims, function<void(ptr<param>&)> a_func);

		};
		typedef ptr<att_lstm_ts> Att_lstm_ts;
	}
}