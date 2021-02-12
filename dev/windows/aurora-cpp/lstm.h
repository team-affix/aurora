#pragma once
#include "data.h"
#include "model.h"
#include "lstm_ts.h"

using aurora::data::ptr;
using aurora::modeling::model;
using aurora::modeling::lstm_ts;

namespace aurora {
	namespace modeling {
		class lstm : public model {
		public:
			size_t units;

		public:
			tensor ctx;
			tensor cty;
			tensor htx;
			tensor hty;
			tensor ctx_grad;
			tensor cty_grad;
			tensor htx_grad;
			tensor hty_grad;

		public:
			ptr<lstm_ts> lstm_ts_template;
			vector<ptr<lstm_ts>> prepared;
			vector<ptr<lstm_ts>> unrolled;

		public:
			RECURRENT_FIELDS
			virtual ~lstm();
			lstm();
			lstm(size_t a_units, function<void(ptr<param>&)> a_init);
			lstm(size_t a_units, ptr<lstm_ts> a_lstm_ts_template);

		};
	}
}