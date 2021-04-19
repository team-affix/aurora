#pragma once
#include "pch.h"
#include "model.h"
#include "sync.h"
#include "att_ts.h"

using aurora::models::model;
using aurora::models::sync;
using aurora::models::att_ts;

namespace aurora {
	namespace models {
		class att {
		public:
			ptr<att_ts> att_ts_template;
			ptr<sync> att_ts_timesteps;

		public:
			ATTENTION_FIELDS
			~att();
			att();


		};
	}
}
