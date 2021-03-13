#pragma once
#include "model.h"
#include "lstm_ts.h"

using aurora::models::model;
using aurora::models::lstm_ts;

namespace aurora {
	namespace models {
		class rnc : public model {
		public:
			size_t units;

		public:
			ptr<lstm_ts> weak_interface;
			ptr<lstm_ts> strong_interface;

		public:
			tensor s;

		public:
			MODEL_FIELDS
			rnc();
			rnc(size_t a_units, function<void(ptr<param>&)> a_init);
			rnc(size_t a_units, ptr<lstm_ts> a_weak_interface, ptr<lstm_ts> a_strong_interface, tensor a_s);
		};
	}
}