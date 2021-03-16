#pragma once
#include "model.h"
#include "lstm_ts.h"

using aurora::models::model;
using aurora::models::lstm_ts;

namespace aurora {
	namespace models {
		class rnc : public model {
		public:
			size_t x_units;
			size_t y_units;
			size_t weak_units;
			size_t strong_units;

		public:
			ptr<model> weak_in;
			ptr<lstm_ts> weak_mid;
			ptr<model> weak_out;

			ptr<model> strong_in;
			ptr<lstm_ts> strong_mid;
			ptr<model> strong_out;

			ptr<model> weak_interface;
			ptr<model> strong_interface;

		public:
			tensor strong_memory;

		public:
			MODEL_FIELDS
			rnc();
			rnc(size_t a_x_units, 
				size_t a_y_units, 
				size_t a_weak_units, 
				size_t a_strong_units, 
				function<void(ptr<param>&)> a_init);
			rnc(size_t a_x_units, 
				size_t a_y_units, 
				size_t a_weak_units, 
				size_t a_strong_units, 
				ptr<model> a_weak_in,
				ptr<lstm_ts> a_weak_mid,
				ptr<model> a_weak_out,
				ptr<model> a_strong_in,
				ptr<lstm_ts> a_strong_mid,
				ptr<model> a_strong_out,
				tensor a_strong_memory);
		};
	}
}