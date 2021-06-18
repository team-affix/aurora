#pragma once
#include "pch.h"
#include "recurrent.h"

using aurora::models::recurrent;

namespace aurora {
	namespace models {
		class stacked_recurrent : public recurrent {
		public:
			size_t prepared_size = 0;
			size_t unrolled_size = 0;

		public:
			vector<Recurrent> models;

		public:
			RECURRENT_FIELDS
			virtual ~stacked_recurrent();
			stacked_recurrent();
			stacked_recurrent(vector<Recurrent> a_models);

		};
		typedef ptr<stacked_recurrent> Stacked_recurrent;
	}
}
