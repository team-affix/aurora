#pragma once
#include "pch.h"
#include "model.h"
#include "lstm_ts.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_ts : public model {
		public:
			size_t units;
			vector<int> valid_shifts;

		public:
			tensor mtx;
			tensor mtxg;
			tensor mty;
			tensor mtyg;
			tensor wtx;
			tensor wtxg;
			tensor wty;
			tensor wtyg;

		public:
			vector<ptr<model>> read_heads;
			vector<ptr<model>> write_heads;

		protected:
			virtual void address();
			virtual void address_content();
			virtual void address_interpolate();
			virtual void address_shift();
			virtual void address_sharpen();

			
		public:
			MODEL_FIELDS
			virtual ~ntm_ts();
			ntm_ts();

		};
	}
}