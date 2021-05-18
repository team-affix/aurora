#pragma once
#include "pch.h"
#include "ntm_rh.h"
#include "layer.h"

using aurora::models::ntm_rh;
using aurora::models::layer;

namespace aurora {
	namespace models {
		class ntm_wh : public ntm_rh {
		public:
			// SM, SIZE == units
			tensor e;
			tensor e_grad;
			// LR, SIZE == units
			tensor a;
			tensor a_grad;

		public:
			virtual ~ntm_wh();
			ntm_wh();
			ntm_wh(size_t a_units, vector<size_t> a_h_dims, size_t a_s_units, function<void(ptr<param>&)> a_func);

		public:
			virtual model* clone();
			virtual model* clone(function<void(ptr<param>&)> a_func);
			virtual void compile();

		};
	}
}