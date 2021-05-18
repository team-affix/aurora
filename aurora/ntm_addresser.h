#pragma once
#include "model.h"
#include "ntm_rh.h"

using aurora::models::model;

namespace aurora {
	namespace models {
		class ntm_addresser : public model {
		public:
			size_t memory_height = 0;
			size_t memory_width = 0;

		public:
			vector<int> valid_shifts;

		public:
			tensor mx;
			tensor mx_grad;
			// LR, SIZE == units
			tensor k;
			tensor k_grad;
			// LR, SIZE == 1
			tensor beta;
			tensor beta_grad;
			// SM, SIZE == 1
			tensor g;
			tensor g_grad;
			// SM, SIZE == shift_units
			tensor s;
			tensor s_grad;
			// LR, SIZE == 1
			tensor gamma;
			tensor gamma_grad;

			tensor wx;
			tensor wx_grad;
			tensor wy;
			tensor wy_grad;

		protected:
			double key_magnitude = 0;
			tensor memory_magnitude_vector;
			tensor magnitude_product_vector;
			tensor dot_product_vector;
			tensor similar;
			tensor similar_grad;

			tensor sparse;
			tensor sparse_grad;

			double sparse_sum = 0;
			tensor sparse_normalize;
			tensor sparse_normalize_grad;

			tensor interpolate;
			tensor interpolate_grad;

			tensor shift;
			tensor shift_grad;

			tensor sharpen;
			tensor sharpen_grad;

			double sharpen_sum = 0;
			tensor sharp_normalize;
			tensor sharp_normalize_grad;

		public:
			MODEL_FIELDS
			virtual ~ntm_addresser();
			ntm_addresser();
			ntm_addresser(size_t a_height, size_t a_width, vector<int> a_valid_shifts);

		protected:
			void fwd_similar();
			void fwd_sparse();
			void fwd_sparse_normalize();
			void fwd_interpolate();
			void fwd_shift();
			void fwd_sharp();
			void fwd_sharp_normalize();

			void bwd_similar();
			void bwd_sparse();
			void bwd_sparse_normalize();
			void bwd_interpolate();
			void bwd_shift();
			void bwd_sharp();
			void bwd_sharp_normalize();
			int positive_modulo(int i, int n);

		};
	}
}
