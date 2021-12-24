#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"

namespace aurora {
	namespace models {
		class cnl : public model {
		private:
			aurora::maths::tensor y_des;
			aurora::maths::tensor y_des_reshaped;

		public:
			size_t filter_height = 0;
			size_t filter_width = 0;
			size_t input_max_height = 0;
			size_t input_max_width = 0;
			size_t stride_len = 0;
			Model filter_template;
			Sync filters;

		public:
			ATTENTION_FIELDS
			virtual ~cnl();
			cnl();
			cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len);
			cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, Model a_filter_template);

			size_t x_strides(size_t a_width);
			size_t x_strides();
			size_t y_strides(size_t a_height);
			size_t y_strides();

		};
		typedef affix_base::data::ptr<cnl> Cnl;
	}
}
