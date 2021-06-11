#pragma once
#include "pch.h"
#include "model.h"
#include "sync.h"

using aurora::models::model;
using aurora::models::sync;

namespace aurora {
	namespace models {
		class cnl : public model {
		private:
			tensor y_des;
			tensor y_des_reshaped;

		public:
			size_t filter_height;
			size_t filter_width;
			size_t input_max_height;
			size_t input_max_width;
			size_t stride_len;
			Model filter_template;
			Sync filters;

		public:
			ATTENTION_FIELDS
			virtual ~cnl();
			cnl();
			cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, function<void(Param&)> a_func);
			cnl(size_t a_input_max_height, size_t a_input_max_width, size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, Model a_filter_template);

			size_t x_strides(size_t a_width);
			size_t x_strides();
			size_t y_strides(size_t a_height);
			size_t y_strides();

		};
		typedef ptr<cnl> Cnl;
	}
}