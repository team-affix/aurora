#pragma once
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
			size_t prepared_x_height;
			size_t prepared_x_width;
			size_t unrolled_x_height;
			size_t unrolled_x_width;
			size_t stride_len;
			ptr<model> filter_template;
			ptr<sync> filters;

		public:
			ATTENTION_FIELDS
			virtual ~cnl();
			cnl();
			cnl(size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, function<void(ptr<param>&)> a_init);
			cnl(size_t a_filter_height, size_t a_filter_width, size_t a_stride_len, ptr<model> a_filter_template);

			size_t x_strides(size_t a_width);
			size_t x_strides();
			size_t y_strides(size_t a_height);
			size_t y_strides();

		};
	}
}