#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"
#include "parameterized_dot_1d.h"

namespace aurora {
	namespace models {
		class cnl : public model
		{
		private:
			aurora::maths::tensor m_y_des;
			aurora::maths::tensor m_y_des_reshaped;

		public:
			size_t m_filter_height = 0;
			size_t m_filter_width = 0;
			size_t m_input_max_height = 0;
			size_t m_input_max_width = 0;
			size_t m_stride_len = 0;
			Parameterized_dot_1d m_filter_template;
			Sync m_filters;

		public:
			ATTENTION_FIELDS
			virtual ~cnl();
			cnl();
			cnl(
				size_t a_filter_height, 
				size_t a_filter_width, 
				size_t a_stride_len
			);

			size_t x_strides(
				size_t a_width
			);
			size_t x_strides();
			size_t y_strides(
				size_t a_height
			);
			size_t y_strides();

		};
		typedef affix_base::data::ptr<cnl> Cnl;
	}
}
