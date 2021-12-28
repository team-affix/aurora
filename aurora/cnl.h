#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sync.h"
#include "parameterized_dot_1d.h"

namespace aurora {
	namespace models {
		class cnl : public model
		{
		public:
			size_t m_filter_height = 0;
			size_t m_filter_width = 0;
			size_t m_x_stride = 0;
			size_t m_y_stride = 0;

		public:
			size_t m_max_input_height = 0;
			size_t m_max_input_width = 0;

		public:
			Sync m_filters;

		public:
			MODEL_FIELDS
			virtual ~cnl();
			cnl();
			cnl(
				const size_t& a_filter_height,
				const size_t& a_filter_width,
				const size_t& a_x_stride = 1,
				const size_t& a_y_stride = 1
			);

		public:
			void prep_for_input(
				const size_t& a_input_height,
				const size_t& a_input_width
			);
			void unroll_for_input(
				const size_t& a_input_height,
				const size_t& a_input_width
			);

		public:
			size_t y_strides(
				const size_t& a_input_height
			) const;
			size_t x_strides(
				const size_t& a_input_width
			) const;
			size_t y_strides() const;
			size_t x_strides() const;

		};
		typedef affix_base::data::ptr<cnl> Cnl;
	}
}
