#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"

namespace aurora {
	namespace maths {

		class tensor {
		public:
			affix_base::data::ptr<double> m_val_ptr = new double(0);
			affix_base::data::ptr<std::vector<tensor>> m_vec_ptr = new std::vector<tensor>();

		public:
			double& val();
			double val() const;
			std::vector<tensor>& vec();
			const std::vector<tensor>& vec() const;

		public:
			virtual ~tensor();

		public:
			tensor();
			tensor(
				const double& a_val
			);
			tensor(
				const std::vector<tensor>& a_vec
			);
			tensor(
				const std::initializer_list<tensor>& a_il
			);

		public:
			void set(
				const tensor& a_other
			);
			void pop(
				const tensor& a_other
			);

		public:
			void resize(
				size_t a_size
			);
			
		public:
			static tensor new_1d(
				size_t a_a
			);
			static tensor new_1d(
				size_t a_a,
				tensor a_val
			);
			static tensor new_1d(
				size_t a_a,
				std::uniform_real_distribution<double>& a_urd,
				std::default_random_engine& a_re
			);
			static tensor new_2d(
				size_t a_a,
				size_t a_b
			);
			static tensor new_2d(
				size_t a_a,
				size_t a_b,
				tensor a_val
			);
			static tensor new_2d(
				size_t a_a,
				size_t a_b,
				std::uniform_real_distribution<double>& a_urd,
				std::default_random_engine& a_re
			);

		public:
			tensor up_rank(
				size_t a_n
			);
			tensor up_rank();
			tensor down_rank(
				size_t a_n
			);
			tensor down_rank();

		public:
			tensor unroll();
			tensor roll(
				size_t a_width
			);

		public:
			size_t width() const;
			size_t height() const;

		public:
			tensor row(
				size_t a_a
			);
			tensor col(
				size_t a_a
			);
			tensor range(
				size_t a_start,
				size_t a_len
			);
			tensor range_2d(
				size_t a_row,
				size_t a_col,
				size_t a_height,
				size_t a_width
			);

		public:
			tensor clone_row(
				size_t a_a
			) const;
			tensor clone_col(
				size_t a_a
			) const;
			tensor clone_range(
				size_t a_start,
				size_t a_len
			) const;

		public:
			void abs_1d(
				tensor& a_output
			);
			void abs_2d(
				tensor& a_output
			);
			void sum_1d(
				double& a_output
			);
			void sum_2d(
				tensor& a_output
			);
			void tanh_1d(
				tensor& a_output
			);
			void tanh_2d(
				tensor& a_output
			);
			void norm_1d(
				tensor& a_output
			);
			void signed_norm_1d(
				tensor& a_output
			);
			void norm_2d(
				tensor& a_output
			);
			void zero_1d();
			void zero_2d();
			void zero();

		public:
			tensor abs_1d();
			tensor abs_2d();
			double sum_1d();
			tensor sum_2d();
			tensor tanh_1d();
			tensor tanh_2d();
			double mag_1d();
			double max_1d();
			double min_1d();
			size_t arg_max_1d();
			size_t arg_min_1d();
			double max_2d();
			double min_2d();
			tensor norm_1d();
			tensor signed_norm_1d();
			tensor norm_2d();

		public:
			void add_1d(
				const tensor& a_other,
				tensor& a_output
			);
			void sub_1d(
				const tensor& a_other,
				tensor& a_output
			);
			void mul_1d(
				const tensor& a_other,
				tensor& a_output
			);
			void div_1d(
				const tensor& a_other,
				tensor& a_output
			);
			void pow_1d(
				const tensor& a_other,
				tensor& a_output
			);
			void dot_1d(
				const tensor& a_other,
				double& a_output
			);
			void add_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void sub_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void mul_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void div_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void pow_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void dot_2d(
				const tensor& a_other,
				tensor& a_output
			);
			void cat(
				tensor& a_other,
				tensor& a_output
			);

		public:
			tensor add_1d(
				const tensor& a_other
			);
			tensor sub_1d(
				const tensor& a_other
			);
			tensor mul_1d(
				const tensor& a_other
			);
			tensor div_1d(
				const tensor& a_other
			);
			tensor pow_1d(
				const tensor& a_other
			);
			double dot_1d(
				const tensor& a_other
			);
			tensor add_2d(
				const tensor& a_other
			);
			tensor sub_2d(
				const tensor& a_other
			);
			tensor mul_2d(
				const tensor& a_other
			);
			tensor div_2d(
				const tensor& a_other
			);
			tensor pow_2d(
				const tensor& a_other
			);
			tensor dot_2d(
				const tensor& a_other
			);
			double cos_sim(
				tensor& a_other
			);
			tensor cat(
				tensor& a_other
			);

		public:
			void link(
				tensor& a_other
			);
			void unlink();
			
		public:
			void group_link(
				tensor& a_other
			);
			void group_unlink();

		public:
			void rank_recur(
				const std::function<void(tensor*)>& a_func
			);
			void lowest_rank_recur(
				const std::function<void(tensor*)>& a_func
			);
			size_t lowest_rank_count();

		public:
			tensor clone() const;
			tensor group_link();

		public:
			std::string to_string() const;

		public:
			void clear();
			size_t size() const;
			tensor& at(
				size_t a_a
			);
			const tensor& at(
				size_t a_a
			) const;

		public:
			operator double& ();
			operator const double& () const;
			tensor& operator[](
				size_t a_a
			);
			const tensor& operator[](
				size_t a_a
			) const;

		};
	}
}
