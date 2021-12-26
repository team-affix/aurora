#pragma once
#include "affix-base/pch.h"
#include "affix-base/ptr.h"

namespace aurora {
	namespace maths {

		class tensor {
		public:
			affix_base::data::ptr<double> val_ptr = new double(0);
			affix_base::data::ptr<std::vector<tensor>> vec_ptr = new std::vector<tensor>();
			
		public:
			tensor* group_prev_ptr = nullptr;
			tensor* group_next_ptr = nullptr;

		public:
			double& val();
			double val() const;
			std::vector<tensor>& vec();
			const std::vector<tensor>& vec() const;

		public:
			tensor& group_head();
			tensor& group_tail();
			size_t group_size();

		public:
			virtual ~tensor();
			tensor();
			tensor(const double& a_val);
			tensor(const std::vector<tensor>& a_vec);
			tensor(const std::initializer_list<tensor>& a_il);

		public:
			void set(const tensor& a_other);
			void pop(const tensor& a_other);

		public:
			void resize(size_t a_size);
			
		public:
			static tensor new_1d(size_t a_a);
			static tensor new_1d(size_t a_a, tensor a_val);
			static tensor new_1d(size_t a_a, std::uniform_real_distribution<double>& a_urd, std::default_random_engine& a_re);
			static tensor new_2d(size_t a_a, size_t a_b);
			static tensor new_2d(size_t a_a, size_t a_b, tensor a_val);
			static tensor new_2d(size_t a_a, size_t a_b, std::uniform_real_distribution<double>& a_urd, std::default_random_engine& a_re);

		public:
			tensor up_rank(size_t a_n);
			tensor up_rank();
			tensor down_rank(size_t a_n);
			tensor down_rank();

		public:
			tensor unroll();
			tensor roll(size_t a_width);

		public:
			size_t width() const;
			size_t height() const;

		public:
			tensor row(size_t a_a);
			tensor col(size_t a_a);
			tensor range(size_t a_start, size_t a_len);
			tensor range_2d(size_t a_row, size_t a_col, size_t a_height, size_t a_width);

		public:
			tensor clone_row(size_t a_a) const;
			tensor clone_col(size_t a_a) const;
			tensor clone_range(size_t a_start, size_t a_len) const;

		public:
			void abs_1d(tensor& a_output);
			void abs_2d(tensor& a_output);
			void sum_1d(double& a_output);
			void sum_2d(tensor& a_output);
			void tanh_1d(tensor& a_output);
			void tanh_2d(tensor& a_output);
			void norm_1d(tensor& a_output);
			void norm_2d(tensor& a_output);
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
			double max();
			double min();
			int arg_max();
			int arg_min();
			tensor norm_1d();
			tensor norm_2d();

		public:
			void add_1d(const tensor& a_other, tensor& a_output);
			void sub_1d(const tensor& a_other, tensor& a_output);
			void mul_1d(const tensor& a_other, tensor& a_output);
			void div_1d(const tensor& a_other, tensor& a_output);
			void pow_1d(const tensor& a_other, tensor& a_output);
			void dot_1d(const tensor& a_other, double& a_output);
			void add_2d(const tensor& a_other, tensor& a_output);
			void sub_2d(const tensor& a_other, tensor& a_output);
			void mul_2d(const tensor& a_other, tensor& a_output);
			void div_2d(const tensor& a_other, tensor& a_output);
			void pow_2d(const tensor& a_other, tensor& a_output);
			void dot_2d(const tensor& a_other, tensor& a_output);
			void cat(tensor& a_other, tensor& a_output);

		public:
			tensor add_1d(const tensor& a_other);
			tensor sub_1d(const tensor& a_other);
			tensor mul_1d(const tensor& a_other);
			tensor div_1d(const tensor& a_other);
			tensor pow_1d(const tensor& a_other);
			double dot_1d(const tensor& a_other);
			tensor add_2d(const tensor& a_other);
			tensor sub_2d(const tensor& a_other);
			tensor mul_2d(const tensor& a_other);
			tensor div_2d(const tensor& a_other);
			tensor pow_2d(const tensor& a_other);
			tensor dot_2d(const tensor& a_other);
			double cos_sim(tensor& a_other);
			tensor cat(tensor& a_other);

		public:
			void link(tensor& a_other);
			void unlink();

		public:
			void rank_recur(const std::function<void(tensor*)>& a_func);

		public:
			void group_recur_fwd(const std::function<void(tensor*)>& a_func);
			void group_recur_bwd(const std::function<void(tensor*)>& a_func);
			void group_recur(const std::function<void(tensor*)>& a_func);

		public:
			bool group_contains(const tensor* a_ptr);
			void group_add(tensor& a_other);
			void group_remove(tensor& a_other);
			void group_join(tensor& a_other);
			void group_leave();
			void group_disband();
			void group_add_all_ranks(tensor& a_other);
			void group_remove_all_ranks(tensor& a_other);
			void group_join_all_ranks(tensor& a_other);
			void group_leave_all_ranks();
			void group_disband_all_ranks();

		public:
			tensor clone() const;
			tensor link();

		public:
			std::string to_string();

		public:
			void clear();
			size_t size() const;
			tensor& at(size_t a_a);
			const tensor& at(size_t a_a) const;

		public:
			operator double& ();
			operator const double& () const;
			tensor& operator[](size_t a_a);
			const tensor& operator[](size_t a_a) const;

		};
	}
}
