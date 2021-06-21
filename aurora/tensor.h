#pragma once
#include "pch.h"
#include "ptr.h"
#include "ref.h"

using affix_base::data::ptr;
using affix_base::data::ref;
using std::vector;
using std::initializer_list;
using std::back_inserter;
using std::string;
using std::stringstream;
using std::function;
using std::uniform_real_distribution;
using std::default_random_engine;

namespace aurora {
	namespace maths {

		class tensor {
		public:
			ptr<double> val_ptr = new double(0);
			vector<tensor> vec_ptr;
			
		public:
			tensor* group_prev_ptr = nullptr;
			tensor* group_next_ptr = nullptr;

		public:
			double& val();
			double val() const;
			vector<tensor>& vec();
			const vector<tensor>& vec() const;

		public:
			tensor& group_head();
			tensor& group_tail();
			size_t group_size();

		public:
			virtual ~tensor();
			tensor();
			tensor(double a_val);
			tensor(vector<tensor> a_vec);
			tensor(initializer_list<tensor> a_il);

		public:
			void set(const tensor& a_other);
			void pop(const tensor& a_other);

		public:
			void resize(size_t a_size);
			
		public:
			static tensor new_1d(size_t a_a);
			static tensor new_1d(size_t a_a, tensor a_val);
			static tensor new_1d(size_t a_a, uniform_real_distribution<double>& a_urd, default_random_engine& a_re);
			static tensor new_2d(size_t a_a, size_t a_b);
			static tensor new_2d(size_t a_a, size_t a_b, tensor a_val);
			static tensor new_2d(size_t a_a, size_t a_b, uniform_real_distribution<double>& a_urd, default_random_engine& a_re);

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
			void sum_1d(tensor& a_output);
			void sum_2d(tensor& a_output);
			void tanh_1d(tensor& a_output);
			void tanh_2d(tensor& a_output);
			void zero_1d();
			void zero_2d();
			void zero();

		public:
			tensor abs_1d();
			tensor abs_2d();
			tensor sum_1d();
			tensor sum_2d();
			tensor tanh_1d();
			tensor tanh_2d();
			tensor mag_1d();
			tensor max();
			tensor min();
			int arg_max();
			int arg_min();

		public:
			void add_1d(const tensor& a_other, tensor& a_output);
			void sub_1d(const tensor& a_other, tensor& a_output);
			void mul_1d(const tensor& a_other, tensor& a_output);
			void div_1d(const tensor& a_other, tensor& a_output);
			void pow_1d(const tensor& a_other, tensor& a_output);
			void dot_1d(const tensor& a_other, tensor& a_output);
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
			tensor dot_1d(const tensor& a_other);
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
			void rank_model_recur(function<void(tensor*)> a_func);

		public:
			void group_recur_fwd(function<void(tensor*)> a_func);
			void group_recur_bwd(function<void(tensor*)> a_func);
			void group_recur(function<void(tensor*)> a_func);
			bool group_contains(tensor* a_ptr);

		public:
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
			string to_string();

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