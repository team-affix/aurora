#pragma once
#include "ptr.h"
#include "ref.h"
#include <vector>
#include <initializer_list>
#include <string>
#include <sstream>

using aurora::data::ptr;
using aurora::data::ref;
using std::vector;
using std::initializer_list;
using std::back_inserter;
using std::string;
using std::stringstream;

namespace aurora {
	namespace math {

		class tensor {
		public:
			ptr<double> val_ptr = new double(0);
			ptr<vector<tensor>> vec_ptr = new vector<tensor>();

		public:
			double& val();
			vector<tensor>& vec();

		public:
			tensor();
			tensor(double a_val);
			tensor(vector<tensor> a_vec);
			tensor(initializer_list<tensor> a_il);

		public:
			void set(tensor a_other);
			
		public:
			static tensor new_1d(size_t a_a);
			static tensor new_2d(size_t a_a, size_t a_b);

		public:
			tensor up_rank(size_t a_n);
			tensor up_rank();
			tensor down_rank(size_t a_n);
			tensor down_rank();

		public:
			tensor unroll();
			tensor roll(size_t a_width);

		public:
			size_t width();
			size_t height();

		public:
			tensor row(size_t a_a);
			tensor col(size_t a_a);
			tensor range(size_t a_start, size_t a_len);

		public:
			tensor clone_row(size_t a_a);
			tensor clone_col(size_t a_a);
			tensor clone_range(size_t a_start, size_t a_len);

		public:
			tensor sum_1d();

		public:
			void add_1d(tensor a_other, tensor& a_output);
			void sub_1d(tensor a_other, tensor& a_output);
			void mul_1d(tensor a_other, tensor& a_output);
			void div_1d(tensor a_other, tensor& a_output);
			void dot_1d(tensor a_other, tensor& a_output);
			void add_2d(tensor a_other, tensor& a_output);
			void sub_2d(tensor a_other, tensor& a_output);
			void mul_2d(tensor a_other, tensor& a_output);
			void div_2d(tensor a_other, tensor& a_output);
			void dot_2d(tensor a_other, tensor& a_output);

		public:
			tensor add_1d(tensor a_other);
			tensor sub_1d(tensor a_other);
			tensor mul_1d(tensor a_other);
			tensor div_1d(tensor a_other);
			tensor dot_1d(tensor a_other);
			tensor add_2d(tensor a_other);
			tensor sub_2d(tensor a_other);
			tensor mul_2d(tensor a_other);
			tensor div_2d(tensor a_other);
			tensor dot_2d(tensor a_other);

		public:
			void clone(tensor& a_output);
			void link(tensor& a_other);
			void unlink();

		public:
			tensor clone();

		public:
			string to_string();

		public:
			void clear();
			size_t size();
			tensor& at(size_t a_a);

		public:
			operator double& ();
			tensor& operator[](size_t a_a);

		};
	}
}