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

		class tensor : public vector<tensor> {
		public:
			ptr<double> val_ptr = new double(0);
			ref<double> val = val_ptr.get();

		public:
			tensor();
			tensor(double val);
			tensor(vector<tensor> vec);
			tensor(initializer_list<tensor> il);

		public:
			void set(tensor other);
			
		public:
			static tensor new_1d(size_t a);
			static tensor new_2d(size_t a, size_t b);

		public:
			tensor up_rank(size_t n);
			tensor up_rank();
			tensor down_rank(size_t n);
			tensor down_rank();

		public:
			tensor unroll();
			tensor roll(size_t width);

		public:
			size_t width();
			size_t height();

		public:
			tensor row(size_t a);
			tensor col(size_t a);
			tensor range(size_t start, size_t len);

		public:
			tensor clone_row(size_t a);
			tensor clone_col(size_t a);
			tensor clone_range(size_t start, size_t len);

		public:
			tensor sum_1d();

		public:
			void add_1d(tensor other, tensor& output);
			void sub_1d(tensor other, tensor& output);
			void mul_1d(tensor other, tensor& output);
			void div_1d(tensor other, tensor& output);
			void dot_1d(tensor other, tensor& output);
			void add_2d(tensor other, tensor& output);
			void sub_2d(tensor other, tensor& output);
			void mul_2d(tensor other, tensor& output);
			void div_2d(tensor other, tensor& output);
			void dot_2d(tensor other, tensor& output);

		public:
			tensor add_1d(tensor other);
			tensor sub_1d(tensor other);
			tensor mul_1d(tensor other);
			tensor div_1d(tensor other);
			tensor dot_1d(tensor other);
			tensor add_2d(tensor other);
			tensor sub_2d(tensor other);
			tensor mul_2d(tensor other);
			tensor div_2d(tensor other);
			tensor dot_2d(tensor other);

		public:
			void clone(tensor& output);
			void link(tensor& other);
			void unlink();

		public:
			tensor clone();

		public:
			string to_string();

		public:
			void operator=(double other);
			void operator=(tensor other);

		public:
			void clear();

		public:
			operator double& ();

		};
	}
}