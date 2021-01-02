#pragma once
#include "ptr.h"
#include <vector>
#include <initializer_list>
#include <string>
#include <sstream>

using aurora::data::ptr;
using std::vector;
using std::initializer_list;
using std::back_inserter;
using std::string;
using std::stringstream;

namespace aurora {
	namespace math {

		class complex : public vector<complex> {
		public:
			ptr<double> val = new double(0);

		public:
			complex();
			complex(double val);
			complex(vector<complex> vec);
			complex(initializer_list<complex> il);

		public:
			void set(complex other);
			
		public:
			static complex new_1d(size_t a);
			static complex new_2d(size_t a, size_t b);

		public:
			complex up_rank(size_t n);
			complex up_rank();
			complex down_rank(size_t n);
			complex down_rank();

		public:
			size_t width();
			size_t height();

		public:
			complex row(size_t a);
			complex col(size_t a);
			complex range(size_t start, size_t len);

		public:
			complex clone_row(size_t a);
			complex clone_col(size_t a);
			complex clone_range(size_t start, size_t len);

		public:
			complex sum_1d();

		public:
			void add_1d(complex other, complex& output);
			void sub_1d(complex other, complex& output);
			void mul_1d(complex other, complex& output);
			void div_1d(complex other, complex& output);
			void dot_1d(complex other, complex& output);
			void add_2d(complex other, complex& output);
			void sub_2d(complex other, complex& output);
			void mul_2d(complex other, complex& output);
			void div_2d(complex other, complex& output);
			void dot_2d(complex other, complex& output);

		public:
			complex add_1d(complex other);
			complex sub_1d(complex other);
			complex mul_1d(complex other);
			complex div_1d(complex other);
			complex dot_1d(complex other);
			complex add_2d(complex other);
			complex sub_2d(complex other);
			complex mul_2d(complex other);
			complex div_2d(complex other);
			complex dot_2d(complex other);

		public:
			void clone(complex& output);
			void link(complex& other);

		public:
			complex clone();

		public:
			string to_string();

		};
	}
}