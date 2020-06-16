#pragma once
#include "superHeader.h"
#include <any>

namespace maths {
	class carryType;
	class activation;
	class activation_softmax;
	class activation_tanh;
	class activation_leakyRelu;
	class tens1D;
	class tens2D;
	class carryType {
	public:
		carryType();
		carryType(double a);
		carryType(vector<shared_ptr<carryType>> a);
		carryType(initializer_list<carryType> a);
		carryType(tens1D a);
		double value_double;
		vector<shared_ptr<carryType>> value_vector;
	};
	class activation {
	public:
		virtual double eval(double x);
		virtual double deriv(double x);
	};
	class activation_softmax : public activation {
	public:
		virtual double eval(double x);
		virtual double deriv(double x);
	};
	class activation_tanh : public activation {
	public:
		virtual double eval(double x);
		virtual double deriv(double x);
	};
	class activation_leakyRelu : public activation {
	public:
		activation_leakyRelu(double m);
		virtual double eval(double x);
		virtual double deriv(double x);
		double m;
	};
	class tens1D : public vector<double> {
	public:
		tens1D();
		tens1D(int a);
		tens1D(int a, function<double()> construct);
		tens1D(int a, tens1D distribution, bool replace);
		tens1D(double minimum, double maximum, double increment);
		tens1D(initializer_list<double> vals);
		explicit tens1D(carryType a);
		tens2D repeat(int count) const;
		tens1D elemWise(function<double(double)> function) const;
		tens1D concat(tens1D a) const;
		tens1D abs();
		double getSum() const;
		tens1D getRange(int startIndex, int count) const;
		tens1D operator ^(const tens1D b);
		tens2D operator ^(const tens2D b);
		double operator *(const tens1D b);
		tens1D operator *(const tens2D b);
		tens1D operator *(const double b);
		tens1D operator +(const tens1D b);
		tens2D operator +(const tens2D b);
		tens1D operator -(const tens1D b);
		tens2D operator -(const tens2D b);
		tens1D operator /(const double b);
		tens1D operator /(const tens1D b);
	};
	class tens2D : public vector<tens1D> {
	public:
		tens2D();
		tens2D(int a);
		tens2D(int a, int b);
		tens2D(int a, function<tens1D()> construct);
		tens2D(initializer_list<tens1D> vals);
		explicit tens2D(carryType a);
		tens1D getRow(int index) const;
		tens1D getCol(int index) const;
		tens2D getRows(int startIndex, int count) const;
		tens2D getCols(int startIndex, int count) const;
		void setRow(int index, tens1D values);
		void setCol(int index, tens1D values);
		tens2D abs();
		tens1D sumDown() const;
		tens1D sumAcross() const;
		tens2D flip() const;
		tens2D elemWise(function<tens1D(tens1D)> function) const;
		tens2D elemWise(function<double(double)> function) const;
		int getHeight() const;
		int getWidth() const;
		tens2D operator ^(const tens2D b);
		tens2D operator *(const tens2D b);
		tens2D operator /(const double b);
		tens2D operator /(const tens2D b);
		tens2D operator +(const tens2D b);
		tens2D operator -(const tens2D b);
	};
#pragma region functions
	double range(double x, double min, double max);
	void zero(vector<shared_ptr<carryType>>* a);
	void add(vector<shared_ptr<carryType>>* a, vector<shared_ptr<carryType>>* b, vector<shared_ptr<carryType>>* output);
	void subtract(vector<shared_ptr<carryType>>* a, vector<shared_ptr<carryType>>* b, vector<shared_ptr<carryType>>* output);
#pragma endregion

}