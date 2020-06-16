#pragma once
#include "maths.h"
using namespace std;
using namespace maths;

#pragma region forwardDeclaration
void processInequalSizeException(const tens1D a, const tens1D b);
void processInequalSizeException(const tens2D a, const tens2D b);
#pragma endregion
#pragma region carryType
carryType::carryType() {

}
carryType::carryType(double a) {
	value_double = a;
}
carryType::carryType(vector<shared_ptr<carryType>> a) {
	value_vector = a;
}
carryType::carryType(initializer_list<carryType> a) {
	vector<carryType> v = vector<carryType>();
	copy(a.begin(), a.end(), back_inserter(v));

	value_vector = vector<shared_ptr<carryType>>();
	for (int i = 0; i < v.size(); i++) {
		value_vector.push_back(shared_ptr<carryType>(new carryType(v.at(i))));
	}
}
carryType::carryType(tens1D a) {
	value_vector = vector<shared_ptr<carryType>>();
	for (int i = 0; i < a.size(); i++) {
		value_vector.push_back(shared_ptr<carryType>(new carryType(a[i])));
	}
}
#pragma endregion
#pragma region activation
	double activation::eval(double x) {
		return x;
	}
	double activation::deriv(double x) {
		return 1;
	}
#pragma endregion
#pragma region tanh
	double activation_tanh::eval(double x) {
		return std::tanh(x);
	}
	double activation_tanh::deriv(double x) {
		return 1 / pow(cosh(x), 2);
	}
#pragma endregion
#pragma region softmax
	double activation_softmax::eval(double x) {
		return 1 / (1 + exp(-x));
	}
	double activation_softmax::deriv(double x) {
		return x * (1 - x);
	}
#pragma endregion
#pragma region leakyRelu
	activation_leakyRelu::activation_leakyRelu(double m) {
		this->m = m;
	}
	double activation_leakyRelu::eval(double x) {
		if (x > 0) {
			return x;
		}
		else {
			return m * x;
		}
	}
	double activation_leakyRelu::deriv(double x) {
		if (x > 0) {
			return 1;
		}
		else {
			return m;
		}
	}
#pragma endregion
#pragma region tens1D
	tens1D::tens1D() {

	}
	tens1D::tens1D(int a) {
		for (int i = 0; i < a; i++) {
			push_back(0);
		}
	}
	tens1D::tens1D(int a, function<double()> construct) {
		for (int i = 0; i < a; i++) {
			push_back(construct());
		}
	}
	tens1D::tens1D(int a, tens1D distribution, bool replace) {
		for (int i = 0; i < a; i++) {
			int index = rand() % (distribution.size());
			push_back(distribution[index]);
			if (!replace) {
				distribution.erase(distribution.begin() + index);
			}
		}
	}
	tens1D::tens1D(double minimum, double maximum, double increment) {
		for (double d = minimum; d < maximum; d += increment) {
			push_back(d);
		}
	}
	tens1D::tens1D(initializer_list<double> vals) {
		copy(vals.begin(), vals.end(), back_inserter(*this));
	}
	tens1D::tens1D(carryType a) {
		vector<shared_ptr<carryType>>* v = &a.value_vector;
		for (int i = 0; i < v->size(); i++) {
			push_back(v->at(i)->value_double);
		}
	}
	tens2D tens1D::repeat(int count) const {
		tens2D result = tens2D();
		for (int i = 0; i < count; i++) {
			result.push_back(*this);
		}
		return result;
	}
	tens1D tens1D::elemWise(function<double(double)> function) const {
		tens1D result = tens1D();
		for (int i = 0; i < size(); i++) {
			result.push_back(function(this->at(i)));
		}
		return result;
	}
	tens1D tens1D::concat(tens1D a) const {
		tens1D result = *this;
		result.insert(result.end(), a.begin(), a.end());
		return result;
	}
	tens1D tens1D::abs() {
		tens1D result = tens1D();
		for (int i = 0; i < size(); i++) {
			result.push_back(std::abs(at(i)));
		}
		return result;
	}
	double tens1D::getSum() const {
		double sum = 0;
		for (int i = 0; i < size(); i++) {
			sum += this->at(i);
		}
		return sum;
	}
	tens1D tens1D::getRange(int startIndex, int count) const {
		tens1D result = tens1D();
		for (int i = startIndex; i < startIndex + count; i++) {
			result.push_back(this->at(i));
		}
		return result;
	}
	tens1D tens1D::operator^(const tens1D b) {
		processInequalSizeException(*this, b);
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) * b.at(i));
		}
		return result;
	} // Fundamental : includes exception handling
	tens1D tens1D::operator+(const tens1D b) {
		processInequalSizeException(*this, b);
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) + b.at(i));
		}
		return result;
	} // Fundamental : includes exception handling
	tens1D tens1D::operator-(const tens1D b) {
		processInequalSizeException(*this, b);
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) - b.at(i));
		}
		return result;
	}
	tens2D tens1D::operator^(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < b.getHeight(); i++) {
			result.push_back(*this ^ b.at(i));
		}
		return result;
	}
	double tens1D::operator*(const tens1D b) {
		tens1D product = *this ^ b;
		double result = product.getSum();
		return result;
	}
	tens1D tens1D::operator*(const tens2D b) {
		tens2D product = *this ^ b;
		tens1D result = product.sumDown();
		return result;
	}
	tens1D tens1D::operator*(const double b) {
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) * b);
		}
		return result;
	}
	tens2D tens1D::operator+(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < b.getWidth(); i++) {
			result.push_back(*this + b.at(i));
		}
		return result;
	}
	tens2D tens1D::operator-(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < b.size(); i++) {
			result.push_back(*this - b.at(i));
		}
		return result;
	}
	tens1D tens1D::operator/(const double b) {
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) / b);
		}
		return result;
	}
	tens1D tens1D::operator/(const tens1D b) {
		tens1D result = tens1D();
		for (int i = 0; i < this->size(); i++) {
			result.push_back(this->at(i) / b.at(i));
		}
		return result;
	}
	void processInequalSizeException(const tens1D a, const tens1D b) {
		int aSize = a.size();
		int bSize = b.size();
		if (aSize != bSize) {
			stringstream exceptionDescription;
			exceptionDescription << "operands are of invalid sizes; " << aSize << " != " << bSize;
			throw exceptionDescription.str();
		}
	}
#pragma endregion
#pragma region tens2D
	tens2D::tens2D() {

	}
	tens2D::tens2D(int a) {
		for (int i = 0; i < a; i++) {
			this->push_back(tens1D());
		}
	}
	tens2D::tens2D(int a, int b) {
		for (int i = 0; i < a; i++) {
			this->push_back(tens1D(b));
		}
	}
	tens2D::tens2D(int a, function<tens1D()> construct) {
		for (int i = 0; i < a; i++) {
			this->push_back(construct());
		}
	}
	tens2D::tens2D(initializer_list<tens1D> vals) {
		copy(vals.begin(), vals.end(), back_inserter(*this));
	}
	tens2D::tens2D(carryType a) {
		vector<shared_ptr<carryType>>* v = &a.value_vector;
		for (int i = 0; i < v->size(); i++) {
			push_back((tens1D)*v->at(i));
		}
	}
	tens1D tens2D::getRow(int index) const {
		return this->at(index);
	}
	tens1D tens2D::getCol(int index) const {
		tens1D result = tens1D();
		for (int i = 0; i < this->getHeight(); i++) {
			result.push_back(this->at(i).at(index));
		}
		return result;
	}
	tens2D tens2D::getRows(int startIndex, int count) const {
		tens2D result = tens2D();
		for (int i = startIndex; i < startIndex + count; i++) {
			result.push_back(this->getRow(i));
		}
		return result;
	}
	tens2D tens2D::getCols(int startIndex, int count) const {
		tens2D result = tens2D();
		for (int i = startIndex; i < startIndex + count; i++) {
			result.push_back(this->getCol(i));
		}
		return result;
	}
	void tens2D::setRow(int index, tens1D values) {
		this->at(index) = values;
	}
	void tens2D::setCol(int index, tens1D values) {
		for (int i = 0; i < this->getHeight(); i++) {
			this->at(i).at(index) = values[i];
		}
	}
	tens2D tens2D::abs() {
		tens2D result = tens2D();
		for (int i = 0; i < size(); i++) {
			result.push_back(at(i).abs());
		}
		return result;
	}
	tens1D tens2D::sumDown() const {
		tens1D result = this->getRow(0);
		for (int i = 1; i < getHeight(); i++) {
			result = result + this->getRow(i);
		}
		return result;
	}
	tens1D tens2D::sumAcross() const {
		tens1D result = this->getCol(0);
		for (int i = 1; i < getWidth(); i++) {
			result = result + this->getCol(i);
		}
		return result;
	}
	tens2D tens2D::flip() const {
		tens2D result = tens2D(this->getWidth(), this->getHeight());
		for (int i = 0; i < getHeight(); i++) {
			result.setCol(i, this->getRow(i));
		}
		return result;
	}
	tens2D tens2D::elemWise(function<tens1D(tens1D)> function) const {
		tens2D result = tens2D();
		for (int i = 0; i < getHeight(); i++) {
			result.push_back(function(this->getRow(i)));
		}
		return result;
	}
	tens2D tens2D::elemWise(function<double(double)> function) const {
		tens2D result = tens2D();
		for (int i = 0; i < getHeight(); i++) {
			result.push_back(this->at(i).elemWise(function));
		}
		return result;
	}
	int tens2D::getHeight() const {
		return size();
	}
	int tens2D::getWidth() const {
		if (size() > 0) {
			return this->at(0).size();
		}
		else {
			return 0;
		}
	}
	tens2D tens2D::operator^(const tens2D b) {
		processInequalSizeException(*this, b);
		tens2D result = tens2D();
		for (int i = 0; i < getHeight(); i++) {
			result.push_back(this->at(i) ^ b.at(i));
		}
		return result;
	}
	tens2D tens2D::operator*(const tens2D b) {
		if (this->getWidth() != b.getHeight()) 
		{
			stringstream exceptionDescription;
			exceptionDescription << "operands are of invalid sizes; " << this->getWidth() << " != " << b.getHeight();
			throw exceptionDescription.str();
		}
		tens2D result = tens2D(this->getHeight(), b.getWidth());
		for (int i = 0; i < this->getHeight(); i++) {
			for (int j = 0; j < b.getWidth(); j++) {
				tens1D row = this->getRow(i);
				tens1D col = b.getCol(j);
				result[i][j] = row * col;
			}
		}
		return result;
	}
	tens2D tens2D::operator/(const double b) {
		tens2D result = tens2D();
		for (int i = 0; i < getHeight(); i++) {
			result.push_back(this->at(i) / b);
		}
		return result;
	}
	tens2D tens2D::operator/(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < this->getHeight(); i++) {
			result.push_back(this->getRow(i) / b.getRow(i));
		}
		return result;
	}
	tens2D tens2D::operator+(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < this->getHeight(); i++) {
			result.push_back(this->getRow(i) + b.at(i));
		}
		return result;
	}
	tens2D tens2D::operator-(const tens2D b) {
		tens2D result = tens2D();
		for (int i = 0; i < this->getHeight(); i++) {
			result.push_back(this->getRow(i) - b.at(i));
		}
		return result;
	}
	void processInequalSizeException(const tens2D a, const tens2D b) {
		int aSize = a.size();
		int bSize = b.size();
		if (aSize != bSize) {
			stringstream exceptionDescription;
			exceptionDescription << "operands are of invalid sizes; " << aSize << " != " << bSize;
			throw exceptionDescription.str();
		}
	}
#pragma endregion
#pragma region functions
double maths::range(double x, double min, double max) {
	if (x < min) {
		return min;
	}
	else if (x > max) {
		return max;
	}
	else {
		return x;
	}
}
void maths::zero(vector<shared_ptr<carryType>>* a) {
	for (int i = 0; i < a->size(); i++) {
		shared_ptr<carryType> aPtr = a->at(i);
		aPtr->value_double = 0;
	}
}
void maths::add(vector<shared_ptr<carryType>>* a, vector<shared_ptr<carryType>>* b, vector<shared_ptr<carryType>>* output) {
	for (int i = 0; i < a->size(); i++) {
		cout << "";
		shared_ptr<carryType> ac = a->at(i);
		shared_ptr<carryType> bc = b->at(i);
		output->at(i)->value_double = ac->value_double + bc->value_double;
	}
}
void maths::subtract(vector<shared_ptr<carryType>>* a, vector<shared_ptr<carryType>>* b, vector<shared_ptr<carryType>>* output) {
	for (int i = 0; i < a->size(); i++) {
		shared_ptr<carryType> ac = a->at(i);
		shared_ptr<carryType> bc = b->at(i);
		output->at(i)->value_double = ac->value_double - bc->value_double;
	}
}
/*
vector<carryType> maths::subtract(vector<carryType>* a, vector<carryType>* b) {

}
vector<carryType> maths::abs(vector<carryType>* a, vector<carryType>* b) {

}
vector<carryType> maths::sum(vector<carryType>* a, vector<carryType>* b) {

}*/
#pragma endregion