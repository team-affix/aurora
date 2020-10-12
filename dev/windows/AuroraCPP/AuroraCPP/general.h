#pragma once
#include "superHeader.h"

template<class T>
class ptr : public shared_ptr<T> {
public:
	ptr() {

	}
	ptr(T* _ptr) {
		shared_ptr<T>::reset(_ptr);
	}
};

void rep(function<void()> func, int iterations);

template<class TIn, class TOut>
vector<TOut> getVals(vector<TIn>* a, function<TOut(TIn)> getFrom) {
	vector<TOut> result = vector<TOut>();
	for (int i = 0; i < a->size(); i++) {
		result.push_back(getFrom(a->at(i)));
	}
	return result;
}

template<class TIn, class TOut>
void setVals(vector<TIn>* a, vector<TOut>* output, function<void(TIn&, TOut&)> setFrom) {

	// make sure the input and output vectors are of the same size
	assert(a->size() == output->size());

	for (int i = 0; i < a->size(); i++) {
		setFrom(a->at(i), output->at(i));
	}
}

template<class T>
void elemWise(vector<T>* a, function<void(T)> func) {
	for (int i = 0; i < a->size(); i++) {
		func(a->at(i));
	}
}

template<class T>
vector<T> concat(vector<T>* a, vector<T>* b) {
	vector<T> result = vector<T>();
	for (int i = 0; i < a->size(); i++) {
		result.push_back(a->at(i));
	}
	for (int i = 0; i < b->size(); i++) {
		result.push_back(b->at(i));
	}
	return result;
}

template<class T>
void push_back(vector<T>& a, vector<T>& b) {
	for (int i = 0; i < b.size(); i++) {
		a.push_back(b.at(i));
	}
}

vector<string> strsplit(const string& input, const string& delim);