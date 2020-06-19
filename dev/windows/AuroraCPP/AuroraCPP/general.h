#pragma once
#include "superHeader.h"

template<class T>
class sPtr : public shared_ptr<T> {
public:
	sPtr() {

	}
	sPtr(T val) {
		shared_ptr<T>::reset(new T(val));
	}
	sPtr(T* _ptr) {
		shared_ptr<T>::reset(_ptr);
	}
};

void rep(function<void()> func, int iterations);