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