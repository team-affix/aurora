#pragma once
#include "superHeader.h"

template<class T>
class sPtr : public shared_ptr<T> {
public:
	sPtr() {

	}
	sPtr(T* ptr) {
		shared_ptr<T>::reset(ptr);
	}
};