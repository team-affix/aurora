#pragma once
#include "general.h"

void rep(function<void()> func, int iterations) {
	for (int i = 0; i < iterations; i++) {
		func();
	}
}