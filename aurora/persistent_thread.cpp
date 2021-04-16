#include "pch.h"
#include "persistent_thread.h"

using aurora::pseudo::persistent_thread;

void persistent_thread::loop() {
	while (active)
		if (executing) {
			func();
			executing = false;
		}
}

persistent_thread::~persistent_thread() {
	this->active = false;
	thd.join();
}

persistent_thread::persistent_thread() {
	thd = std::thread([&] {loop(); });
}

void persistent_thread::execute(function<void()> a_func) {
	this->func = a_func;
	executing = true;
}

void persistent_thread::join() {
	while (executing);
}