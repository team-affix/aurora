#include "pch.h"
#include "param_vector.h"

using aurora::params::param_vector;

void param_vector::update() {
	for (int i = 0; i < size(); i++)
		at(i)->update();
}
