#include "pch.h"
#include "basic_model.h"

using aurora::basic::basic_model;

void basic_model::update() {
	for (param*& pmt : m_params)
		pmt->update();
}
