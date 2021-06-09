#pragma once
#include "pch.h"

#define INIT_PMT(right_side, param_vector) \
[&](ptr<param>& pmt) { \
auto l_pmt = new right_side; \
pmt = l_pmt; \
param_vector.push_back(l_pmt); \
}