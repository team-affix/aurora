#pragma once
#include "affix-base/pch.h"
#include "model.h"
#include "sequential.h"

namespace aurora {
	namespace pseudo {
		aurora::models::Sequential nsm();
		aurora::models::Sequential nth();
		aurora::models::Sequential nth(
			double a_a,
			double a_b,
			double a_c
		);
		aurora::models::Sequential nlr(
			double a_m
		);
		aurora::models::Sequential nlrexu(
			double a_k
		);
	}
}
