#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("mul_agg_1d", model_types::model) << std::endl;

	return 0;
}