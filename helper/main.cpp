#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("ntm_reader", helper::model) << std::endl;

	return 0;
}