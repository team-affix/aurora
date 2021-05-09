#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("ntm_read_head", helper::model) << std::endl;

	return 0;
}