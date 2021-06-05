#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("ntm", helper::recurrent) << std::endl;

	return 0;
}