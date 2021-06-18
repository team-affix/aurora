#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("stacked_recurrent", helper::recurrent) << std::endl;

	return 0;
}