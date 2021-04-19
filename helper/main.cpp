#include "code_generator.h"
#include <iostream>


int main() {

	std::cout << helper::model_impl("att", helper::attention) << std::endl;

	return 0;
}