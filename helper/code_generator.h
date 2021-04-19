#pragma once
#include "enums.h"
#include <string>

using helper::model_types;
using std::string;

namespace helper {
	string model_impl(string a_class_name, model_types a_model_type = model_types::model);
}