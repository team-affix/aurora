#include "code_generator.h"
#include <fstream>

std::string file_read_all_text(string file_name) {
	std::ifstream is(file_name);
	std::string contents;
	// Here is one way to read the whole file
	for (char ch; is.get(ch); contents.push_back(ch));
	is.close();
	return contents;
}

string find_and_replace(std::string file_contents,
	const std::string& word_1, const std::string& word_2) {
	// This searches the file for the first occurence of the morn string.
	auto pos = file_contents.find(word_1);
	while (pos != std::string::npos) {
		file_contents.replace(pos, word_1.length(), word_2);
		// Continue searching from here.
		pos = file_contents.find(word_1, pos);
	}

	return file_contents;
}

string read_and_replace(string file_name, string word_1, string word_2) {
	return find_and_replace(file_read_all_text(file_name), word_1, word_2);
}

string helper::model_impl(string a_class_name, model_types a_model_type) {
	string model_impl = read_and_replace("model_impl.txt", "MODEL", a_class_name);
	string recurrent_impl;
	string attention_impl;
	switch (a_model_type) {
	case helper::recurrent:
		recurrent_impl = read_and_replace("recurrent_impl.txt", "MODEL", a_class_name);
		break;
	case helper::attention:
		attention_impl = read_and_replace("attention_impl.txt", "MODEL", a_class_name);
		break;
	}
	return model_impl + recurrent_impl + attention_impl;
}