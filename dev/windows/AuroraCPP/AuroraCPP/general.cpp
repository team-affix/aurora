#pragma once
#include "general.h"

void rep(function<void()> func, int iterations) {
	for (int i = 0; i < iterations; i++) {
		func();
	}
}

vector<string> strsplit(const string& input, const char& delim) {

	vector<int> splitIndices = vector<int>();
	for (int i = 0; i < input.size(); i++) {
		if (input[i] == delim) {
			splitIndices.push_back(i);
		}
	}

	vector<string> result = vector<string>();
	int prevIndex = -1;
	if (input.length() > 0) {
		for (int i = 0; i < splitIndices.size(); i++) {
			result.push_back(input.substr(prevIndex + 1, splitIndices[i] - (prevIndex + 1)));
			prevIndex = splitIndices[i];
		}
	}
	result.push_back(input.substr(prevIndex + 1, input.length() - (prevIndex + 1)));
	return result;

}