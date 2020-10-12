#include "processing.h"

vector<ptr<cType>> stride(ptr<cType> a, int kWidth, int kHeight, int horStride, int verStride) {

	int width;
	int height;
	bool is2D;
	info2D(a, width, height, is2D);

	// ensure the inputted rectangle is rectangular, and therefore fit for convolutional kernal striding
	assert(is2D);

	vector<ptr<cType>> result = vector<ptr<cType>>();

	for (int x = 0; x + kWidth <= width; x += horStride) {
		for (int y = 0; y + kHeight <= height; y += verStride) {

			result.push_back(get2D(a, x, y, kWidth, kHeight));

		}
	}

	return result;

}