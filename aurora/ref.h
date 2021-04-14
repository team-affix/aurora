#pragma once

namespace aurora {
	namespace data {
		template<class T>
		class ref {
		public:
			T* pointer = nullptr;

		public:
			ref() {

			}
			ref(T* _pointer) {
				pointer = _pointer;
			}

		public:
			void operator=(T _other) {
				*pointer = _other;
			}
			void operator=(T* _pointer) {
				pointer = _pointer;
			}
			void operator=(ref<T>& _other) {
				*pointer = *_other.pointer;
			}
			operator T& () {
				return *pointer;
			}

		};
	}
}
