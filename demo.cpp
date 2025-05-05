#include "mgil.hpp"

#include <print>
#include <vector>

auto main() -> int {
    using namespace mgil;
    auto image = readImage<BMPFileIO<>>("./test1.bmp");
    if (not image) {
        std::println("{}", std::to_underlying(image.error()));
    }
    auto blurred_image = gaussianBlur(image.value().toView(), 5.);
    writeImage<BMPFileIO<>>(blurred_image.toView(), "./sobel.bmp");
}
