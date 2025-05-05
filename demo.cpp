#include "mgil.hpp"

#include <print>

auto main() -> int {
    using namespace mgil;
    auto image = readImage<BMPFileIO<>>("./test1.bmp");
    if (not image) {
        std::println("Image load failed with error code {}", std::to_underlying(image.error()));
        return 1;
    }
    auto processed_image = image.value().toView() | rotate90() | transform([](Pixel<UInt8_0255, rgb_layout_t> pixel) {
        pixel.get<red_color_t>() = 0;
        pixel.get<green_color_t>() = 0;
        return pixel;
    });
    auto concat_image = concatHorizontal(processed_image, processed_image);
    writeImage<BMPFileIO<>>(concat_image, "./processed.bmp");
}
