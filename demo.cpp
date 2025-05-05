#include "mgil.hpp"

#include <print>
#include <vector>

auto main() -> int {
    using namespace mgil;
    auto view = gradient(4, 4, Pixel<int, rgb_layout_t>(0, 0, 0), Pixel<int, rgb_layout_t>(1, 2, 0),
                         Pixel<int, rgb_layout_t>(0, 0, 3));
    std::println("{}", view);
    auto crop_flipped_view =
            view | colorConvert(int{}, gray_layout_t{}) | padConstant(2, 1, Pixel<int, gray_layout_t>(-1));
    std::println("{}", crop_flipped_view);
}
