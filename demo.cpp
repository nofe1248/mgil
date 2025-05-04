#include "mgil.hpp"

#include <print>
#include <vector>

auto main() -> int {
    using namespace mgil;
    auto view = gradient(12, 12, Pixel<int, rgb_layout_t>(0, 0, 0), Pixel<int, rgb_layout_t>(1, 2, 0),
                         Pixel<int, rgb_layout_t>(0, 0, 3));
    std::println("{}", view);
    auto subsampled = nearest(view, 16, 16);
    std::println("{}", subsampled);
}
