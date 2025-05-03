#pragma once

#ifndef MGIL_H
#define MGIL_H

/*
 * MGIL: Modern Generic Image Library
 *
 * Zhige Chen, Southern University of Science and Technology
 *
 * This library is a project assignment for the CS219 (Advanced Programming) course.
 */

#include <algorithm>
#include <array>
#include <compare>
#include <concepts>
#include <cstdint>
#include <format>
#include <iterator>
#include <numeric>
#include <ostream>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <version>

#include "mgil.hpp"

// A Stripped Version of qlibs/mp, a lightweight, self-contained C++ metaprogramming library
// See https://github.com/qlibs/mp/

namespace mgil::details::mp {
    using size_t = decltype(sizeof(int));

    namespace utility {
        template<class T, T...>
        struct integer_sequence {};
        template<size_t... Ns>
        using index_sequence = integer_sequence<size_t, Ns...>;
        template<size_t N>
        using make_index_sequence =
#if defined(__clang__) || defined(_MSC_VER)
                __make_integer_seq<integer_sequence, size_t, N>;
#else
                index_sequence<__integer_pack(N)...>;
#endif
        template<class T>
        auto declval() -> T &&;
    } // namespace utility

    namespace concepts {
        template<class TRange>
        concept range = requires(TRange range) {
            range.begin();
            range.end();
        };
        template<class Fn, class... TArgs>
        concept invocable = requires(Fn fn, TArgs... args) { fn(args...); };
    } // namespace concepts

    template<class T, size_t N>
    struct array {
        using value_type = T;
        [[nodiscard]] constexpr auto begin() const {
            return &data[0];
        }
        [[nodiscard]] constexpr auto begin() {
            return &data[0];
        }
        [[nodiscard]] constexpr auto end() const {
            return &data[0] + N;
        }
        [[nodiscard]] constexpr auto end() {
            return &data[0] + N;
        }
        [[nodiscard]] constexpr auto operator[](size_t I) const {
            return data[I];
        }
        [[nodiscard]] constexpr auto &operator[](size_t I) {
            return data[I];
        }
        [[nodiscard]] constexpr auto size() const {
            return N;
        }
        [[nodiscard]] constexpr auto operator==(const array &other) const -> bool {
            for (auto i = 0u; i < N; ++i) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }
            return true;
        }
        [[nodiscard]] constexpr auto operator!=(const array &other) const -> bool {
            return not operator==(other);
        }
        T data[N];
    };
    template<class T, class... Ts>
    array(T, Ts...) -> array<T, 1u + sizeof...(Ts)>;

    template<class T, size_t N = 1024u>
    struct vector {
        using value_type = T;
        constexpr vector() = default;
        constexpr vector(size_t size) : size_{size} {
        }
        constexpr explicit vector(const auto &...ts)
            requires(requires { T(ts); } and ...)
            : data{ts...}, size_{sizeof...(ts)} {
        }
        constexpr vector(concepts::range auto range) {
            for (const auto &t: range) {
                data[size_++] = t;
            }
        }
        constexpr void push_back(const T &t) {
            data[size_++] = t;
        }
        constexpr void emplace_back(T &&t) {
            data[size_++] = static_cast<T &&>(t);
        }
        [[nodiscard]] constexpr auto begin() const {
            return &data[0];
        }
        [[nodiscard]] constexpr auto begin() {
            return &data[0];
        }
        [[nodiscard]] constexpr auto end() const {
            return &data[0] + size_;
        }
        [[nodiscard]] constexpr auto end() {
            return &data[0] + size_;
        }
        [[nodiscard]] constexpr auto operator[](size_t i) const {
            return data[i];
        }
        [[nodiscard]] constexpr auto &operator[](size_t i) {
            return data[i];
        }
        [[nodiscard]] constexpr auto size() const {
            return size_;
        }
        [[nodiscard]] constexpr auto resize(size_t size) {
            size_ = size;
        }
        [[nodiscard]] constexpr auto capacity() const {
            return N;
        }
        [[nodiscard]] constexpr auto operator==(const vector<T> &other) const -> bool {
            if (size_ != other.size_) {
                return false;
            }
            for (auto i = 0u; i < size_; ++i) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }
            return true;
        }
        template<size_t size>
        [[nodiscard]] constexpr auto operator!=(const vector<T> &other) const -> bool {
            return not operator==(other);
        }
        constexpr void clear() {
            size_ = {};
        }

        union {
            T data[N]{};
        }; // __cpp_trivial_union
        size_t size_{};
    };
    template<class T, class... Ts>
    vector(T, Ts...) -> vector<T>;

    enum class info : size_t {
    };

    namespace detail {
        template<info>
        struct info {
            constexpr auto friend get(info);
        };
        template<class T>
        struct meta {
            using value_type = T;
            template<size_t left = 0u, size_t right = 1024u - 1u>
            static constexpr auto gen() -> size_t {
                if constexpr (left >= right) {
                    return left + requires { get(info<mp::info{left}>{}); };
                } else if constexpr (constexpr auto mid = left + (right - left) / 2u;
                                     requires { get(info<mp::info{mid}>{}); }) {
                    return gen<mid + 1u, right>();
                } else {
                    return gen<left, mid - 1u>();
                }
            }
            static constexpr auto id = mp::info{gen()};
            constexpr auto friend get(info<id>) {
                return meta{};
            }
        };
        void failed();
    } // namespace detail

    template<class T>
    inline constexpr info meta = detail::meta<T>::id;

    template<info meta>
    using type_of = typename decltype(get(detail::info<meta>{}))::value_type;

    template<template<class...> class T>
    [[nodiscard]] constexpr auto apply(concepts::invocable auto expr) {
        constexpr concepts::range auto range = expr();
        return [range]<size_t... Ns>(utility::index_sequence<Ns...>) {
            return T<type_of<range[Ns]>...>{};
        }(utility::make_index_sequence<range.size()>{});
    }

    template<template<class...> class T, concepts::range auto range>
    inline constexpr auto apply_v = []<size_t... Ns>(utility::index_sequence<Ns...>) {
        return T<type_of<range[Ns]>...>{};
    }(utility::make_index_sequence<range.size()>{});

    template<template<class...> class T, concepts::range auto range>
    using apply_t = decltype([]<size_t... Ns>(utility::index_sequence<Ns...>) {
        return utility::declval<T<type_of<range[Ns]>...>>();
    }(utility::make_index_sequence<range.size()>{}));

    template<class Fn, class T = decltype([] {})>
    [[nodiscard]] inline constexpr auto invoke(Fn &&fn, info meta) {
        constexpr auto dispatch = [&]<size_t... Ns>(utility::index_sequence<Ns...>) {
            return array {
                []<info N> {
                    return +[](Fn fn) {
                        if constexpr (requires { fn.template operator()<N>(); }) {
                            return fn.template operator()<N>();
                        }
                    };
                }.template operator()<info{Ns}>()...
            };
        }(utility::make_index_sequence<size_t(mp::meta<T>)>{});
        return dispatch[size_t(meta)](fn);
    }

    template<template<class...> class T, class... Ts, auto = [] {}>
    [[nodiscard]] inline constexpr auto invoke(info meta) {
        return invoke(
                []<info meta> {
                    using type = type_of<meta>;
                    if constexpr (requires { T<Ts..., type>::value; }) {
                        return T<Ts..., type>::value;
                    } else {
                        return mp::meta<typename T<Ts..., type>::type>;
                    }
                },
                meta);
    }

    template<size_t N>
    inline constexpr auto unroll = [](auto &&fn) {
        const auto invoke = [&]<size_t I> {
            if constexpr (requires { fn.template operator()<size_t{}>(); }) {
                fn.template operator()<I>();
            } else {
                fn();
            }
        };
        [&]<size_t... Ns>(utility::index_sequence<Ns...>) {
            (invoke.template operator()<Ns>(), ...);
        }(utility::make_index_sequence<N>{});
    };

    template<concepts::range auto range>
    inline constexpr auto for_each = [](auto &&fn) {
        [fn]<size_t... Ns>(utility::index_sequence<Ns...>) {
            (fn.template operator()<range[Ns]>(), ...);
        }(utility::make_index_sequence<range.size()>{});
    };
} // namespace mgil::details::mp

#if __cpp_deleted_function >= 202403L
#define DELETE_DEF(str) delete (str)
#elif
#define DELETE_DEF(str) delete
#endif

// Metaprogramming Utils
// WARNING: TONS of template tricks

namespace mgil::details {
    // A simple utility to extract nth element from a pack
#if __cpp_pack_indexing >= 202311L
    template<std::size_t Index, typename... Types>
    struct nth_type {
        using type = Types...[Index];
    };
    template<std::size_t Index, typename... Types>
    using nth_type_t = typename nth_type<Index, Types...>::type;

    template<std::size_t Index, auto... Values>
    struct nth_nttp {
        static constexpr auto value = Values...[Index];
    };
    template<std::size_t Index, auto... Values>
    constexpr auto nth_nttp_v = nth_nttp<Index, Values...>::value;
#elif
    template<std::size_t Index, typename Head, typename... Types>
    struct nth_type {
        using type = typename nth_type_impl<Index - 1, Pack...>::type;
    };
    template<typename Head, typename... Types>
    struct nth_type<0, Head, Types...> {
        using type = Head;
    };
    template<std::size_t Index, typename Head, typename... Types>
    using nth_type_t = nth_type<Index, Head, Pack...>::type;

    template<std::size_t Index, typename Head, auto... Values>
    struct nth_nttp {
        static constexpr auto value = typename nth_type_impl<Index - 1, Values...>::value;
    };
    template<typename Head, auto... Values>
    struct nth_nttp<0, Head, Values...> {
        static constexpr auto value = Head;
    };
    template<std::size_t Index, typename Head, auto... Values>
    constexpr auto nth_nttp_v = nth_type<Index, Head, Values...>::value;
#endif

    // A not-so-simple type list
    template<typename... Types>
    struct type_list {
        static constexpr size_t size = sizeof...(Types);
        template<std::size_t Index>
        using NthType = typename nth_type<Index, Types...>::type;
        static constexpr auto getVector() noexcept {
            return std::vector<mp::info>{mp::meta<Types>...};
        }
        static constexpr auto getMPVector() noexcept {
            return mp::vector<mp::info>{mp::meta<Types>...};
        }
        template<template<typename> class Pred>
        static constexpr auto allConformTo() noexcept -> bool {
            return (Pred<Types>::value && ...);
        }
        template<auto Pred>
        static constexpr auto allConformTo() noexcept -> bool {
            return (Pred.template operator()<Types>() && ...);
        }
    };

    // Again a not-so-simple index list
    template<std::size_t... indexes>
    struct index_list {
        static constexpr std::size_t size = sizeof...(indexes);
        template<std::size_t N>
        static constexpr auto get() -> std::size_t {
            return nth_nttp_v<N, indexes...>;
        }
        static constexpr auto getVector() noexcept {
            return std::vector<std::size_t>{indexes...};
        }
        static constexpr auto getMPVector() noexcept {
            return mp::vector<std::size_t>{indexes...};
        }
    };

    template<typename T, template<auto...> class Templ>
    struct is_specialization_of_nttp : std::false_type {};

    template<template<auto...> class Templ, auto... Args>
    struct is_specialization_of_nttp<Templ<Args...>, Templ> : std::true_type {};

    template<typename T, template<auto...> class Templ>
    constexpr bool is_specialization_of_nttp_v = is_specialization_of_nttp<T, Templ>::value;

    template<typename T, template<typename...> class Templ>
    struct is_specialization_of : std::false_type {};

    template<template<typename...> class Templ, typename... Args>
    struct is_specialization_of<Templ<Args...>, Templ> : std::true_type {};

    template<typename T, template<typename...> class Templ>
    constexpr bool is_specialization_of_v = is_specialization_of<T, Templ>::value;

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(is_specialization_of_nttp_v<index_list<1, 2, 3>, index_list>);
    static_assert(is_specialization_of_v<std::vector<int>, std::vector>);
#endif

    // Compile-time algorithm: Sort types
    template<typename... Types>
    struct sort_types {
        using type = mp::apply_t<type_list, [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            std::ranges::sort(vec);
            return vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename... Types>
    struct sort_types<type_list<Types...>> : sort_types<Types...> {};

    // Compile-time algorithm: Deduplicate types
    template<typename... Types>
    struct deduplicate_types {
        using type = mp::apply_t<type_list, [] {
            auto vec = mp::vector{mp::meta<Types>...} |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_cvref>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_all_extents>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::remove_pointer>(m); }) |
                       std::views::transform([](auto m) { return mp::invoke<std::decay>(m); }) |
                       std::ranges::to<std::vector<mp::info>>();
            std::ranges::sort(vec);
            auto ret = std::ranges::unique(vec);
            vec.erase(ret.begin(), ret.end());
            return vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename... Types>
    struct deduplicate_types<type_list<Types...>> : deduplicate_types<Types...> {};

    // Compile-time algorithm: Find a type in a pack of types
    template<typename TypeToFind, typename... Types>
    struct find_type {
        static constexpr std::size_t value = [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            return std::distance(vec.begin(), std::ranges::find(vec, mp::meta<TypeToFind>));
        }();
    };

    template<typename TypeToFind, typename... Types>
    struct find_type<TypeToFind, type_list<Types...>> : find_type<TypeToFind, Types...> {};

    template<typename TypeToFind, typename... Types>
    constexpr std::size_t find_type_v = find_type<TypeToFind, Types...>::value;

    // Compile-time algorithm: Find if a type in a pack of types
    template<typename TypeToFind, typename... Types>
    struct contain_type {
        static constexpr bool value = [] {
            auto vec = std::vector<mp::info>{mp::meta<Types>...};
            return std::ranges::contains(vec, mp::meta<TypeToFind>);
        }();
    };

    template<typename TypeToFind, typename... Types>
    struct contain_type<TypeToFind, type_list<Types...>> : contain_type<TypeToFind, Types...> {};

    template<typename TypeToFind, typename... Types>
    constexpr bool contain_type_v = contain_type<TypeToFind, Types...>::value;

    template<typename TypeList, typename IndexList>
        requires is_specialization_of_v<TypeList, type_list> and is_specialization_of_nttp_v<IndexList, index_list> and
                 (TypeList::size == IndexList::size)
    struct rearrange_type_list {
        using type = mp::apply_t<type_list, [] {
            auto const type_vec = TypeList::getVector();
            auto result_vec = std::vector<mp::info>{};
            for (auto const index_vec = IndexList::getVector(); auto const index: index_vec) {
                result_vec.push_back(type_vec.at(index));
            }
            return result_vec | std::ranges::to<mp::vector<mp::info>>();
        }()>;
    };

    template<typename TypeList, typename IndexList>
    using rearrange_type_list_t = typename rearrange_type_list<TypeList, IndexList>::type;

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(std::same_as<type_list<int, float, char>,
                               rearrange_type_list_t<type_list<float, char, int>, index_list<2, 0, 1>>>);
#endif

    // The base class of every image adaptor closure
    struct ImageAdaptorClosure {};

    inline namespace concepts {
        template<typename Lhs, typename Rhs, typename Image>
        concept PipeInvocable = requires { std::declval<Rhs>()(std::declval<Lhs>()(std::declval<Image>())); };

        template<typename Type>
            requires(not std::same_as<Type, ImageAdaptorClosure>)
        auto isImageAdaptorClosureFunction(Type const &, ImageAdaptorClosure const &);

        template<typename Type>
        concept IsImageAdaptorClosure = requires(Type t) { isImageAdaptorClosureFunction(t, t); };

        template<typename Adaptor, typename... Args>
        concept AdaptorInvocable = requires { std::declval<Adaptor>()(std::declval<Args>()...); };
    } // namespace concepts

    template<typename T, typename U>
    using like_t = decltype(std::forward_like<T>(std::declval<U>()));

    // Type that represents the pipe structure
    template<typename Lhs, typename Rhs>
    struct Pipe : ImageAdaptorClosure {
        [[no_unique_address]] Lhs _lhs;
        [[no_unique_address]] Rhs _rhs;

        template<typename Tp, typename Up>
        constexpr Pipe(Tp &&lhs, Up &&rhs) : _lhs(std::forward<Tp>(lhs)), _rhs(std::forward<Up>(rhs)) {
        }

        // Invoke _rhs(_lhs(img)) according to the value category of this image adaptor closure object
#if __cpp_explicit_this_parameter >= 202110L
        template<typename Self, typename Image>
            requires PipeInvocable<like_t<Self, Lhs>, like_t<Self, Rhs>, Image>
        constexpr auto operator()(this Self &&self, Image &&img) {
            return std::forward<Self>(self)._rhs(std::forward<Self>(self)._lhs(std::forward<Image>(img)));
        }
#elif
        template<typename Image>
            requires PipeInvocable<Lhs const &, Rhs const &, Image>
        constexpr auto operator()(Image &&img) const & {
            return _rhs(_lhs(std::forward<Image>(img)));
        }

        template<typename Image>
            requires PipeInvocable<Lhs &&, Rhs &&, Image>
        constexpr auto operator()(Image &&img) && {
            return std::move(_rhs)(std::move(_lhs)(std::forward<Image>(img)));
        }

        template<typename Image>
        constexpr auto operator()(Image &&img) const && =
                DELETE_DEF("Invoking a pipe object with const r-value references is prohibited");
#endif
    };

    // img | adaptor is equivalent to adaptor(img)
    template<typename Self, typename Image>
        requires IsImageAdaptorClosure<Self> and AdaptorInvocable<Self, Image>
    constexpr auto operator|(Image &&img, Self &&self) {
        return std::forward<Self>(self)._rhs(std::forward<Image>(img));
    }

    // adaptor1 | adaptor2 is equivalent to adaptor2(adaptor1)
    template<typename Lhs, typename Rhs>
        requires IsImageAdaptorClosure<Lhs> and IsImageAdaptorClosure<Rhs>
    constexpr auto operator|(Lhs &&lhs, Rhs &&rhs) {
        return Pipe<std::decay_t<Lhs>, std::decay_t<Rhs>>{std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)};
    }

    template<typename T>
    concept IsArithmetic = std::is_arithmetic_v<T>;

    // Helper classes and concepts
    template<class Reference>
    struct arrow_proxy {
        Reference r;
        auto operator->() -> Reference * {
            return &r;
        }
    };

    template<typename T>
    concept ImplsDistanceTo = requires(T const it) { it.distance_to(std::declval<T>()); };

    template<typename>
    struct infer_difference_type {
        using type = std::ptrdiff_t;
    };
    template<typename T>
        requires ImplsDistanceTo<T>
    struct infer_difference_type<T> {
        static const T &it_;
        using type = decltype(it_.distance_to(it_));
    };
    template<typename T>
    using infer_difference_type_t = typename infer_difference_type<T>::type;

    template<typename T>
    struct infer_value_type {
        static const T &it_;
        using type = std::remove_cvref_t<decltype(*it_)>;
    };
    template<typename T>
        requires requires { typename T::value_type; }
    struct infer_value_type<T> {
        using type = typename T::value_type;
    };
    template<typename T>
    using infer_value_type_t = typename infer_value_type<T>::type;

    template<typename T>
    concept ImplsIncrement = requires(T t) { t.increment(); };
    template<typename T>
    concept ImplsDecrement = requires(T t) { t.decrement(); };
    template<typename T>
    concept ImplsAdvance = requires(T it, infer_difference_type_t<T> const offset) { it.advance(offset); };
    template<typename T>
    concept ImplsEqualTo = requires(T const it) {
        { it.equal_to(it) } -> std::same_as<bool>;
    };
    template<typename T>
    concept IsRandomAccessIterator = ImplsAdvance<T> and ImplsDistanceTo<T>;
    template<typename T>
    concept IsBidirectionalIterator = IsRandomAccessIterator<T> and ImplsDecrement<T>;
    template<typename T>
    concept IsSinglePassIterator = static_cast<bool>(T::single_pass_iterator);
    template<typename Arg, typename It>
    concept DifferenceTypeArg = std::convertible_to<Arg, infer_difference_type_t<It>>;

    // A modernized version of boost::iterator_facade
    // Helps to reduce the boilerplate code when writing iterators
    template<typename Derived>
    class iterator_facade {
    public:
        using self_type = Derived;

    private:
        auto self() -> self_type & {
            return static_cast<self_type &>(*this);
        }
        auto self() const -> self_type const & {
            return static_cast<self_type const &>(*this);
        }

    public:
        decltype(auto) operator*() const {
            return self().dereference();
        }
        auto operator->() const {
            decltype(auto) ref = **this;
            if constexpr (std::is_reference_v<decltype(ref)>) {
                return std::addressof(ref);
            } else {
                return arrow_proxy(std::move(ref));
            }
        }

        friend auto operator==(self_type const &lhs, self_type const &rhs) -> bool {
            return lhs.equal_to(rhs);
        }

        auto operator++() -> self_type & {
            if constexpr (ImplsIncrement<self_type>) {
                self().increment();
            } else {
                static_assert(ImplsAdvance<self_type>,
                              "Iterator subclass must provide either .advance() or .increment() functions");
                self().advance(1);
            }
            return self();
        }
        auto operator++(int) -> self_type {
            auto copy = self();
            ++*this;
            return copy;
        }
        auto operator--() -> self_type & {
            if constexpr (ImplsDecrement<self_type>) {
                self().decrement();
            } else {
                static_assert(ImplsAdvance<self_type>,
                              "Iterator subclass must provide either .advance() or .decrement() functions");
                self().advance(-1);
            }
            return self();
        }
        auto operator--(int) -> self_type
            requires ImplsDecrement<self_type>
        {
            auto copy = self();
            --*this;
            return copy;
        }

        friend auto operator+=(self_type &self, DifferenceTypeArg<self_type> auto offset) -> self_type &
            requires ImplsAdvance<self_type>
        {
            self.advance(offset);
            return self;
        }
        friend auto operator+(self_type self, DifferenceTypeArg<self_type> auto offset) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self += offset;
        }
        friend auto operator+(DifferenceTypeArg<self_type> auto offset, self_type self) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self += offset;
        }
        friend auto operator-(self_type self, DifferenceTypeArg<self_type> auto offset) -> self_type
            requires ImplsAdvance<self_type>
        {
            return self + -offset;
        }
        friend auto operator-=(self_type &self, DifferenceTypeArg<self_type> auto offset) -> self_type &
            requires ImplsAdvance<self_type>
        {
            return self = self - offset;
        }
        decltype(auto) operator[](DifferenceTypeArg<self_type> auto offset)
            requires ImplsAdvance<self_type>
        {
            return *(self() + offset);
        }
        decltype(auto) operator[](DifferenceTypeArg<self_type> auto offset) const
            requires ImplsAdvance<self_type>
        {
            return *(self() + offset);
        }
        friend auto operator-(self_type const &lhs, self_type const &rhs)
            requires ImplsDistanceTo<self_type>
        {
            return rhs.distance_to(lhs);
        }
        friend auto operator<=>(self_type const &lhs, self_type const &rhs)
            requires ImplsDistanceTo<self_type>
        {
            return (lhs - rhs) <=> 0;
        }
    };
} // namespace mgil::details

template<typename It>
    requires std::derived_from<It, mgil::details::iterator_facade<It>>
struct std::iterator_traits<It> {
    static const It &it_;
    using reference = decltype(*it_);
    using pointer = decltype(it_.operator->());
    using difference_type = mgil::details::infer_difference_type_t<It>;
    using value_type = mgil::details::infer_value_type_t<It>;

    using iterator_category =
            conditional_t<mgil::details::IsRandomAccessIterator<It>, random_access_iterator_tag,
                          conditional_t<mgil::details::IsBidirectionalIterator<It>, bidirectional_iterator_tag,
                                        conditional_t<mgil::details::IsSinglePassIterator<It>, input_iterator_tag,
                                                      forward_iterator_tag>>>;
    using iterator_concept = iterator_category;
};

namespace mgil::details {
    template<typename Loc, typename XIt, typename YIt, template <typename> class PointType>
    class position_locator_base {
    public:
        using x_iterator = XIt;
        using y_iterator = YIt;

        using value_type = typename std::iterator_traits<x_iterator>::value_type;
        using reference = typename std::iterator_traits<x_iterator>::reference;
        using coordinate_type = typename std::iterator_traits<x_iterator>::difference_type;
        using difference_type = PointType<coordinate_type>;
        using point_type = PointType<coordinate_type>;
        using x_coordinate_type = coordinate_type;
        using y_coordinate_type = coordinate_type;

    private:
        auto self() -> Loc & { return static_cast<Loc &>(*this); }
        auto self() const -> Loc const & { return static_cast<Loc const &>(*this); }

    public:

    };
}

// Traits
namespace mgil::inline traits {
    template<typename T>
    struct ChannelTraitsImpl {};

    template<typename T>
        requires std::integral<T> or std::floating_point<T>
    struct ChannelTraitsImpl<T> {
        using value_type = T;
        using reference = value_type &;
        using pointer = value_type *;
        using const_reference = value_type const &;
        using const_pointer = value_type const *;
        static constexpr bool is_mutable = true;
        static constexpr auto minValue() -> value_type {
            return std::numeric_limits<value_type>::min();
        }
        static constexpr auto maxValue() -> value_type {
            return std::numeric_limits<value_type>::max();
        }
        static constexpr auto setValue(T &t, value_type v) -> void {
            t = v;
        }
        static constexpr auto getValue(T const &t) -> T {
            return t;
        }
    };

    template<typename T>
        requires std::integral<T> or std::floating_point<T>
    struct ChannelTraitsImpl<T const> : ChannelTraitsImpl<T> {
        using reference = T const &;
        using pointer = T const *;
        static constexpr bool is_mutable = false;
    };

    struct channel_tag {};

    template<typename T>
        requires std::is_class_v<T> and std::derived_from<T, channel_tag>
    struct ChannelTraitsImpl<T> {
        using value_type = typename T::value_type;
        using reference = typename T::reference;
        using pointer = typename T::pointer;
        using const_reference = typename T::const_reference;
        using const_pointer = typename T::const_pointer;
        static constexpr bool is_mutable = T::is_mutable;
        static constexpr auto minValue() -> value_type {
            return typename T::minValue();
        }
        static constexpr auto maxValue() -> value_type {
            return typename T::maxValue();
        }
        static constexpr auto setValue(T &t, value_type v) -> void {
            t.setValue(v);
        }
        static constexpr auto getValue(T const &t) -> value_type {
            return t.getValue();
        }
    };

    template<typename T>
    struct ChannelTraits : ChannelTraitsImpl<T> {};

    template<typename T>
    struct ChannelTraits<T &> : ChannelTraits<T> {};

    template<typename T>
    struct ChannelTraits<T const &> : ChannelTraits<T> {};

    struct pixel_iterator_tag {};

    template<typename T>
    struct PixelIteratorTraits {};

    template<typename T>
        requires std::derived_from<T, pixel_iterator_tag>
    struct PixelIteratorTraits<T> {
        using const_iterator_type = typename T::const_iterator_type;
        static constexpr bool is_mutable = T::is_mutable;
    };
} // namespace mgil::inline traits

// Concepts for pixels, images, etc.
namespace mgil::inline concepts {
    // A point defined the location of a pixel inside an image.
    template<typename T>
    concept IsPoint = std::regular<T> && requires(T t, T const ct, T u, int i) {
        typename T::value_type;
        { t.x() } -> std::same_as<typename T::value_type &>;
        { t.y() } -> std::same_as<typename T::value_type &>;
        { ct.x() } -> std::same_as<typename T::value_type const &>;
        { ct.y() } -> std::same_as<typename T::value_type const &>;
        { t + u } -> std::same_as<T>;
        { t - u } -> std::same_as<T>;
        { t *u } -> std::same_as<T>;
        { t / u } -> std::same_as<T>;
        { t + i } -> std::same_as<T>;
        { t - i } -> std::same_as<T>;
        { t *i } -> std::same_as<T>;
        { t / i } -> std::same_as<T>;
    };

    // A channel indicates the intensity of a color component
    template<typename T>
    concept IsChannel = std::regular<T> and requires(T t, typename ChannelTraits<T>::value_type v) {
        typename ChannelTraits<T>::value_type;
        typename ChannelTraits<T>::reference;
        typename ChannelTraits<T>::pointer;
        typename ChannelTraits<T>::const_reference;
        typename ChannelTraits<T>::const_pointer;
        { ChannelTraits<T>::is_mutable } -> std::convertible_to<bool>;
        { ChannelTraits<T>::minValue() } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
        { ChannelTraits<T>::maxValue() } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
        { ChannelTraits<T>::setValue(t, v) };
        { ChannelTraits<T>::getValue(t) } -> std::convertible_to<typename ChannelTraits<T>::value_type>;
    };

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(IsChannel<int>);
    static_assert(IsChannel<float>);
    static_assert(IsChannel<double>);
#endif

    template<typename T, typename U>
    concept IsChannelsCompatible =
            std::same_as<typename ChannelTraits<T>::value_type, typename ChannelTraits<U>::value_type> and
            IsChannel<T> and IsChannel<U>;

    template<typename Src, typename Dst>
    concept IsChannelsConvertible = IsChannel<Src> and IsChannel<Dst> and requires(Src src) {
        { ChannelTraits<Src>::template convertTo<Dst>(src) } -> std::same_as<Dst>;
    };

    // Color Space and Layout concepts
    template<typename T>
    concept IsColorSpace = details::is_specialization_of_v<T, details::type_list> and (T::size >= 1 and T::size <= 4);

    struct color_space_component_tag {};

    template<typename T>
    concept IsColorSpaceComponent = std::derived_from<T, color_space_component_tag>;

    template<typename ColorSpace_, typename ChannelMapping_>
        requires details::is_specialization_of_nttp_v<ChannelMapping_, details::index_list> and
                 details::is_specialization_of_v<ColorSpace_, details::type_list> and
                 (ColorSpace_::template allConformTo<[]<typename T>() { return IsColorSpaceComponent<T>; }>()) and
                 (ColorSpace_::size == ChannelMapping_::size)
    struct ColorSpaceLayout {
        using color_space = ColorSpace_;
        using channel_mapping = ChannelMapping_;
        using mapped_color_space = details::rearrange_type_list_t<ColorSpace_, ChannelMapping_>;
    };

    template<typename T>
    concept IsLayout = details::is_specialization_of_v<T, ColorSpaceLayout>;

    template<typename T, typename U>
    concept IsLayoutCompatible =
            IsLayout<T> and IsLayout<U> and std::same_as<typename T::color_space, typename U::color_space>;

    // A pixel is a set of channels defining the color at a given point in an image
    template<typename T>
    concept IsPixel = requires(T t, T u, T const ct, std::size_t i) {
        typename T::layout_type;
        typename T::channel_type;
        typename T::value_type;
        typename T::reference;
        typename T::const_reference;

        requires IsLayout<typename T::layout_type>;
        requires IsChannel<typename T::channel_type>;

        { t.get(i) } -> std::convertible_to<typename T::channel_type const &>;
        { t.template getSemantic<0>() } -> std::convertible_to<typename T::channel_type const &>;
        { t.set(i, std::declval<typename T::channel_type>()) };
        { t[i] } -> std::convertible_to<typename T::channel_type &>;
        { ct[i] } -> std::convertible_to<typename T::channel_type const &>;
        { t + u } -> std::convertible_to<T>;
        { t - u } -> std::convertible_to<T>;
        { t *u } -> std::convertible_to<T>;
        { t / u } -> std::convertible_to<T>;
        { t + std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t - std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t *std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
        { t / std::declval<typename T::channel_type>() } -> std::convertible_to<T>;
    } and std::equality_comparable<T>;

    template<typename T, typename U>
    concept IsPixelsCompatible =
            IsPixel<T> and IsPixel<U> and IsChannelsCompatible<typename T::channel_type, typename U::channel_type> and
            IsLayoutCompatible<typename T::layout_type, typename U::layout_type>;

    template<typename T, typename U>
    concept IsPixelsConvertible = IsPixel<T> and IsPixel<U> and
                                  (IsChannelsConvertible<typename T::channel_type, typename U::channel_type> or
                                   IsChannelsCompatible<typename T::channel_type, typename U::channel_type>) and
                                  IsLayoutCompatible<typename T::layout_type, typename U::layout_type>;

    // Iterator concepts
    // Models a random-access iterator over pixels
    template<typename It>
    concept IsPixelIterator = std::random_access_iterator<It> and IsPixel<std::iter_value_t<It>> and requires {
        typename PixelIteratorTraits<It>::const_iterator_type;
        { PixelIteratorTraits<It>::is_mutable } -> std::convertible_to<bool>;
    };

    // Encapsulates on-th-fly pixel transformation
    template<typename T>
    concept IsPixelDereferenceAdaptor =
            std::default_initializable<T> and std::copyable<T> and std::is_copy_assignable_v<T> and requires(T t) {
                typename T::const_adaptor;
                typename T::value_type;
                typename T::result_type;
                typename T::reference;
                typename T::const_reference;

                requires IsPixel<typename T::value_type>;
                requires IsPixel<std::remove_cvref_t<typename T::reference>>;
                requires IsPixel<std::remove_cvref_t<typename T::const_reference>>;

                { T::is_mutable } -> std::convertible_to<bool>;
            };
    template<typename ConstT, typename Value, typename Reference, typename ConstReference, typename ArgType,
             typename ResultType, bool IsMutable>
    struct deref_base {
        using argument_type = ArgType;
        using result_type = ResultType;
        using const_adaptor = ConstT;
        using value_type = Value;
        using reference = Reference;
        using const_reference = ConstReference;
        static constexpr bool is_mutable = IsMutable;
    };
    template<typename D1, typename D2>
        requires IsPixelDereferenceAdaptor<D1> and IsPixelDereferenceAdaptor<D2>
    class deref_compose
        : public deref_base<deref_compose<typename D1::const_adaptor, typename D2::const_adaptor>,
                            typename D1::value_type, typename D1::reference, typename D1::const_reference,
                            typename D2::argument_type, typename D1::result_type, D1::is_mutable and D2::is_mutable> {
    public:
        D1 fn1_;
        D2 fn2_;

        using argument_type = typename D2::argument_type;
        using result_type = typename D1::result_type;

        deref_compose() = default;
        deref_compose(D1 const &d1, D2 const &d2) : fn1_(d1), fn2_(d2) {
        }
        deref_compose(deref_compose const &that) : fn1_(that.fn1_), fn2_(that.fn2_) {
        }

        template<typename D1_, typename D2_>
        explicit deref_compose(deref_compose<D1_, D2_> const &that) : fn1_(that.fn1_), fn2_(that.fn2_) {
        }

        auto operator()(argument_type x) const {
            return fn1_(fn2_(x));
        }
        auto operator()(argument_type x) {
            return fn1_(fn2_(x));
        }
    };

    // Allow changing the stride between elements (e.g. column iterators)
    template<typename It>
    concept IsStepIterator = std::forward_iterator<It> and IsPixel<std::iter_value_t<It>> and requires(It it) {
        { it.setStep(std::declval<int>()) };
    };
    template<typename It>
    concept IsMutableStepIterator = IsStepIterator<It> and IsPixel<std::iter_value_t<It>> and
                                    std::indirectly_writable<It, std::iter_value_t<It>>;

    // Pixel iterators that advance in bits
    template<typename It>
    concept IsMemoryBasedIterator = std::random_access_iterator<It> and IsPixel<std::iter_value_t<It>> and
                                    requires(It it, std::ptrdiff_t diff) {
                                        { it.bitsStep() } -> std::same_as<std::ptrdiff_t>;
                                        { it.distance(it) } -> std::same_as<std::ptrdiff_t>;
                                        { it.advance(diff) } -> std::same_as<void>;
                                        { it.advanced(diff) } -> std::same_as<It>;
                                        { it.advancedRef(diff) } -> std::same_as<typename It::reference>;
                                    };

    // Pixel locator concepts
    template<typename Loc>
    concept Is2DLocator = std::regular<Loc> and requires(Loc loc, Loc loc2) {
        typename Loc::value_type;
        typename Loc::reference;
        typename Loc::difference_type;
        typename Loc::const_locator;
        typename Loc::cached_location_type;
        typename Loc::point_type;

        typename Loc::x_iterator;
        typename Loc::y_iterator;
        typename Loc::x_coordinate_type;
        typename Loc::y_coordinate_type;

        { Loc::is_mutable } -> std::same_as<bool>;

        { loc += std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc &>;
        { loc -= std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc &>;
        { loc + std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc>;
        { loc - std::declval<typename Loc::difference_type>() } -> std::convertible_to<Loc>;
        { *loc } -> std::convertible_to<typename Loc::reference>;
        { loc[std::declval<typename Loc::difference_type>()] } -> std::convertible_to<typename Loc::reference>;
        {
            Loc::cacheLocation(std::declval<typename Loc::difference_type>())
        } -> std::convertible_to<typename Loc::cached_location_type>;
        { loc[std::declval<typename Loc::cached_location_type>()] } -> std::convertible_to<typename Loc::reference>;

        { loc.x() } -> std::convertible_to<typename Loc::x_iterator const &>;
        { loc.y() } -> std::convertible_to<typename Loc::y_iterator const &>;
        { loc.xAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<typename Loc::x_iterator>;
        { loc.yAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<typename Loc::y_iterator>;
        { loc.xyAt(std::declval<typename Loc::difference_type>()) } -> std::convertible_to<Loc>;
        {
            loc.xAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::x_iterator>;
        {
            loc.yAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::y_iterator>;
        {
            loc.xyAt(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<Loc>;
        {
            loc(std::declval<typename Loc::x_coordinate_type>(), std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::reference>;
        {
            loc.cache_location(std::declval<typename Loc::x_coordinate_type>(),
                               std::declval<typename Loc::y_coordinate_type>())
        } -> std::convertible_to<typename Loc::cache_location_type>;

        { loc.is1DTraversable(std::declval<typename Loc::x_coordinate_type>()) } -> std::convertible_to<bool>;
        {
            loc.yDistanceTo(loc2, std::declval<typename Loc::x_coordinate_type>())
        } -> std::convertible_to<typename Loc::y_coordinate_type>;
    };

    template<typename Loc>
    concept IsPixelLocator = Is2DLocator<Loc> and requires {
        requires IsPixel<typename Loc::value_type>;
        requires IsPixel<typename Loc::reference>;
        requires IsPoint<typename Loc::difference_type>;
        requires IsPoint<typename Loc::point_type>;

        typename Loc::coordinate_type;
        requires std::same_as<typename Loc::x_coordinate_type, typename Loc::y_coordinate_type>;

        typename Loc::transposed_type;
    };

    template<typename Loc>
    concept IsPixelLocatorHasTransposedType = IsPixelLocator<Loc> and requires { typename Loc::transposed_type; };
} // namespace mgil::inline concepts

// The main namespace
namespace mgil {
    // Color Spaces and Layouts
    struct red_color_t : color_space_component_tag {};
    struct green_color_t : color_space_component_tag {};
    struct blue_color_t : color_space_component_tag {};
    struct alpha_color_t : color_space_component_tag {};
    struct gray_color_t : color_space_component_tag {};
    struct cyan_color_t : color_space_component_tag {};
    struct magenta_color_t : color_space_component_tag {};
    struct yellow_color_t : color_space_component_tag {};
    struct black_color_t : color_space_component_tag {};

    // Color Space Models
    using gray_t = details::type_list<gray_color_t>;
    using rgb_t = details::type_list<red_color_t, green_color_t, blue_color_t>;
    using rgba_t = details::type_list<red_color_t, green_color_t, blue_color_t, alpha_color_t>;
    using cmyk_t = details::type_list<cyan_color_t, magenta_color_t, yellow_color_t, black_color_t>;

    // Color Space Layouts
    using gray_layout_t = ColorSpaceLayout<gray_t, details::index_list<0>>;
    using rgb_layout_t = ColorSpaceLayout<rgb_t, details::index_list<0, 1, 2>>;
    using bgr_layout_t = ColorSpaceLayout<rgb_t, details::index_list<2, 1, 0>>;
    using rgba_layout_t = ColorSpaceLayout<rgba_t, details::index_list<0, 1, 2, 3>>;
    using bgra_layout_t = ColorSpaceLayout<rgba_t, details::index_list<2, 1, 0, 3>>;
    using abgr_layout_t = ColorSpaceLayout<rgba_t, details::index_list<3, 2, 1, 0>>;
    using argb_layout_t = ColorSpaceLayout<rgba_t, details::index_list<3, 0, 1, 2>>;
    using cmyk_layout_t = ColorSpaceLayout<cmyk_t, details::index_list<0, 1, 2, 3>>;

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(IsColorSpace<gray_t> && IsColorSpace<rgb_t> and IsColorSpace<rgba_t> and IsColorSpace<cmyk_t>);
    static_assert(std::same_as<details::type_list<gray_color_t>, gray_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<red_color_t, green_color_t, blue_color_t>,
                               rgb_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<blue_color_t, green_color_t, red_color_t>,
                               bgr_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<blue_color_t, green_color_t, red_color_t, alpha_color_t>,
                               bgra_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<alpha_color_t, blue_color_t, green_color_t, red_color_t>,
                               abgr_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<alpha_color_t, red_color_t, green_color_t, blue_color_t>,
                               argb_layout_t::mapped_color_space>);
    static_assert(std::same_as<details::type_list<cyan_color_t, magenta_color_t, yellow_color_t, black_color_t>,
                               cmyk_layout_t::mapped_color_space>);
#endif
} // namespace mgil

namespace mgil {
    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
    struct RescopedChannel;

    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
        requires std::integral<T> or std::floating_point<T>
    struct RescopedChannel<T, Min, Max> : channel_tag {
        T value;

        using value_type = T;
        using reference = T &;
        using pointer = T *;
        using const_reference = T const &;
        using const_pointer = T const *;
        static constexpr bool is_mutable = true;
        constexpr RescopedChannel() : value(Min) {
        }
        constexpr RescopedChannel(T value) : value(value) {
        }
        static constexpr auto minValue() -> value_type {
            return Min;
        }
        static constexpr auto maxValue() -> value_type {
            return Max;
        }
        [[nodiscard]] constexpr auto getValue() const -> value_type {
            return value;
        }
        constexpr auto setValue(value_type v) -> void {
            value = v;
        }
        friend auto operator<=>(RescopedChannel const &lhs, RescopedChannel const &rhs) -> std::strong_ordering {
            return lhs.value <=> rhs.value;
        }
        friend auto operator==(RescopedChannel const &lhs, RescopedChannel const &rhs) -> bool {
            return lhs.value == rhs.value;
        }
        friend auto operator!=(RescopedChannel const &lhs, RescopedChannel const &rhs) -> bool {
            return lhs.value != rhs.value;
        }
        explicit operator T() const {
            return value;
        }
        friend auto operator<<(std::ostream &os, RescopedChannel const &rhs) -> std::ostream & {
            os << rhs.value;
            return os;
        }
    };

    template<IsChannel T, typename ChannelTraits<T>::value_type Min, typename ChannelTraits<T>::value_type Max>
        requires std::is_class_v<T>
    struct RescopedChannel<T, Min, Max> : T {
        static constexpr auto minValue() -> typename ChannelTraits<T>::value_type {
            return ChannelTraits<T>::minValue();
        }
        static constexpr auto maxValue() -> typename ChannelTraits<T>::value_type {
            return ChannelTraits<T>::maxValue();
        }
        [[nodiscard]] constexpr auto getValue() const -> typename ChannelTraits<T>::value_type {
            return getValue();
        }
        constexpr auto setValue(typename ChannelTraits<T>::value_type v) -> void {
            return setValue(v);
        }
    };

    using Float_01 = RescopedChannel<float, 0.0f, 1.0f>;
    using Double_01 = RescopedChannel<double, 0.0, 1.0>;
    using UInt8_0255 = RescopedChannel<std::uint8_t, 0, 255>;
    using UInt16_0255 = RescopedChannel<std::uint16_t, 0, 255>;
    using UInt32_0255 = RescopedChannel<std::uint32_t, 0, 255>;
    using UInt64_0255 = RescopedChannel<std::uint64_t, 0, 255>;

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(IsChannel<Float_01>);
    static_assert(IsChannel<Double_01>);
    static_assert(IsChannel<UInt8_0255>);
    static_assert(IsChannel<UInt16_0255>);
    static_assert(IsChannel<UInt32_0255>);
    static_assert(IsChannel<UInt64_0255>);
#endif
} // namespace mgil

namespace mgil {
    // Pixel class
    template<typename ChannelType, typename Layout>
        requires IsChannel<ChannelType> and IsLayout<Layout>
    class Pixel {
    public:
        using layout_type = Layout;
        using channel_type = ChannelType;
        using value_type = Pixel;
        using reference = value_type &;
        using const_reference = value_type const &;

    private:
        static constexpr std::size_t size = Layout::color_space::size;
        std::array<ChannelType, size> _components = {};

        template<std::size_t Index>
        static constexpr auto indexOfNthSemanticComponent() -> std::size_t {
            return details::find_type_v<typename layout_type::color_space::template NthType<Index>,
                                        typename layout_type::mapped_color_space>;
        }

    public:
        // Default constructor
        Pixel() = default;
        // Copy constructor
        Pixel(Pixel const &) = default;
        // Move constructor
        Pixel(Pixel &&) noexcept(std::is_nothrow_move_constructible_v<ChannelType>) = default;
        // Copy assignment
        auto operator=(Pixel const &) -> Pixel & = default;
        // Move assignment
        auto operator=(Pixel &&) noexcept(std::is_nothrow_move_assignable_v<ChannelType>) -> Pixel & = default;

        // Initialize the underlying array by setting all the elements v
        explicit constexpr Pixel(ChannelType v) : _components() {
            _components.fill(v);
        }

        // Initializing the underlying array by a pack
        template<typename... Ts>
            requires(std::convertible_to<Ts, ChannelType> && ...) and (sizeof...(Ts) == size)
        explicit constexpr Pixel(Ts... components) : _components{static_cast<ChannelType>(components)...} {
        }

        // List initialization
        constexpr Pixel(std::initializer_list<ChannelType> components) : _components(components) {
        }

        // Initializing from a compatible pixel
        template<typename SrcPixel>
            requires IsPixelsConvertible<SrcPixel, Pixel>
        explicit constexpr Pixel(SrcPixel const &src) : Pixel(convertPixelTo<SrcPixel, Pixel>(src)) {
        }

        template<typename SrcPixel, typename DstPixel>
            requires IsPixelsConvertible<SrcPixel, DstPixel>
        static constexpr auto convertPixelTo(SrcPixel const &src) -> DstPixel {
            using SrcChannelType = typename SrcPixel::channel_type;
            using DstChannelType = typename DstPixel::channel_type;
            using SrcLayoutType = typename SrcPixel::layout_type;
            using DstLayoutType = typename DstPixel::layout_type;
            DstPixel dst;
            if constexpr (std::same_as<typename SrcPixel::layout_type, DstLayoutType>) {
                for (std::size_t i = 0; i < size; i++) {
                    if constexpr (std::same_as<SrcChannelType, DstChannelType>) {
                        dst.set(i, src.get(i));
                    } else {
                        dst.set(i, ChannelTraits<SrcChannelType>::template convertTo<DstChannelType>(src.get(i)));
                    }
                }
            } else {
                details::mp::for_each<DstLayoutType::mapped_color_space::getMPVector()>([&]<details::mp::info meta> {
                    constexpr std::size_t src_index = details::find_type_v<details::mp::type_of<meta>,
                                                                           typename SrcLayoutType::mapped_color_space>;
                    constexpr std::size_t dst_index = details::find_type_v<details::mp::type_of<meta>,
                                                                           typename DstLayoutType::mapped_color_space>;
                    if constexpr (std::same_as<SrcChannelType, DstChannelType>) {
                        dst.set(dst_index, src.get(src_index));
                    } else {
                        dst.set(dst_index,
                                ChannelTraits<SrcChannelType>::template convertTo<DstChannelType>(src.get(src_index)));
                    }
                });
            }
            return dst;
        }

        friend constexpr auto operator<=>(Pixel const &lhs, Pixel const &rhs) -> std::strong_ordering {
            return lhs._components <=> rhs._components;
        }
        friend constexpr auto operator==(Pixel const &lhs, Pixel const &rhs) -> bool {
            return lhs._components == rhs._components;
        }
        friend constexpr auto operator!=(Pixel const &lhs, Pixel const &rhs) -> bool {
            return lhs._components != rhs._components;
        }

        friend constexpr auto operator+(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] + rhs._components[i];
            }
            return result;
        }
        friend constexpr auto operator-(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] - rhs._components[i];
            }
            return result;
        }
        friend constexpr auto operator*(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] * rhs._components[i];
            }
            return result;
        }
        friend constexpr auto operator/(Pixel const &lhs, Pixel const &rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] / rhs._components[i];
            }
            return result;
        }

        friend constexpr auto operator+(Pixel const &lhs, channel_type rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] + rhs;
            }
            return result;
        }
        friend constexpr auto operator-(Pixel const &lhs, channel_type rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] - rhs;
            }
            return result;
        }
        friend constexpr auto operator*(Pixel const &lhs, channel_type rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] * rhs;
            }
            return result;
        }
        friend constexpr auto operator/(Pixel const &lhs, channel_type rhs) -> Pixel {
            Pixel result;
            for (std::size_t i = 0; i < size; i++) {
                result._components[i] = lhs._components[i] / rhs;
            }
            return result;
        }

        [[nodiscard]] constexpr auto get(std::size_t index) noexcept -> ChannelType & {
            return _components[index];
        }
        [[nodiscard]] constexpr auto get(std::size_t index) const noexcept -> ChannelType const & {
            return _components[index];
        }
        template<std::size_t Index>
        [[nodiscard]] constexpr auto getSemantic() noexcept -> ChannelType & {
            return _components[indexOfNthSemanticComponent<Index>()];
        }
        template<std::size_t Index>
        [[nodiscard]] constexpr auto getSemantic() const noexcept -> ChannelType const & {
            return _components[indexOfNthSemanticComponent<Index>()];
        }
        template<typename Component>
            requires IsColorSpaceComponent<Component> and
                     details::contain_type_v<Component, typename layout_type::color_space>
        [[nodiscard]] constexpr auto get() noexcept -> ChannelType & {
            return _components[details::find_type_v<Component, typename layout_type::mapped_color_space>];
        }
        template<typename Component>
            requires IsColorSpaceComponent<Component> and
                     details::contain_type_v<Component, typename layout_type::color_space>
        [[nodiscard]] constexpr auto get() const noexcept -> ChannelType const & {
            return _components[details::find_type_v<Component, typename layout_type::mapped_color_space>];
        }
        constexpr auto set(std::size_t index, ChannelType v) -> void {
            _components[index] = v;
        }
        constexpr auto operator[](std::size_t const index) const noexcept -> ChannelType const & {
            return _components[index];
        }
        constexpr auto operator[](std::size_t const index) -> ChannelType & {
            return _components[index];
        }
        static constexpr auto getSize() noexcept -> std::size_t {
            return size;
        }

        friend auto operator<<(std::ostream &out, Pixel const &pixel) -> std::ostream & {
            out << std::string{"Pixel{"};
            if (size > 1) {
                for (std::size_t i = 0; i < getSize() - 1; i++) {
                    out << pixel.get(i) << ",";
                }
                out << pixel.get(getSize() - 1) << "}";
            } else {
                out << pixel.get(0) << "}";
            }
            return out;
        }
    };

    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &pixel) noexcept -> Channel const & {
        return pixel.get(Index);
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &pixel) noexcept -> Channel & {
        return pixel.get(Index);
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &&pixel) noexcept -> Channel const && {
        return std::move(pixel.get(Index));
    }
    template<std::size_t Index, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &&pixel) noexcept -> Channel && {
        return std::move(pixel.get(Index));
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &pixel) noexcept -> Channel const & {
        return pixel.template get<Component>();
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &pixel) noexcept -> Channel & {
        return pixel.template get<Component>();
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> const &&pixel) noexcept -> Channel const && {
        return std::move(pixel.template get<Component>());
    }
    template<typename Component, typename Channel, typename Layout>
    constexpr auto get(Pixel<Channel, Layout> &&pixel) noexcept -> Channel && {
        return std::move(pixel.template get<Component>());
    }
} // namespace mgil

template<typename Channel, typename Layout>
struct std::tuple_size<mgil::Pixel<Channel, Layout>>
    : std::integral_constant<std::size_t, mgil::Pixel<Channel, Layout>::getSize()> {};
template<std::size_t Index, typename Channel, typename Layout>
struct std::tuple_element<Index, mgil::Pixel<Channel, Layout>> {
    using type = Channel;
};
template<typename Channel, typename Layout>
    requires mgil::IsChannel<Channel> and mgil::IsLayout<Layout>
struct std::formatter<mgil::Pixel<Channel, Layout>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(mgil::Pixel<Channel, Layout> const &pixel, std::format_context &ctx) const {
        std::format_to(ctx.out(), "Pixel{{");
        if (mgil::Pixel<Channel, Layout>::getSize() > 1) {
            for (std::size_t i = 0; i < mgil::Pixel<Channel, Layout>::getSize() - 1; i++) {
                std::format_to(ctx.out(), "{},", pixel.get(i));
            }
            std::format_to(ctx.out(), "{}", pixel.get(mgil::Pixel<Channel, Layout>::getSize() - 1));
        } else {
            std::format_to(ctx.out(), "{}", pixel.get(0));
        }
        std::format_to(ctx.out(), "}}");
        return ctx.out();
    }
};
template<typename Channel>
    requires mgil::IsChannel<Channel> and std::is_class_v<Channel>
struct std::formatter<Channel> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(Channel const &channel, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "{}", mgil::ChannelTraits<Channel>::getValue(channel));
    }
};

namespace mgil {
#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(IsPixel<Pixel<int, rgb_layout_t>>);
    static_assert(IsPixelsConvertible<Pixel<int, abgr_layout_t>, Pixel<int, bgra_layout_t>>);
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4) ==
                  Pixel<int, rgba_layout_t>(Pixel<int, bgra_layout_t>(3, 2, 1, 4)));
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4).get<blue_color_t>() == 3);
    static_assert(Pixel<int, rgba_layout_t>(1, 2, 3, 4).get(1) == 2);
    static_assert(Pixel<int, abgr_layout_t>(1, 2, 3, 4).getSemantic<1>() == 3);
    static_assert([] {
        Pixel<int, rgba_layout_t> pixel(1, 2, 3, 4);
        auto [r, g, b, a] = pixel;
        return r == 1 and g == 2 and b == 3 and a == 4;
    }());
#endif
} // namespace mgil

// Point class
namespace mgil {
    template<typename ValueType>
        requires std::is_arithmetic_v<ValueType>
    class Point {
    private:
        ValueType x_ = {};
        ValueType y_ = {};

    public:
        using value_type = ValueType;

        constexpr Point() = default;
        constexpr Point(Point const &that) : x_(that.x_), y_(that.y_) {
        }
        constexpr Point(Point &&that) noexcept : x_(std::move(that.x_)), y_(std::move(that.y_)) {
        }
        constexpr Point(value_type x, value_type y) : x_(x), y_(y) {
        }
        constexpr Point(value_type v) : x_(v), y_(v) {
        }
        constexpr auto operator=(Point const &that) -> Point & = default;
        constexpr auto operator=(Point &&that) noexcept -> Point & {
            this->x_ = std::move(that.x_);
            this->y_ = std::move(that.y_);
            return *this;
        }

        constexpr auto x() noexcept -> value_type & {
            return x_;
        }
        constexpr auto y() noexcept -> value_type & {
            return y_;
        }
        [[nodiscard]] constexpr auto x() const noexcept -> value_type const & {
            return x_;
        }
        [[nodiscard]] constexpr auto y() const noexcept -> value_type const & {
            return y_;
        }

        friend constexpr auto operator+(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ + rhs.x_, lhs.y_ + rhs.y_);
        }
        friend constexpr auto operator-(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ - rhs.x_, lhs.y_ - rhs.y_);
        }
        friend constexpr auto operator*(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ * rhs.x_, lhs.y_ * rhs.y_);
        }
        friend constexpr auto operator/(Point const &lhs, Point const &rhs) -> Point {
            return Point(lhs.x_ / rhs.x_, lhs.y_ / rhs.y_);
        }
        friend constexpr auto operator+(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/(Point const &lhs, details::IsArithmetic auto rhs) -> Point {
            return Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }
        friend constexpr auto operator+(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/(details::IsArithmetic auto rhs, Point const &lhs) -> Point {
            return Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }

        friend constexpr auto operator+=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ + rhs.x_, lhs.y_ + rhs.y_);
        }
        friend constexpr auto operator-=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ - rhs.x_, lhs.y_ - rhs.y_);
        }
        friend constexpr auto operator*=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ * rhs.x_, lhs.y_ * rhs.y_);
        }
        friend constexpr auto operator/=(Point &lhs, Point const &rhs) -> Point & {
            return lhs = Point(lhs.x_ / rhs.x_, lhs.y_ / rhs.y_);
        }
        friend constexpr auto operator+=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ + rhs, lhs.y_ + rhs);
        }
        friend constexpr auto operator-=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ - rhs, lhs.y_ - rhs);
        }
        friend constexpr auto operator*=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ * rhs, lhs.y_ * rhs);
        }
        friend constexpr auto operator/=(Point &lhs, details::IsArithmetic auto rhs) -> Point & {
            return lhs = Point(lhs.x_ / rhs, lhs.y_ / rhs);
        }

        friend constexpr auto operator==(Point const &lhs, Point const &rhs) -> bool {
            return lhs.x_ == rhs.x_ and lhs.y_ == rhs.y_;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::strong_ordering
            requires std::same_as<std::strong_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::strong_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::strong_ordering::greater;
            }
            return std::strong_ordering::equal;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::weak_ordering
            requires std::same_as<std::weak_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::weak_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::weak_ordering::greater;
            }
            return std::weak_ordering::equivalent;
        }
        friend constexpr auto operator<=>(Point const &lhs, Point const &rhs) -> std::partial_ordering
            requires std::same_as<std::partial_ordering,
                                  decltype(std::declval<value_type>() <=> std::declval<value_type>())>
        {
            if (lhs.x_ == rhs.x_ and lhs.y_ == rhs.y_) {
                return std::partial_ordering::equivalent;
            }
            if (lhs.x_ < rhs.x_ or (lhs.x_ == rhs.x_ and lhs.y_ < rhs.y_)) {
                return std::partial_ordering::less;
            }
            if (lhs.x_ > rhs.x_ and (lhs.x_ == rhs.x_ and lhs.y_ > rhs.y_)) {
                return std::partial_ordering::greater;
            }
            return std::partial_ordering::unordered;
        }

        friend auto operator<<(std::ostream &os, Point const &rhs) -> std::ostream & {
            os << "Point(" << rhs.x_ << "," << rhs.y_ << ")";
            return os;
        }
    };
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> const &point) -> ValueType const & {
        if constexpr (Index == 0) {
            return point.x();
        }
        return point.y();
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> &point) -> ValueType & {
        if constexpr (Index == 0) {
            return point.x();
        }
        return point.y();
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> const &&point) -> ValueType const && {
        if constexpr (Index == 0) {
            return std::move(point.x());
        }
        return std::move(point.y());
    }
    template<std::size_t Index, typename ValueType>
        requires(Index == 0 or Index == 1)
    constexpr auto get(Point<ValueType> &&point) -> ValueType && {
        if constexpr (Index == 0) {
            return std::move(point.x());
        }
        return std::move(point.y());
    }
} // namespace mgil

template<typename ValueType>
struct std::tuple_size<mgil::Point<ValueType>> : std::integral_constant<std::size_t, 2> {};
template<std::size_t Index, typename ValueType>
    requires(Index == 0 or Index == 1)
struct std::tuple_element<Index, mgil::Point<ValueType>> {
    using type = ValueType;
};

template<typename ValueType>
struct std::formatter<mgil::Point<ValueType>> {
    constexpr auto parse(std::format_parse_context &ctx) {
        return ctx.begin();
    }
    auto format(mgil::Point<ValueType> const &point, std::format_context &ctx) const {
        return std::format_to(ctx.out(), "Point({},{})", point.x(), point.y());
    }
};

namespace mgil {
#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    static_assert(IsPoint<Point<int>>);
    static_assert([] {
        constexpr Point point{114, 514};
        auto [x, y] = point;
        return x == 114 and y == 514;
    }());
    static_assert([] {
        constexpr Point point1{2, 4};
        constexpr Point point2{3, 5};
        constexpr Point point3 = point1 * point2;
        auto [x, y] = point3;
        return x == 6 and y == 20;
    }());
#endif
} // namespace mgil

// Iterators
namespace mgil {
    // An iterator that remembers its current X and Y position and invokes a function object with it upon dereferencing.
    // Conforms to IsPixelIterator and IsStepIterator concepts
    template<typename Deref, typename PointType = Point<int>>
        requires IsPoint<PointType> and IsPixelDereferenceAdaptor<Deref>
    struct position_iterator : pixel_iterator_tag, details::iterator_facade<position_iterator<Deref>> {
    private:
        PointType pos_ = {};
        PointType step_ = {};
        Deref deref_ = {};

    public:
        using const_iterator_type = position_iterator const;
        using value_type = typename Deref::value_type;
        using difference_type = std::int64_t;
        using point_type = PointType;
        static constexpr bool is_mutable = false;

        constexpr position_iterator() = default;
        constexpr position_iterator(position_iterator const &that) :
            pos_{that.pos_}, step_{that.step_}, deref_{that.deref_} {
        }

        constexpr auto operator=(position_iterator const &that) -> position_iterator & {
            pos_ = that.pos_;
            step_ = that.step_;
            deref_ = that.deref_;
            return *this;
        }

        constexpr auto pos() const -> PointType const & {
            return pos_;
        }
        constexpr auto step() const -> PointType const & {
            return step_;
        }
        static_assert(std::invocable<Deref, difference_type>,
                      "The dereference function object must accept a difference_type as the only input");
        constexpr auto derefFn() const {
            return deref_;
        }

        constexpr auto setStep(difference_type step) -> void {
            step_ = step;
        }

        constexpr auto dereference() const {
            return deref_(pos_);
        }
        constexpr auto increment() -> void {
            pos_ += step_;
        }
        constexpr auto decrement() -> void {
            pos_ -= step_;
        }
        constexpr auto advance(difference_type d) -> void {
            pos_ += d * step_;
        }
        constexpr auto distance_to(position_iterator const &that) const -> difference_type {
            return ((that.pos_ - this->pos_) / step_).x();
        }
    };

#ifndef IMAGE_PROCESSING_NO_COMPILE_TIME_TESTING
    struct test_deref_adaptor
        : deref_base<test_deref_adaptor, Pixel<int, gray_layout_t>, Pixel<int, gray_layout_t> const,
                     Pixel<int, gray_layout_t> const &, Point<int>, Pixel<int, gray_layout_t>, false> {
        constexpr auto operator()(Point<int> point) const -> Pixel<int, gray_layout_t> {
            return Pixel<int, gray_layout_t>(point.x() + point.y());
        }
        constexpr auto operator=(test_deref_adaptor const &that) -> test_deref_adaptor & = default;
    };
    static_assert(IsPixelIterator<position_iterator<test_deref_adaptor>>);
    static_assert(IsStepIterator<position_iterator<test_deref_adaptor>>);
#endif
} // namespace mgil

// Locators
namespace mgil {
    // A 2D locator over a virtual image
    // Invokes a given function object passing its coordinates upon dereferencing
    // Conforms to IsPixelLocator, IsPixelLocatorHasTransposedType concepts
    template<typename Deref, bool IsTransposed, typename PointType = Point<int>>
        requires IsPixelDereferenceAdaptor<Deref> and IsPoint<PointType>
    class position_locator {};
} // namespace mgil

// Image container
namespace mgil {}

// Image factories, views, and algorithms
namespace mgil {}

#endif // MGIL_H
