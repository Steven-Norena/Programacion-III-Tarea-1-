//
// Created by STEVEN on 2/04/2026.
//

#include <vector>
#include <iostream>
#include <random>

class TensorTransform;

class Tensor {
private:
    std::vector<size_t> shape;
    double* data;
    size_t total_size = 1;
public:
    Tensor(const std::vector<size_t>& shape,
        const std::vector<double>& values) : shape(shape) {
        for (const auto dim : shape) {
            this->total_size *= dim;
        }
        this->data = new double[total_size];
        for (int i = 0; i < total_size; i++) {
            this->data[i] = values[i];
        }
    };
    Tensor(const Tensor& other) {
        this->shape = other.shape;
        this->total_size = other.total_size;
        this->data = new double[total_size];
        for (int i = 0; i < total_size; i++) {
            this->data[i] = other.data[i];
        }
    };
    Tensor(Tensor&& other) noexcept {
        this->shape = other.shape;
        this->total_size = other.total_size;
        this->data = new double[total_size];
        for (int i = 0; i < total_size; i++) {
            this->data[i] = other.data[i];
        }
        other.data = nullptr;
    };
    Tensor& operator=(const Tensor& other){};
    Tensor& operator=(Tensor&& other) noexcept;
    ~Tensor() {
        delete[] data;
    };
    [[nodiscard]] Tensor apply(const TensorTransform& transform) const;
    static Tensor zeros(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (const auto dim : shape) {
            n *= dim;
        }
        Tensor tensor(shape, std::vector<double>(n, 0.0));
        return tensor;
    };
    static Tensor ones(const std::vector<size_t>& shape) {
        size_t n = 1;
        for (const auto dim : shape) {
            n *= dim;
        }
        Tensor tensor(shape, std::vector<double>(n, 1.0));
        return tensor;
    };
    static Tensor random(const std::vector<size_t>& shape, double min, double max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        size_t n = 1;
        std::vector<double> values;
        for (const auto dim : shape) {
            n *= dim;
        }
        for (int i = 0; i < n; i++) {
            values.push_back(dis(gen));
        }
        Tensor tensor(shape, values);
        return tensor;
    };
    static Tensor arange(double start, double end);
};

class TensorTransform {
public:
    [[nodiscard]] virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

class ReLU final: public TensorTransform {
public:
    [[nodiscard]] Tensor apply(const Tensor& t) const override;
};