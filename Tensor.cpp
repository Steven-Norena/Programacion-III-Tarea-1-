//
// Created by STEVEN on 2/04/2026.
//

#include <vector>
#include <iostream>
#include <random>

class TensorTransform;

class Tensor {
    std::vector<size_t> shape;
    double* data = nullptr;
    size_t total_size = 1;

    friend class ReLU;
    friend class Sigmoid;
    friend Tensor Matmul(const Tensor& a, const Tensor& b);
    friend Tensor dot(const Tensor& A, const Tensor& B);

    explicit Tensor(const std::vector<size_t>& shape)
        : shape(shape) {
        if (shape.size() > 3) {
            throw std::invalid_argument("El tensor no puede tener mas de 3 dimensiones.");
        }
        for (const auto dim: shape) {
            this->total_size *= dim;
        }
        data = new double[total_size];
    };

public:
    Tensor(const std::vector<size_t>& shape, const std::vector<double>& values) : Tensor(shape) {
        if (values.size() != total_size) {
            throw std::invalid_argument("La cantidad de valores no coincide con las dimensiones.");
        }
        std::copy(values.begin(), values.end(), data);
    };

    Tensor(const Tensor& other)
        : shape(other.shape), total_size(other.total_size){
        this->data = new double[total_size];
        std::copy(other.data, other.data + total_size, data);
    };

    Tensor(Tensor&& other) noexcept
        : shape(std::move(other.shape)), data(other.data), total_size(other.total_size) {
        other.data = nullptr;
        other.total_size = 0;
    };

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete[] this->data;

            this->shape = other.shape;
            this->total_size = other.total_size;
            this->data = new double[total_size];
            std::copy(other.data, other.data + total_size, data);
        }
        return *this;
    };

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            delete[] this->data;

            this->shape = std::move(other.shape);
            this->total_size = other.total_size;
            this->data = other.data;

            other.data = nullptr;
            other.total_size = 0;
        }
        return *this;
    };

    ~Tensor() {
        delete[] data;
    };

    static Tensor zeros(const std::vector<size_t>& shape) {
        Tensor t(shape);
        std::fill(t.data, t.data + t.total_size, 0.0);
        return t;
    };

    static Tensor ones(const std::vector<size_t>& shape) {
        Tensor t(shape);
        std::fill(t.data, t.data + t.total_size, 1.0);
        return t;
    };

    static Tensor random(const std::vector<size_t>& shape, double min, double max) {
        Tensor t(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        for (size_t i = 0; i < t.total_size; i++) {
            t.data[i] = dis(gen);
        }
        return t;
    };

    static Tensor arange(double start, double end) {
        if (start >= end) {
            throw std::invalid_argument("El valor de inicio debe ser menor que el final.");
        }
        auto n = static_cast<size_t>(std::ceil(end - start));
        std::vector<double> vals(n);
        for (size_t i = 0; i < n; ++i) {
            vals[i] = start + i;
        }
        return Tensor({n}, vals);
    };

    [[nodiscard]] Tensor view(const std::vector<size_t>& new_shape) const{
        size_t new_total_size = 1;
        for (const auto dim : new_shape) {
            new_total_size *= dim;
        }

        if (new_total_size != this->total_size) {
            throw std::invalid_argument("El nuevo shape no coincide con el numero total de elementos.");
        }

        Tensor t(std::move(*this));
        t.shape = new_shape;
        return t;
    };

    [[nodiscard]] Tensor unsqueeze(const size_t dim) const {
        if (this->shape.size() >= 3) {
            throw std::invalid_argument("No se puede hacer unsqueeze: el tensor ya tiene el maximo de 3 dimensiones.");
        }
        if (dim > shape.size()) {
            throw std::invalid_argument("La dimension especificada esta fuera de rango.");
        }

        Tensor t(*this);
        t.shape.insert(t.shape.begin() + dim, 1);
        return t;
    }

    [[nodiscard]] Tensor apply(const TensorTransform& transform) const;

    double operator()(const size_t i, const size_t j = 0, const size_t k = 0) const {
        size_t index = 0;

        if (shape.size() == 1) {
            index = i;
        }
        else if (shape.size() == 2) {
            index = i * shape[1] + j;
        }
        else if (shape.size() == 3) {
            index = i * (shape[1] * shape[2]) + j * (shape[2]) + k;
        }
        return data[index];
    }

    double& operator()(const size_t i, const size_t j = 0, const size_t k = 0) {
        size_t index = 0;

        if (shape.size() == 1) {
            index = i;
        }
        else if (shape.size() == 2) {
            index = i * shape[1] + j;
        }
        else if (shape.size() == 3) {
            index = i * (shape[1] * shape[2]) + j * (shape[2]) + k;
        }
        return data[index];
    }

    Tensor operator+(const Tensor& other) const {
        if (this->shape == other.shape) {
            Tensor t(this->shape);
            for (size_t i = 0; i < this->total_size; i++) {
                t.data[i] = this->data[i] + other.data[i];
            }
            return t;
        }

        else if (this->shape.size() == 2 && other.shape.size() == 2) {
            size_t M = this->shape[0];
            size_t N = this->shape[1];

            if (other.shape[0] == 1 && other.shape[1] == N) {
                Tensor t(this->shape);
                for (size_t i = 0; i < M; i++) {
                    for (size_t j = 0; j < N; j++) {
                        t.data[i*N + j] = this->data[i*N + j] + other.data[j];
                    }
                }
                return t;
            }
        }
        throw std::invalid_argument("Las dimensiones no son compatibles para la suma o broadcasting.");
    }

    Tensor operator-(const Tensor& other) const {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Las dimensiones deben ser iguales para la resta.");
        }
        Tensor result(this->shape);
        for (size_t i = 0; i < total_size; ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        if (this->shape != other.shape) {
            throw std::invalid_argument("Las dimensiones deben ser iguales para la multiplicacion elemento a elemento.");
        }

        Tensor result(this->shape);
        for (size_t i = 0; i < total_size; ++i) {
            result.data[i] = this->data[i] * other.data[i];
        }
        return result;
    }

    Tensor operator*(double scalar) const {
        Tensor result(this->shape);
        for (size_t i = 0; i < total_size; ++i) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }

    static Tensor concat(const std::vector<Tensor>& tensors, size_t dim) {
        if (tensors.empty()) throw std::invalid_argument("No se pasaron tensores.");
        if (dim != 0) throw std::invalid_argument("Solo se soporta concat en dim=0 para este ejemplo.");

        std::vector<size_t> base_shape = tensors[0].shape;
        size_t total_dim0 = 0;

        for (const auto& t : tensors) {
            if (t.shape.size() != base_shape.size()) throw std::invalid_argument("Dimensiones incompatibles.");
            for (size_t i = 1; i < base_shape.size(); ++i) {
                if (t.shape[i] != base_shape[i]) throw std::invalid_argument("Dimensiones incompatibles.");
            }
            total_dim0 += t.shape[0];
        }

        std::vector<size_t> new_shape = base_shape;
        new_shape[0] = total_dim0;

        Tensor result(new_shape);
        size_t offset = 0;
        for (const auto& t : tensors) {
            std::copy(t.data, t.data + t.total_size, result.data + offset);
            offset += t.total_size;
        }
        return result;
    }
};

class TensorTransform {
public:
    [[nodiscard]] virtual Tensor apply(const Tensor& t) const = 0;
    virtual ~TensorTransform() = default;
};

[[nodiscard]] Tensor Tensor::apply(const TensorTransform& transform) const {
    return transform.apply(*this);
}

class ReLU final: public TensorTransform {
public:
    [[nodiscard]] Tensor apply(const Tensor& t) const override {
        Tensor result(t);

        for (size_t i = 0; i < result.total_size; i++) {
            if (result.data[i] < 0.0) {
                result.data[i] = 0.0;
            }
        }
        return result;
    };
};

class Sigmoid final : public TensorTransform {
public:
    [[nodiscard]] Tensor apply(const Tensor& t) const override {
        Tensor result(t);

        for (size_t i = 0; i < result.total_size; i++) {
            result.data[i] = 1.0 / (1.0 + std::exp(-result.data[i]));
        }
        return result;
    }
};

Tensor Matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 || b.shape.size() != 2) {
        throw std::invalid_argument("Matmul solo soporta tensores de 2 dimensiones.");
    }
    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument("Las dimensiones no son compatibles para multiplicacion de matrices.");
    }

    size_t M = a.shape[0];
    size_t K = a.shape[1];
    size_t N = b.shape[1];

    Tensor result = Tensor::zeros({M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += a(i,k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
};

Tensor dot(const Tensor& A, const Tensor& B) {
    if (A.shape != B.shape) {
        throw std::invalid_argument("Las dimensiones deben ser iguales para el producto punto.");
    }

    double sum = 0.0;
    for (size_t i = 0; i < A.total_size; i++) {
        sum += A.data[i] * B.data[i];
    }
    return Tensor({1}, {sum});
}