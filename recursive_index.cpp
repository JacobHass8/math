#include <vector>
#include <iostream>
#include <chrono>
#include <numeric>
#include <map>
#include <functional>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/special_functions/binomial.hpp>

template <class Real>
class trap_points
{
   unsigned q;
   std::vector<Real> weight_vals;
   std::vector<Real> abscissa_vals;

public:
    std::vector<Real> abscissa()
    {
        return abscissa_vals;
    }
    std::vector<Real> weights()
    {
        return weight_vals;
    }

    trap_points(unsigned q_) : q(q_)
    {
        if (q == 0 || q == 1)
        {
            throw std::invalid_argument("Number of points must be greater than or equal to 2");
        }
        for (unsigned l=1; l < q; l++)
        {
            if (l == 1){
                abscissa_vals.push_back(0.0);
                weight_vals.push_back(2.0);
                continue;
            }
            unsigned nl;

            nl = pow(2, l-1) + 1;

            for (unsigned j=1; j <= nl; j++)
            {
                Real h = pow(2.0, 2.0 - Real(l));
                weight_vals.push_back(h);
                abscissa_vals.push_back((Real(j) - 1.0) * h - 1.0);
                if (j == 1 || j == nl)
                {
                    weight_vals.back() /= 2.0;
                }
            }
        }
        Real total_weight = std::accumulate(weight_vals.begin(), weight_vals.end(), 0.0);
        std::transform(weight_vals.begin(), weight_vals.end(), weight_vals.begin(), [total_weight](Real w) { return 2 * w / total_weight; });
    };

};

// Custom hasher functor for std::vector
template <class Real>
struct VectorHasher {
    int operator()(const std::vector<Real>& V) const {
        int hash = V.size();
        for (auto &i : V) {
            // A common shift-and-XOR combination algorithm
            hash ^= std::hash<Real>()(i) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

template <class Real>
class gauss_points
{
   std::vector<Real> calculate_weights()
   {
      std::vector<Real> result(abscissa().size(), 0);
      for (unsigned i = 0; i < abscissa().size(); ++i)
      {
         Real x = abscissa()[i];
         Real p = boost::math::legendre_p_prime(N, x);
         result[i] = 2 / ((1 - x * x) * p * p);
      }
      return result;
   }
   unsigned N;

public:
   std::vector<Real> abscissa()
   {
      std::vector<Real> data = boost::math::legendre_p_zeros<Real>(N);
      return data;
   }
   std::vector<Real> weights()
   {
      std::vector<Real> data = calculate_weights();
      return data;
   }
   gauss_points(unsigned N_) : N(N_) {};

};

template <typename Real, class Quad>
std::pair<std::vector<Real>, std::vector<Real> > getDiff(unsigned level)
{
    if( level < 1)
    {
        throw std::invalid_argument("Level must be greater than or equal to 1");
    }

    Quad currentLevel(level);
    if (level == 1 ){
        return std::make_pair(currentLevel.abscissa(), currentLevel.weights());
    }

    Quad previousLevel(level - 1);
    std::vector<Real> currentAbscissa = currentLevel.abscissa();
    std::vector<Real> previousAbscissa = previousLevel.abscissa();

    std::vector<Real> currentWeights = currentLevel.weights();
    std::vector<Real> previousWeights = previousLevel.weights();

    currentAbscissa.insert(currentAbscissa.end(), previousAbscissa.begin(), previousAbscissa.end());
    std::transform(previousWeights.cbegin(), previousWeights.cend(), previousWeights.begin(), std::negate<Real>());

    currentWeights.insert(currentWeights.end(), previousWeights.begin(), previousWeights.end());
    return std::make_pair(currentAbscissa, currentWeights);
}

template <typename Real, class Quad>
std::pair<std::vector<Real>, std::vector<Real> > getDiff(unsigned level, Quad currentLevel, Quad previousLevel)
{
    if( level < 1)
    {
        throw std::invalid_argument("Level must be greater than or equal to 1");
    }

    if (level == 1 ){
        return std::make_pair(currentLevel.abscissa(), currentLevel.weights());
    }

    std::vector<Real> currentAbscissa = currentLevel.abscissa();
    std::vector<Real> previousAbscissa = previousLevel.abscissa();

    std::vector<Real> currentWeights = currentLevel.weights();
    std::vector<Real> previousWeights = previousLevel.weights();

    currentAbscissa.insert(currentAbscissa.end(), previousAbscissa.begin(), previousAbscissa.end());
    std::transform(previousWeights.cbegin(), previousWeights.cend(), previousWeights.begin(), std::negate<Real>());

    currentWeights.insert(currentWeights.end(), previousWeights.begin(), previousWeights.end());
    return std::make_pair(currentAbscissa, currentWeights);
}

template <typename Real>
std::vector<std::vector<Real> > cartesian_product(const std::vector<Real>& a, const std::vector<Real>& b)
{
    std::vector<std::vector<Real>> result(a.size() * b.size(), std::vector<Real>(2));

    for (size_t i = 0; i < a.size(); ++i)
    {
        const auto& x = a[i];
        for (size_t j = 0; j < b.size(); ++j)
        {
            const auto& y = b[j];
            result[i * b.size() + j] = {x, y};
        }
    }
    return result;
}

template <typename Real>
std::vector<std::vector<Real> > cartesian_product(const std::vector<std::vector<Real>>& a, const std::vector<Real>& b)
{
    std::vector<std::vector<Real>> result(a.size() * b.size(), std::vector<Real>(a[0].size() + 1));

    for (size_t i = 0; i < a.size(); ++i)
    {
        const auto& vec_a = a[i];
        for (size_t j = 0; j < b.size(); ++j)
        {
            Real b_val = b[j];
            std::vector<Real> combined(vec_a);
            combined.push_back(b_val);
            result[i * b.size() + j] = combined;
        }
    }
    return result;
}

template <typename Real>
std::vector<Real> cartesian_product_weights(const std::vector<Real>& a, const std::vector<Real>& b)
{
    std::vector<Real> result;
    result.reserve(a.size() * b.size());

    for (const auto& x : a)
    {
        for (const auto& y : b)
        {
            result.push_back(x * y);
        }
    }
    return result;
}

template <typename Real, class Quad, typename GrowthMap>
std::map<std::vector<Real>, Real> smolyak_nd(std::size_t dimensions, unsigned level, GrowthMap growth_map)
{
    std::map<std::vector<Real>, Real> point_weight_map;

    std::vector<int> slots(dimensions, 1);
    const unsigned int max_sum = level + dimensions - 1;
    unsigned int current_sum = dimensions;
    int index = 0;

    // Precompute the abscissas and weights for each level to avoid redundant calculations
    std::vector<Quad> gauss_points_per_level;
    std::vector<std::vector<Real> > diffAbscissa;
    std::vector<std::vector<Real> > diffWeights;

    for (size_t i = 0; i < level + 1; ++i)
    {
        unsigned num_points = growth_map(i); // Map level i to N points
        Quad currentLevel(num_points);
        gauss_points_per_level.push_back(currentLevel);
    }

    for (size_t i = 1; i < level+1; ++i)
    {
        const auto [abscissa, weights] = getDiff<Real, Quad>(i, gauss_points_per_level[i], gauss_points_per_level[i - 1]);
        diffAbscissa.push_back(abscissa);
        diffWeights.push_back(weights);
    }
    // Finished precomputing abscissas and weights for each level

    // Compute the Smolyak grid points and weights
    while (true)
    {
        std::vector<Real> abscissa_0 = diffAbscissa[slots[0]-1];
        std::vector<Real> weights_0 = diffWeights[slots[0]-1];

        std::vector<Real> abscissa_1 = diffAbscissa[slots[1]-1];
        std::vector<Real> weights_1 = diffWeights[slots[1]-1];

        std::vector<std::vector<Real> > points_by_index = cartesian_product(abscissa_0, abscissa_1);
        std::vector<Real> weights_by_index = cartesian_product_weights(weights_0, weights_1);

        for (int i = 2; i < dimensions; ++i)
        {
            std::vector<Real> abscissa_j = diffAbscissa[slots[i]-1];
            std::vector<Real> weights_j = diffWeights[slots[i]-1];
            points_by_index = cartesian_product(points_by_index, abscissa_j);
            weights_by_index = cartesian_product_weights(weights_by_index, weights_j);
        }

        for (size_t i = 0; i < points_by_index.size(); ++i)
        {
            std::vector<Real>& point = points_by_index[i];
            Real& weight = weights_by_index[i];

            auto [iterator, success] = point_weight_map.insert({point, weight});
            if (!success)
            {
                iterator->second += weight;
            }
        }

        // Now increment slots to handle indexing for the next iteration
        slots[0]++;
        current_sum++;

        // Carry
        while (current_sum > max_sum)
        {
            // Overflow, we're done
            if (index == dimensions - 1)
            {
                return point_weight_map;
            }

            current_sum -= slots[index] - 1;
            slots[index] = 1;

            index++;

            slots[index]++;
            current_sum++;
        }

        index = 0;
    }

    return point_weight_map;
}

std::vector<std::vector<unsigned> > getIndices(size_t dimensions, unsigned level)
{
    std::vector<std::vector<unsigned> > indices;
    std::vector<unsigned> slots(dimensions, 1);
    unsigned int current_sum = dimensions;
    const unsigned int max_sum = level;
    const unsigned min_sum = level - dimensions + 1;
    unsigned index = 0;

    while (true)
    {
        if ((current_sum >= min_sum) && (current_sum <= max_sum) && (current_sum >= dimensions))
        {
            indices.push_back(slots);
        }

        // Now increment slots to handle indexing for the next iteration
        slots[0]++;
        current_sum++;

        // Carry
        while (current_sum > max_sum)
        {
            // Overflow, we're done
            if (index == dimensions - 1)
            {
                return indices;
            }

            current_sum -= slots[index] - 1;
            slots[index] = 1;

            index++;

            slots[index]++;
            current_sum++;
        }

        index = 0;
    }
    return indices;
}

template <typename Real>
Real combination_coefficient(size_t dimensions, unsigned level, const std::vector<unsigned>& index)
{
    unsigned sum = 0;
    for (unsigned v : index)
    {
        sum += v;
    }

    unsigned k = level - sum;
    if (k > dimensions - 1)
    {
        return 0;
    }

    Real sign = (k % 2 == 0) ? static_cast<Real>(1) : static_cast<Real>(-1);
    Real coeff = static_cast<Real>(boost::math::binomial_coefficient<Real>(dimensions - 1, k));
    return sign * coeff;
}

template <typename Real, class Quad, typename GrowthMap>
std::map<std::vector<Real>, Real> combination_grid(size_t dimensions, unsigned level, GrowthMap growth_map)
{
    std::vector<std::vector<unsigned> > indices = getIndices(dimensions, level);
    std::map<std::vector<Real>, Real> point_weight_map;

    // Precompute the abscissas and weights for each level to avoid redundant calculations
    std::vector<Quad> points_per_level;

    for (size_t i = 0; i < level + 1; ++i)
    {
        unsigned num_points = growth_map(i); // Map level i to N points
        Quad currentLevel(num_points);
        points_per_level.push_back(currentLevel);
    }

    for (size_t i=0; i < indices.size(); ++i)
    {
        std::vector<unsigned> current_indices = indices[i];
        Real coeff = combination_coefficient<Real>(dimensions, level, current_indices);

        // Calculate the first level's abscissas and weights to initialize the cartesian product
        std::vector<Real> abscissa_0 = points_per_level[current_indices[0]].abscissa();
        std::vector<Real> weights_0 = points_per_level[current_indices[0]].weights();

        std::vector<Real> abscissa_1 = points_per_level[current_indices[1]].abscissa();
        std::vector<Real> weights_1 = points_per_level[current_indices[1]].weights();

        std::vector<std::vector<Real> > abscissas = cartesian_product(abscissa_0, abscissa_1);
        std::vector<Real> weights = cartesian_product_weights(weights_0, weights_1);

        for (size_t j=2; j < current_indices.size(); j++)
        {
            std::vector<Real> abscissa_j = points_per_level[current_indices[j]].abscissa();
            std::vector<Real> weights_j = points_per_level[current_indices[j]].weights();

            abscissas = cartesian_product(abscissas, abscissa_j);
            weights = cartesian_product_weights(weights, weights_j);
        }

        for (size_t j = 0; j < weights.size(); ++j)
        {
            weights[j] *= coeff;
        }

        for (size_t j = 0; j < abscissas.size(); ++j)
        {
            std::vector<Real>& point = abscissas[j];
            Real& weight = weights[j];

            auto [iterator, success] = point_weight_map.insert({point, weight});
            if (!success)
            {
                iterator->second += weight;
            }
        }
    }
    return point_weight_map;
}

void print_indices(size_t dimensions, unsigned level)
{
    std::vector<std::vector<unsigned>> indices = getIndices(dimensions, level);
    std::cout << "Indices for dimensions = " << dimensions << ", level = " << level << ":\n";
    for (const auto& index : indices)
    {
        for (const auto& val : index)
        {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
}

void get_and_print_map(size_t dimensions, unsigned level)
{
    std::map<std::vector<double>, double> result = combination_grid<double, trap_points<double>, std::function<unsigned(unsigned)>>(
        dimensions, level, [](unsigned l) { return l + 2; });

    std::cout << "Combination grid points and weights:\n";
    for (const auto& [point, weight] : result)
    {
        for (size_t i = 0; i < point.size(); ++i)
        {
            std::cout << point[i];
            if (i < point.size() - 1)
                std::cout << " ";
        }
        std::cout << " " << weight << "\n";
    }
}

template <typename Real>
Real integrate_function(const std::function<Real(const std::vector<Real>&)>& func, unsigned level, size_t dimensions)
{
    auto growth_map = [](unsigned l) { return l + 2; }; // Example growth map

    auto point_weight_map = combination_grid<Real, trap_points<Real>, std::function<unsigned(unsigned)>>(
        dimensions, level, growth_map);
    std::cout << "Number of unique points: " << point_weight_map.size() << std::endl;
    Real integral = 0.0;
    for (const auto& [point, weight] : point_weight_map)
    {
        integral += func(point) * weight;
    }

    return integral;
}

int main()
{
    std::size_t dimensions = 2;
    unsigned level = 12;

    double integral = integrate_function<double>([](const std::vector<double>& x) {
        return x[0] * x[0] * x[1] * x[1]; // Example function to integrate
    }, level, dimensions);
    std::cout << integral << std::endl;

    return 0;
}
