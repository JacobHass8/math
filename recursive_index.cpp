#include <vector>
#include <iostream>
#include <chrono>
#include <map>
#include <boost/math/quadrature/gauss.hpp>

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

template <typename Real>
std::pair<std::vector<Real>, std::vector<Real> > getDiff(unsigned level)
{
    if( level < 1)
    {
        throw std::invalid_argument("Level must be greater than or equal to 1");
    }
    
    gauss_points<Real> currentLevel(level);
    if (level == 1 ){
        return std::make_pair(currentLevel.abscissa(), currentLevel.weights());
    }

    gauss_points<Real> previousLevel(level - 1);
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
std::pair<std::vector<Real>, std::vector<Real> > getDiff(unsigned level, gauss_points<Real> currentLevel, gauss_points<Real> previousLevel)
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

template <typename Real>
std::pair<std::vector<std::vector<Real>>, std::vector<Real>> smolyak_nd(std::size_t dimensions, unsigned level)
{
    std::vector<std::vector<Real>> total_points;
    total_points.reserve(dimensions);
    std::unordered_map<std::vector<Real>, Real, VectorHasher<Real>> point_weight_map;
    std::vector<Real> total_weights;

    std::vector<int> slots(dimensions, 1);
    unsigned int current_sum = dimensions;
    int index = 0;
    
    // Precompute the abscissas and weights for each level to avoid redundant calculations
    std::vector<gauss_points<Real>> gauss_points_per_level;
    std::vector<std::vector<Real> > diffAbscissa;
    std::vector<std::vector<Real> > diffWeights;

    for (size_t i = 0; i < level+1; ++i)
    {
        gauss_points<Real> currentLevel(i);
        gauss_points_per_level.push_back(currentLevel);
    }

    for (size_t i = 1; i < level+1; ++i)
    {
        const auto [abscissa, weights] = getDiff(i, gauss_points_per_level[i], gauss_points_per_level[i - 1]);
        diffAbscissa.push_back(abscissa);
        diffWeights.push_back(weights);
    }
    // Finished precomputing abscissas and weights for each level

    // Now compute the Smolyak grid points and weights
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

        // total_points.insert(total_points.end(), points_by_index.begin(), points_by_index.end());
        // total_weights.insert(total_weights.end(), weights_by_index.begin(), weights_by_index.end());

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
        while (current_sum > level+1)
        {
            // Overflow, we're done
            if (index == dimensions - 1)
            {
                return std::make_pair(total_points, total_weights);
            }

            current_sum -= slots[index] - 1;
            slots[index] = 1;

            index++;
            
            slots[index]++;
            current_sum++;
        }

        index = 0;
    }

    return std::make_pair(total_points, total_weights);
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    std::size_t dimensions = 2;
    unsigned level = 3;

    auto t1 = high_resolution_clock::now();
    const auto [points, weights] = smolyak_nd<double>(dimensions, level);
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";    

    std::cout << "Points:" << points.size() << std::endl;
    // for (const auto& point : points)
    // {
    //     for (const auto& coordinate : point)
    //     {
    //         std::cout << coordinate << " ";
    //     }
    //     std::cout << weights[&point - &points[0]] << std::endl;
    // }

    return 0;
}