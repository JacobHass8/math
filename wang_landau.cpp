#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <iomanip>

template <typename Real>
std::vector<Real> expandVector(const std::vector<Real>& vec, Real& dy)
{
    if (vec.empty())
    {
        throw std::invalid_argument("Cannot expand an empty vector.");
    }

    const std::size_t oldSize = vec.size();
    std::vector<Real> expandedVector(2 * oldSize);
    for (std::size_t i = 0; i < expandedVector.size(); ++i)
    {
        const Real position = (static_cast<Real>(i) + Real(0.5)) / Real(2) - Real(0.5);
        if (oldSize == 1)
        {
            expandedVector[i] = vec[0];
        }
        else if (position <= Real(0))
        {
            expandedVector[i] = vec[0] + position * (vec[1] - vec[0]);
        }
        else if (position >= static_cast<Real>(oldSize - 1))
        {
            const Real offset = position - static_cast<Real>(oldSize - 1);
            expandedVector[i] = vec[oldSize - 1] + offset * (vec[oldSize - 1] - vec[oldSize - 2]);
        }
        else
        {
            const std::size_t left = static_cast<std::size_t>(position);
            const Real fraction = position - static_cast<Real>(left);
            expandedVector[i] = vec[left] + fraction * (vec[left + 1] - vec[left]);
        }
    }

    dy /= 2.0; // Update dy to reflect the new bin width
    return expandedVector;
}

template <typename Real>
Real computeIntegral()
{
    // Now need to normalize the density of states.
    std::vector<Real> g(L);
    Real maxS = *std::max_element(S.begin(), S.end());
    std::transform(S.begin(), S.end(), g.begin(), [maxS](Real s) { return std::exp(s - maxS); });
    Real sum = std::accumulate(g.begin(), g.end(), Real(0));

    std::vector<Real> normalized_g(L);
    std::transform(g.begin(), g.end(), normalized_g.begin(), [sum, a, b](Real val) { return val * (b - a) / sum; });

    Real integral = 0.0;
    for (std::size_t i = 0; i < L; ++i)
    {
        integral += normalized_g[i] * (ymin + (i + 0.5) * dy);
    }
    return integral;
}

template <typename Func, typename Real>
static auto wang_landau_integration(Func f, Real a, Real b, Real ymin, Real ymax, Real dy, Real intEstimate,bool verbose=false, bool longExpand=false)->decltype(std::declval<Func>()(std::declval<Real>()))
{
    using K = decltype(f(Real(0)));
    static_assert(!std::is_integral<K>::value,
                "The return type cannot be integral, it must be either a real or complex floating point type.");

    // Initialize the bin centers. Then can check bin_centers +/- dy if the current
    // state is within the bin.
    if (!(dy > 0) || !(ymax > ymin))
    {
        throw std::invalid_argument("dy must be positive and ymax must be greater than ymin.");
    }

    Real binCount = (ymax - ymin) / dy;
    Real roundedBinCount = std::round(binCount);
    Real tolerance = 8 * std::numeric_limits<Real>::epsilon() * std::max(Real(1), std::abs(binCount));
    if (!std::isfinite(binCount) || roundedBinCount < 1 ||
        std::abs(binCount - roundedBinCount) > tolerance ||
        roundedBinCount > std::numeric_limits<unsigned int>::max())
    {
        throw std::invalid_argument("(ymax - ymin) must be a positive integer multiple of dy.");
    }
    unsigned int L = static_cast<unsigned int>(roundedBinCount);
    std::vector<unsigned int> binVisits(L, 0);

    std::vector<Real> S(L, 0.0); // Initialize S(y) = ln(g(y)) to 0
    Real F = 1.0; // Modification factor
    Real Ffinal = 5e-4; // Final modification factor
    Real t = 0.0; // Monte Carlo time
    std::size_t trialsSinceCheck = 0; // Counter for MC trials
    bool longTimeRegime = false; // Flag for going to the 1/t regime
    Real prevLongTimecheck;

    // Set up random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Real> dist(a, b);
    std::uniform_real_distribution<Real> unif(0.0, 1.0);

    Real xi = dist(gen);
    K yi = f(xi);
    std::size_t currentBin = static_cast<unsigned int>((yi - ymin) / dy);
    K currentY = yi;
    binVisits[currentBin] += 1;
    S[currentBin] += F;

    if (verbose)
    {
        std::cout << "Starting Wang-Landau integration with " << L << " bins." << std::endl;
    }
    while (F > Ffinal)
    {
        Real xf = dist(gen);
        Real yf = f(xf);
        std::size_t proposedBin = static_cast<unsigned int>((yf - ymin) / dy);

        if (unif(gen) < std::min(1.0, std::exp(S[currentBin] - S[proposedBin])))
        {
            currentBin = proposedBin;
            currentY = yf;
        }
        S[currentBin] += F;
        binVisits[currentBin] += 1;

        ++trialsSinceCheck;
        t += 1.0 / L;
        if (!longTimeRegime && (trialsSinceCheck % 1000 == 0))
        {
            if (verbose)
            {
                std::cout << "Current F: " << F << ", time: " << trialsSinceCheck << std::endl;
            }
            if(*std::min_element(binVisits.begin(), binVisits.end()) > 0)
            {
                F /= 2.0;
                std::fill(binVisits.begin(), binVisits.end(), 0);
                if (F <= 1.0 / t)
                {
                    longTimeRegime = true;
                    F = 1.0 / t;
                    prevLongTimecheck = t;
                }
            }
        }
        else if (longTimeRegime)
        {
            F = 1.0 / t;
            if (trialsSinceCheck % 1000 == 0)
            {
                if (verbose)
                {
                    std::cout << trialsSinceCheck << "," << std::endl;
                }
            }
            if (t > 2 * prevLongTimecheck)
            {
                prevLongTimecheck = t;
                S = expandVector(S, dy);
                L = static_cast<unsigned int>(S.size());
                binVisits.assign(L, 0);
                std::cout << "New step size dy: " << dy << std::endl;
                currentBin = std::min(
                    static_cast<std::size_t>(L - 1),
                    static_cast<std::size_t>((currentY - ymin) / dy));
            }
        }
    }

    Real integral = computeIntegral<Real>();
    return integral;
}

int main() {

    auto f = [](double x) { return x * x; }; // Example function
    double a = 0.0; // Lower limit
    double b = 1.0; // Upper limit
    double ymin = 0.0;
    double ymax = 1.0;
    double dy = 0.00078125; // Bin width
    double intEstimate = 0.3333333333333333; // Example integral estimate
    double integral = wang_landau_integration(f, a, b, ymin, ymax, dy, intEstimate, false, false);

    std::cout << std::setprecision(32) << "Estimated integral: " << integral << std::endl;

    return 0;
}