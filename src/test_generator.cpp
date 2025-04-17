#include <iostream>
#include <vector>
#include <random>
#include <unordered_set>
#include <cstdlib>
#include <algorithm>

// Function to generate data with controlled randomness
void generate_data(std::vector<int> &buffer, size_t tuples_count,
                   int offset = 0, float randomness_factor = 1.0f) {
    buffer.resize(tuples_count);

    // Clamp randomness_factor between 0 and 1
    if (randomness_factor < 0.0f) randomness_factor = 0.0f;
    if (randomness_factor > 1.0f) randomness_factor = 1.0f;

    // Determine number of unique values
    size_t unique_values = std::max((size_t)(tuples_count * randomness_factor), (size_t)1);

    // Set up RNG
    std::default_random_engine rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, unique_values - 1);

    for (size_t i = 0; i < tuples_count; ++i) {
        buffer[i] = offset + dist(rng);
        printf("Generated value: %d\n", buffer[i]);
    }
}


// Utility function to count number of unique values
size_t count_unique(const std::vector<int> &buffer) {
    std::unordered_set<int> seen(buffer.begin(), buffer.end());
    return seen.size();
}

int main() {
    size_t tuples_count = 32;
    float randomness_values[] = {0.0f, 0.1f, 0.5f, 0.9f, 1.0f};

    for (float rf : randomness_values) {
        std::vector<int> buffer;
        generate_data(buffer, tuples_count, 0, rf);
        size_t unique_count = count_unique(buffer);

        std::cout << "Randomness Factor: " << rf
                  << " | Unique Values: " << unique_count
                  << " / " << tuples_count
                  << " (" << (100.0 * unique_count / tuples_count) << "% unique)"
                  << std::endl;
    }

    return 0;
}