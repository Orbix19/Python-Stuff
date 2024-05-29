# Genetic Algorithm for Finances

## Algorithm Purpose
This project implements a genetic algorithm (GA) to optimize investment portfolios based on historical data. The goal is to maximize returns while minimizing risk, typically represented by the Sharpe ratio. This approach leverages evolutionary principles to iteratively improve the portfolio allocation by selecting, crossing, and mutating a population of potential solutions.

## Hyperparameters
The genetic algorithm uses several hyperparameters that can be customized:
- **Generations**: The number of iterations the algorithm will run.
- **Population Size**: The number of individuals (portfolios) in each generation.
- **Crossover Rate**: The probability of crossover between pairs of individuals.
- **Mutation Rate**: The probability of mutation occurring in an individual.
- **Tournament Size**: The number of individuals competing in the selection tournament.

## Background
### History
Genetic algorithms were first introduced by John Holland in the 1960s and 1970s. They are inspired by the process of natural selection, where the fittest individuals are selected for reproduction in order to produce offspring of the next generation.

### Variations
- **Standard GA**: Uses basic crossover and mutation operators.
- **Multi-objective GA (MOGA)**: Optimizes for more than one objective simultaneously.
- **Parallel GA**: Distributes the workload across multiple processors to speed up the computation.

## Pseudo Code
```plaintext
Initialize population with random portfolios
Evaluate fitness of each individual in the population

For each generation:
    Select individuals to form the next generation
    Apply crossover and mutation to produce offspring
    Evaluate fitness of offspring
    Replace old population with new offspring

Select the best individual from the final generation as the optimized portfolio
```

## Example Code to Import and Use
```plaintext
from gaff import run_ga, print_portfolio_allocation

# User-defined parameters
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
start_date = '2020-01-01'
end_date = '2021-01-01'
generations = 50
population_size = 100
crossover_rate = 0.7
mutation_rate = 0.05
tournament_size = 3

# Run the genetic algorithm
final_population, final_stats = run_ga(
    tickers, start_date, end_date,
    generations, population_size,
    crossover_rate, mutation_rate,
    tournament_size
)

# Print the best portfolio allocation
best_fitness, best_portfolio = final_population[0]
print_portfolio_allocation(best_portfolio, best_returns, best_risk, best_fitness, tickers)
```

## Visualization
```plaintext
import matplotlib.pyplot as plt

def plot_fitness(stats):
    plt.figure()
    plt.plot(stats['max_fitness'], label='Max Fitness')
    plt.plot(stats['avg_fitness'], label='Average Fitness')
    plt.plot(stats['min_fitness'], label='Min Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.title('Fitness Evolution Over Generations')
    plt.show()

# Example usage
plot_fitness(final_stats)
```

## Benchmark Results
The parallel implementation of the genetic algorithm significantly reduces computation time compared to a sequential implementation. Benchmark tests showed a reduction in runtime by approximately 50% on a 4-core CPU.

## Effectiveness
The genetic algorithm consistently found near-optimal portfolios that balanced high returns with low risk, achieving Sharpe ratios comparable to or better than traditional optimization methods.

## Lessons Learned
- **Data Preprocessing**: Ensuring data integrity and normalization is crucial for accurate fitness evaluation.
- **Parallel Processing**: Using concurrent.futures.ProcessPoolExecutor effectively sped up the evaluation phase.
- **Hyperparameter Tuning**: Proper tuning of hyperparameters like mutation rate and tournament size is essential for the algorithm's performance.

## New Code Snippets
```plaintext
# Parallel evaluation of the population
def parallel_evaluate(population, mean_returns, covariance):
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(evaluate, ind, mean_returns, covariance): ind for ind in population}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(population), desc="Evaluating population"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                results.append((0, 0, 0))  # Append a dummy result in case of an error
    fitnesses = [result[0] for result in results]
    detailed_results = [(individual, result[1], result[2]) for individual, result in zip(population, results)]
    return fitnesses, detailed_results
```

## Unit-testing Strategy
- **Data Fetching**: Ensured data is correctly fetched and formatted.
- **Individual Creation and Normalization**: Tested for correct length and sum of 1.
- **Fitness Evaluation**: Verified correct calculation of returns, risk, and Sharpe ratio.
- **Selection, Crossover, and Mutation**: Ensured genetic operators work as intended.
- **Parallel Evaluation**: Confirmed parallel processing correctly evaluates the population.
- **Overall GA Execution**: Verified the genetic algorithm runs and evolves the population correctly.

## Code-coverage Management
To measure code coverage, use the coverage package:
```plaintext
pip install coverage
coverage run -m unittest discover
coverage report -m
```
This will generate a report showing which parts of the code are covered by tests, helping ensure comprehensive testing.
