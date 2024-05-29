import unittest
from gaff import *

class TestGAModule(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'FB']
        self.start_date = '2020-01-01'
        self.end_date = '2021-01-01'
        self.data, self.mean_returns, self.covariance = initialize_data(self.tickers, self.start_date, self.end_date)
    
    def test_fetch_data(self):
        data = fetch_data(self.tickers, self.start_date, self.end_date)
        self.assertIsNotNone(data)
        self.assertEqual(len(data.columns), len(self.tickers))
    
    def test_initialize_data(self):
        data, mean_returns, covariance = initialize_data(self.tickers, self.start_date, self.end_date)
        self.assertIsNotNone(data)
        self.assertEqual(len(mean_returns), len(self.tickers))
        self.assertEqual(covariance.shape[0], len(self.tickers))
        self.assertEqual(covariance.shape[1], len(self.tickers))
    
    def test_create_individual(self):
        individual = create_individual(len(self.tickers))
        self.assertEqual(len(individual), len(self.tickers))
        self.assertAlmostEqual(sum(individual), 1.0, places=5)
    
    def test_normalize(self):
        individual = [random.random() for _ in range(len(self.tickers))]
        normalized_individual = normalize(individual)
        self.assertAlmostEqual(sum(normalized_individual), 1.0, places=5)
    
    def test_evaluate(self):
        individual = create_individual(len(self.tickers))
        fitness, returns, risk = evaluate(individual, self.mean_returns, self.covariance)
        self.assertIsNotNone(fitness)
        self.assertIsNotNone(returns)
        self.assertIsNotNone(risk)
    
    def test_select(self):
        population = [create_individual(len(self.tickers)) for _ in range(10)]
        fitnesses = [evaluate(ind, self.mean_returns, self.covariance)[0] for ind in population]
        selected = select(population, fitnesses, tournament_size=3)
        self.assertEqual(len(selected), len(population))
    
    def test_crossover(self):
        parent1 = create_individual(len(self.tickers))
        parent2 = create_individual(len(self.tickers))
        child1, child2 = crossover(parent1, parent2)
        self.assertEqual(len(child1), len(self.tickers))
        self.assertEqual(len(child2), len(self.tickers))
        self.assertAlmostEqual(sum(child1), 1.0, places=5)
        self.assertAlmostEqual(sum(child2), 1.0, places=5)
    
    def test_mutate(self):
        individual = create_individual(len(self.tickers))
        mutated_individual = mutate(individual, mutation_rate=0.1)
        self.assertEqual(len(mutated_individual), len(self.tickers))
        self.assertAlmostEqual(sum(mutated_individual), 1.0, places=5)
    
    def test_parallel_evaluate(self):
        population = [create_individual(len(self.tickers)) for _ in range(10)]
        fitnesses, detailed_results = parallel_evaluate(population, self.mean_returns, self.covariance)
        self.assertEqual(len(fitnesses), len(population))
        self.assertEqual(len(detailed_results), len(population))
    
    def test_run_ga(self):
        population, stats = run_ga(self.tickers, self.start_date, self.end_date, generations=5, population_size=10)
        self.assertIsNotNone(population)
        self.assertEqual(len(population), 10)
        self.assertEqual(len(stats['max_fitness']), 5)
        self.assertEqual(len(stats['avg_fitness']), 5)
        self.assertEqual(len(stats['min_fitness']), 5)

if __name__ == '__main__':
    unittest.main()
