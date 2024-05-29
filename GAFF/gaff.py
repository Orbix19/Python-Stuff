import numpy as np
import pandas as pd
import yfinance as yf
import random
import logging
import concurrent.futures
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Function to prompt user input for stock tickers, date range, and hyperparameters
def get_user_input():
    # Prompt for stock tickers
    tickers = input("Enter the stock tickers separated by commas (e.g., AAPL, GOOGL, MSFT): ").split(',')
    tickers = [ticker.strip() for ticker in tickers]

    # Prompt for date range
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Prompt for hyperparameters
    generations = int(input("Enter the number of generations: "))
    population_size = int(input("Enter the population size: "))
    crossover_rate = float(input("Enter the crossover rate (e.g., 0.7): "))
    mutation_rate = float(input("Enter the mutation rate (e.g., 0.05): "))
    tournament_size = int(input("Enter the tournament size: "))

    return tickers, start_date, end_date, generations, population_size, crossover_rate, mutation_rate, tournament_size

# Create a directory for the figures if it doesn't exist
figures_directory = "gaff_figures"
if not os.path.exists(figures_directory):
    os.makedirs(figures_directory)

# Fetch data from Yahoo Finance
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)
    return data['Adj Close']

def initialize_data(tickers, start_date, end_date):
    data = fetch_data(tickers, start_date, end_date)
    mean_returns = data.pct_change().fillna(0).mean().values
    covariance = data.pct_change().fillna(0).cov().values
    return data, mean_returns, covariance

# Individual representation: list of investment percentages
def create_individual(num_tickers):
    individual = [random.random() for _ in range(num_tickers)]
    return normalize(individual)

# Normalize the individual to sum to 1 (100% investment)
def normalize(individual):
    total = sum(individual)
    return [x / total for x in individual]

# Fitness evaluation: Calculate returns and risk
def evaluate(individual, mean_returns, covariance):
    returns = np.dot(mean_returns, individual)
    risk = np.dot(individual, np.dot(covariance, individual))
    sharpe_ratio = returns / risk  # Sharpe ratio as fitness
    return sharpe_ratio, returns, risk

# Selection mechanism: Tournament selection
def select(population, fitnesses, tournament_size):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.append(max(participants, key=lambda x: x[1])[0])
    return selected

# Crossover: Single point crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return normalize(child1), normalize(child2)

# Mutation: Perturb a gene
def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        index = random.randint(0, len(individual) - 1)
        individual[index] += np.random.normal(0, 0.1)
        return normalize(individual)
    return individual

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

# Print the portfolio allocation with respective stocks
def print_portfolio_allocation(best_portfolio, best_returns, best_risk, best_fitness, tickers):
    print(f"\nBest Portfolio Allocation:")
    for ticker, allocation in zip(tickers, best_portfolio):
        print(f"{ticker}: {allocation:.4f}")
    print(f"\nBest Fitness (Sharpe Ratio): {best_fitness:.4f}")
    print(f"Expected Returns: {best_returns:.4f}")
    print(f"Risk (Volatility): {best_risk:.4f}")

def run_ga(tickers, start_date, end_date, generations, population_size, crossover_rate, mutation_rate, tournament_size):
    data, mean_returns, covariance = initialize_data(tickers, start_date, end_date)
    population = [create_individual(len(tickers)) for _ in range(population_size)]
    stats = {"max_fitness": [], "avg_fitness": [], "min_fitness": []}
    best_individual = None
    best_details = None
    
    for generation in tqdm(range(generations), desc="Generations"):
        fitnesses, detailed_results = parallel_evaluate(population, mean_returns, covariance)
        
        # Record statistics for plotting
        max_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)
        stats['max_fitness'].append(max_fitness)
        stats['avg_fitness'].append(avg_fitness)
        stats['min_fitness'].append(min_fitness)
        
        # Identify the best individual and its details
        if best_individual is None or max_fitness > best_individual[0]:
            best_index = fitnesses.index(max_fitness)
            best_individual = (max_fitness, population[best_index])
            best_details = detailed_results[best_index]
        
        # Selection and reproduction process
        selected = select(population, fitnesses, tournament_size)
        offspring = []
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):  # Ensure there's a pair for crossover
                if random.random() < crossover_rate:
                    child1, child2 = crossover(selected[i], selected[i+1])
                else:
                    child1, child2 = selected[i], selected[i+1]
                offspring.append(mutate(child1, mutation_rate))
                offspring.append(mutate(child2, mutation_rate))
        population = offspring
        
        print(f'Generation {generation}, Best Fitness: {max_fitness}')
        
        # Plotting and saving the figure for each generation
        plot_stats(stats, generation, figures_directory)
    
    # Print the best individual details
    best_fitness, best_portfolio = best_individual
    _, best_returns, best_risk = best_details
    print_portfolio_allocation(best_portfolio, best_returns, best_risk, best_fitness, tickers)
    
    return population, stats

def plot_stats(stats, generation, directory):
    plt.figure()
    plt.plot(stats['max_fitness'], label='Max Fitness')
    plt.plot(stats['avg_fitness'], label='Average Fitness')
    plt.plot(stats['min_fitness'], label='Min Fitness')
    plt.title(f'Fitness over Generations - Gen {generation}')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{directory}/gen_{generation}.png')
    plt.close()

# Prompt user for input
tickers, start_date, end_date, generations, population_size, crossover_rate, mutation_rate, tournament_size = get_user_input()

# Run the genetic algorithm with user-specified hyper-parameters
final_population, final_stats = run_ga(tickers, start_date, end_date, generations, population_size, crossover_rate, mutation_rate, tournament_size)