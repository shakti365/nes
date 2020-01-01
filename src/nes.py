import numpy as np
import logging

np.random.seed(0)

class NaturalEvolutionStrategy:

    def __init__(self, fitness, init_solution, learning_rate, sd, log_level=logging.WARN):

        logging.basicConfig(level=log_level)

        # Initialize the fitness function
        self.fitness = fitness

        # Initialize learning rate
        self.learning_rate = learning_rate

        # Initialize standard deviation of the noise
        self.sd = sd

        # Initialize solution (policy parameter)
        self.solution = init_solution


    def optimize(self, num_iter, num_workers):

        logging.debug(f"inital solution: {self.solution}")

        # Run optimization for num_iter
        for i in range(num_iter):

            # Sample set of noise from a normal distribution
            noises = np.random.normal(loc=0.0, scale=1.0, size=num_workers)
            logging.debug(f"noise: {noises}")

            # Create new solutions with the sampled noise and SD
            new_solutions = np.array([self.solution + self.sd * noise for noise in noises])
            logging.debug(f"solutions: {new_solutions}")

            # Evaluate function over the set of solutions
            fitness_evals = [self.fitness(sol) for sol in new_solutions]
            logging.debug(f"fitness: {fitness_evals}")
            
            # Estimate gradient for each worker
            grad = np.sum(fitness_evals*noises) / (num_workers * self.sd)
            logging.debug(f"gradient: {grad}")

            # Update the solution using gradient ascent
            self.solution += self.learning_rate * grad

            logging.info(f"Iteration: {i} Max Fitness: {max(fitness_evals)} Solution: {self.solution}")

        print(f"Max Fitness: {max(fitness_evals)} Solution: {self.solution}")
        return self.solution