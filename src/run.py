import numpy as np
import click

from nes import NaturalEvolutionStrategy

np.random.seed(0)

def fitness(solution):
    """Objective function for optimization"""
    return sum(solution)

@click.command()
@click.option("--learning_rate", default=0.1, help="Learning rate")
@click.option("--sd", default=1.0, help="Standard deviation")
@click.option("--num_iter", default=1, help="Number of iterations")
@click.option("--num_workers", default=2, help="Number of workers")
@click.option("--log", default="warn", help="Logging level")
def main(learning_rate, sd, num_iter, num_workers, log):
    init_solution = np.random.random(size=3)
    nes = NaturalEvolutionStrategy(fitness, init_solution, learning_rate, sd, log_level=log.upper())
    solution = nes.optimize(num_iter=num_iter, num_workers=num_workers)

if __name__=="__main__":
    main()