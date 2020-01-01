# Natural Evolution Strategies

Implementation from the paper (Evolution Strategies as a
Scalable Alternative to Reinforcement Learning)[https://arxiv.org/pdf/1703.03864.pdf]



Note: This is under development



## Getting Started:

- Create a virtual environment

  ```
  mkvirtualenv -p python3 <env-name>
  ```

- Install the dependencies

  ```
  pip install -r requirements.txt
  ```

- Run the algorithm

  ```
  python src/run.py
  ```

- Check help for a list of parameters

  ```
  python src/run.py --help
  ```

- Change the objective function and initial solution accordingly to suit your problem statement

  ```
  def fitness(solution):
      """Objective function for optimization"""
      # this is the objective that NES will try to maximize
      objective = ...
      return objective
  .
  .
  .
  
  # this is where you define your solution space
  init_solution = np.random.random(size=3)
  ```

  

## TODO:

- [x] Implement a basic version
- [ ] Try the basic solution on a toy problem
- [ ] Implement a distributed version
- [ ] Try the distributed version on a toy problem
- [ ] Try the distributed version on larger problem
