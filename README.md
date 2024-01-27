# NEAT
NeuroEvolution of Augmenting Topologies

## Description
- The code is written according to my own understanding of the paper.
- The project is still in progress.

## Usage
### Solving XOR Problem 
type ```python TestXOR.py```

```
usage: TestXOR.py [-h] [--pop POP] [--gen GEN] [--thr THR] [--cmp CMP]
                  [--mat MAT] [--cpy CPY] [--slf SLF] [--exc EXC] [--dsj DSJ]
                  [--wgh WGH] [--srv SRV]

Change the evolutionary parameters.

optional arguments:
  -h, --help  show this help message and exit
  --pop POP   The initial population size.
  --gen GEN   The maximum generations.
  --thr THR   The compatibility threshold.
  --cmp CMP   The number of genomes used to compare compatibility.
  --mat MAT   The mating probability.
  --cpy CPY   The copy mutation probability.
  --slf SLF   The self mutation probability.
  --exc EXC   The excess weight.
  --dsj DSJ   The disjoint weight.
  --wgh WGH   The average weight differences weight.
  --srv SRV   The number of survivors per generation.
```

Finally, the resulting network structure will look like this:
```
Genome 542(fitness = 4.00):
	Total Nodes:5	Hidden Nodes:1
	Enabled Connections(7):
		[Input Node 1] = 1.00	**[ -4.53]**	[Output Node 3] = 0.46	Enable = True	Innovation = 0
		[Input Node 2] = 1.00	**[ -1.87]**	[Output Node 3] = 0.46	Enable = True	Innovation = 1
		[Bias Node 0] = 1.00	**[ -2.07]**	[Output Node 3] = 0.46	Enable = True	Innovation = 2
		[Bias Node 0] = 1.00	**[ -3.85]**	[Hidden Node 4] = 1.00	Enable = True	Innovation = 3
		[Hidden Node 4] = 1.00 **[  8.32]**	[Output Node 3] = 0.46	Enable = True	Innovation = 4
		[Input Node 1] = 1.00	**[  7.42]**	[Hidden Node 4] = 1.00	Enable = True	Innovation = 5
		[Input Node 2] = 1.00	**[  5.19]**	[Hidden Node 4] = 1.00	Enable = True	Innovation = 6
```

### Playing Tictactoe game
type ```python TestTictactoe.py```

The AI can learn how to play tictactoe from scratch, but it's very slow to converge.


## Paper Link
[Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
