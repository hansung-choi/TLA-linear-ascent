# TLA and linear ascent 

Implementations of main experiments for the paper "Deep Minimax Classifiers for Imbalanced Datasets with a Small Number of Minority Samples" published in IEEE Journal of Selected Topics in Signal Processing (https://ieeexplore.ieee.org/abstract/document/10908081).

## Requirements
1. python 3.8.5
2. pytorch 1.7.1
3. cuda 10.1
4. numpy 1.24.4
5. matplotlib 3.7.4
6. pillow 8.0.1

## Experiment code manual

### Arguments for terminal execution
1. s: simulation number (possible value: 1,2,3,4, and 5)
2. d: data name (possible value: CIFAR10 and CIFAR100)
3. t: imbalance type (possible value: LT and step)
4. r: imbalance ratio (possible value: 0.2, 0.1, and 0.01)
5. train_method: plain (Only SGD used), DRW, minimax
6. loss_function: CE, WCE, Focal, Focal_alpha, LDAM, LA, VS, GML, TWCE_EGA, TWCE_linear_ascent, TLA_EGA, and TLA_linear_ascent


From CE to GML: Use the 'plain' train method.

LDAM: The 'DRW' train method can be used.

From TWCE_EGA to TLA_linear_ascent: Use the 'minimax' train method.



### Example for an experiment of minimax approach

    python3 run_simulation.py -s 1 -d CIFAR10 -t step -r 0.01 --train_method minimax --loss_function TLA_linear_ascent


### Example for an experiment of imbalanced-data approach

    python3 run_simulation.py -s 1 -d CIFAR10 -t step -r 0.01 --train_method plain --loss_function VS

### Example for showing a specific simulation result

    python3 result_compare.py -s 1 -d CIFAR10 -t step -r 0.01 --train_method plain

### Example for showing a simulation summary result from simulation 1 to 5

    python3 result_compare.py -s 1 -d CIFAR10 -t step -r 0.01 --train_method minimax --summary 1
