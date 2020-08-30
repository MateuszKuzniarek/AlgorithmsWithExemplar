import operator
import random
import argparse
import numpy

from deap import base
from deap import creator
from deap import tools
from deap import benchmarks

from common import get_common_parser, get_function, generate_particle, set_creator, save_fitness_history, \
    save_best_fitness_history, display_and_save_results, genetically_modify_exemplar


def get_crossover_value(part):
    return part.best


def update_particle(part, best, weight, acceleration_factor, pop, mutation_probability,
                    minimum, maximum, stopping_gap, toolbox):
    genetically_modify_exemplar(part, pop, toolbox, best, mutation_probability, minimum, maximum, stopping_gap,
                                get_crossover_value)

    #update
    for d in range(0, len(part)):
        part.speed[d] = weight * part.speed[d] + \
                        acceleration_factor * random.uniform(0, 1) * (part.exemplar[d] - part[d])
        part[d] += part.speed[d]


def get_pso_parameters(args):
    weight = random.uniform(0.1, 1) if args.randomWeight else args.weight
    c = random.uniform(0.5, 1.5) if args.randomAccelerationFactor else args.accelerationFactor
    return weight, c


def get_toolbox(size, pminimum, pmaximum, function):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle, size=size, pmin=pminimum, pmax=pmaximum)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle)
    toolbox.register("evaluate", get_function(function))
    return toolbox


def run(args):
    toolbox = get_toolbox(args.size, args.pminimum, args.pmaximum, args.function)
    pop = toolbox.population(n=args.population)

    solution = args.solution
    best = None
    best_value = None
    best_history = []
    history = []
    epoch = 0
    accuracy = float("inf")

    while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or accuracy > args.accuracy):
        weight, c = get_pso_parameters(args)

        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            history.append([part.fitness.values[0], epoch])
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        for part in pop:
            toolbox.update(part, best, weight, c, pop, args.mutationProbability,
                           args.pminimum, args.pmaximum, args.stoppingGap, toolbox)

        best_value = toolbox.evaluate(best)[0]
        best_history.append(best_value)
        epoch += 1
        accuracy = abs(solution - best_value)
        #print(best_value)

    return epoch, best_value, accuracy, best_history, history


def main():
    parser = get_common_parser()
    weighOptions = parser.add_mutually_exclusive_group(required=True)
    weighOptions.add_argument("-w", "--weight", type=float, help='constant inertial weight')
    weighOptions.add_argument("-rw", "--randomWeight", action="store_true", help='random inertial weight')

    accelerationOptions = parser.add_mutually_exclusive_group(required=True)
    accelerationOptions.add_argument("-c", "--accelerationFactor", type=float, help='acceleration factor')
    accelerationOptions.add_argument("-rc", "--randomAccelerationFactor", action="store_true",
                                     help='random acceleration factor')
    args = parser.parse_args()

    set_creator(args.minimum)
    best_histories = []
    epochs = []
    accuracies = []
    best_values = []
    for i in range(args.iteration):
        result = run(args)
        best_values.append(result[1])
        save_fitness_history("../results/" + args.logCatalog + "/", result[4])
        best_histories.append(result[3])
        accuracies.append(result[2])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0])

    save_best_fitness_history("../results/" + args.logCatalog + "/", best_histories)
    display_and_save_results(epochs, best_values, accuracies, args.accuracy, "../results/" + args.logCatalog + "/")


if __name__ == "__main__":
    main()
