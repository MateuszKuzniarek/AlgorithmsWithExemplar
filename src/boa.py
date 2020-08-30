import operator

from deap import base, creator, tools
from numpy import random, math

from common import get_common_parser, display_and_save_results, generate_particle, get_function, \
    save_fitness_history, save_best_fitness_history, genetically_modify_exemplar


def parse_args():
    parser = get_common_parser()
    parser.add_argument("-c", "--sensorModality", type=float, default=0.3, help='sensor modality')
    parser.add_argument("-sp", "--switchProbability", type=float, default=0.6, help='switch probability')
    parser.add_argument("-ae", "--aExponent", type=float, default=0.3, help='a exponent')
    args = parser.parse_args()
    return args


def evaluate_butterflies(pop, toolbox):
    best = None
    for part in pop:
        if not part.best or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        part.fitness.values = toolbox.evaluate(part)
        if not best or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values

    return best


def set_fragrance(part, sensor_modality, a_exponent):
    part.fragrance = sensor_modality * (part.fitness.values[0] ** a_exponent)


def move_butterfly(part, vector1, vector2):
    random_number = random.rand()
    random_squared = random_number * random_number
    modification = [i * random_squared for i in vector1]
    modification = [modification[i] - vector2[i] for i in range(0, len(modification))]
    modification = [i * part.fragrance for i in modification]
    part[:] = list(map(operator.add, part, modification))


def move_towards_best(part, best):
    move_butterfly(part, best, part)


def move_randomly(part, random_part_1, random_part_2):
    move_butterfly(part, random_part_1, random_part_2)


def get_crossover_value(part):
    return part.best


def run_boa(args):
    toolbox = base.Toolbox()
    toolbox.register("particle", generate_particle, size=args.size, pmin=args.pminimum, pmax=args.pmaximum)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("evaluate", get_function(args.function))
    pop = toolbox.population(n=args.population)

    epoch = 0
    best_accuracy = float("inf")
    history = []
    best_history = []
    best = evaluate_butterflies(pop, toolbox)
    for part in pop:
        part.exemplar = random.uniform(low=args.pminimum, high=args.pmaximum, size=len(part))
        #part.exemplar = best

    while (args.epoch is None or epoch < args.epoch) and (args.accuracy is None or best_accuracy > args.accuracy):
        best_history.append(best.fitness.values[0])
        for part in pop:
            set_fragrance(part, args.sensorModality, args.aExponent)
        for part in pop:
            history.append([part.fitness.values[0], epoch])
            random_number = random.rand()

            genetically_modify_exemplar(part, pop, toolbox, best, args.mutationProbability, args.pminimum,
                                        args.pmaximum, args.stoppingGap, get_crossover_value)

            #update
            if random_number < args.switchProbability:
                move_towards_best(part, part.exemplar)
            else:
                random_index_1 = random.choice(len(pop))
                random_index_2 = random.choice(len(pop))
                move_randomly(part, pop[random_index_1], pop[random_index_2])

        best = evaluate_butterflies(pop, toolbox)
        best_value = toolbox.evaluate(best)[0]
        best_accuracy = abs(args.solution - best_value)
        epoch += 1
        #print(best_value)
    return epoch, best_value, best_accuracy, best_history, history


def main():
    args = parse_args()

    creator.create("Maximum", base.Fitness, weights=(1.0,))
    creator.create("Minimum", base.Fitness, weights=(-1.0,))
    fitness = creator.Minimum if args.minimum else creator.Maximum
    creator.create("Particle", list, exemplar=list, best=None, no_improvement_counter=0, fitness=fitness, fragrance=float)

    best_histories = []
    epochs = []
    accuracies = []
    best_values = []
    for i in range(args.iteration):
        result = run_boa(args)
        best_values.append(result[1])
        save_fitness_history("../results/" + args.logCatalog + "/", result[4])
        best_histories.append(result[3])
        accuracies.append(result[2])
        if accuracies[i] <= args.accuracy:
            epochs.append(result[0])

    save_best_fitness_history("../results/" + args.logCatalog + "/", best_histories)
    filtered_best_values = [value for value in best_values if not (math.isinf(value) or math.isnan(value))]
    display_and_save_results(epochs, filtered_best_values, accuracies,
                             args.accuracy, "../results/" + args.logCatalog + "/")


if __name__ == "__main__":
    main()
