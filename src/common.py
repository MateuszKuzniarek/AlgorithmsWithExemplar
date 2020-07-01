import argparse
import csv
import random
import operator
from pathlib import Path

import numpy

from deap import creator, tools
from deap import benchmarks
from deap import base


def get_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--iteration", type=int, default=1)
    parser.add_argument("-n", "--population", type=int, default=10, help='number of parts')
    parser.add_argument("-N", "--size", type=int, default=2, help='size')
    parser.add_argument("-f", "--function", type=str, required=True, help='selected function')
    parser.add_argument("-pmin", "--pminimum", type=float, default=-100.0, help='partition minimum')
    parser.add_argument("-pmax", "--pmaximum", type=float, default=100.0, help='partition maximum')
    parser.add_argument("-s", "--solution", type=float, default=0.0, help='expected solution (for -a stop condition)')
    parser.add_argument("-ss", "--subswarms", type=int, default=3, help='number of sub-swarms')
    parser.add_argument("-sg", "--stoppingGap", type=int, default=5, help='stopping gap')
    parser.add_argument("-mp", "--mutationProbability", type=float, default=0.3, help='mutation probability')
    parser.add_argument("-l", "--logCatalog", type=str, default='logs', help='number of sub-swarms')

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("-min", "--minimum", action="store_true", help='minimum searching mode')
    mode.add_argument("-max", "--maximum", action="store_true", help='maximum searching mode')

    parser.add_argument("-e", "--epoch", type=int, help='number of epoch')
    parser.add_argument("-a", "--accuracy", type=float, help='expected accuracy')

    return parser


def generate_particle(size, pmin, pmax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [0 for _ in range(size)]
    return part


def display_and_save_results(epochs, best_values, accuracies, expected_accuracy, path):
    Path(path).mkdir(parents=True, exist_ok=True)
    fitness_file = open(path + 'measures.csv', 'w', newline='')
    with fitness_file:
        writer = csv.writer(fitness_file)
        print(epochs)
        is_solution_found_list = list(map(lambda acc: acc <= expected_accuracy, accuracies))
        print(is_solution_found_list)
        found_solutions_count = is_solution_found_list.count(True)
        writer.writerow(('found_solutions_count', found_solutions_count))
        print(found_solutions_count)
        writer.writerow(('average best value', numpy.average(best_values)))
        writer.writerow(('standard deviation', numpy.std(best_values)))
        if len(epochs) > 0:
            average_records = numpy.average(epochs)
            print(average_records)
            writer.writerow(('mean records', average_records))
        else:
            print("-")


def set_creator(is_minimum):
    creator.create("Maximum", base.Fitness, weights=(1.0,))
    creator.create("Minimum", base.Fitness, weights=(-1.0,))
    fitness = creator.Minimum if is_minimum else creator.Maximum
    creator.create("Particle", list, fitness=fitness, speed=list, best=None, exemplar=list, no_improvement_counter=0)


def save_fitness_history(path, data):
    Path(path).mkdir(parents=True, exist_ok=True)
    fitness_file = open(path + 'fitness.csv', 'w', newline='')
    with fitness_file:
        writer = csv.writer(fitness_file)
        writer.writerow(('fitness', 'epoch'))
        for row in data:
            writer.writerow(row)


def save_best_fitness_history(path, best_histories):
    Path(path).mkdir(parents=True, exist_ok=True)
    fitness_file = open(path + 'best_fitness.csv', 'w', newline='')
    with fitness_file:
        writer = csv.writer(fitness_file)
        writer.writerow(('fitness', 'epoch'))
        min_length = max(map(len, best_histories))
        for i in range(0, min_length):
            fitness_sum = 0
            number_of_runs = 0
            for j in range(0, len(best_histories)):
                if i < len(best_histories[j]) and not numpy.isnan(best_histories[j][i]):
                    number_of_runs += 1
                    fitness_sum += best_histories[j][i]
            if number_of_runs != 0:
                writer.writerow((fitness_sum/number_of_runs, i))


def get_function(function_name):
    switch = {
        'sphere': benchmarks.sphere,
        'griewank': benchmarks.griewank
    }

    return switch.get(function_name, benchmarks.sphere)
