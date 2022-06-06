# This script contains several functions used to warn the user if a function is misused
import warnings
import numpy as np

def Q1_warning(nb_games, eps, eps_opt, alpha, gamma, step, question, nb_samples, save):
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if eps < 0 or eps > 1:
        raise Exception("eps is a probability and should be taken in the interval [0, 1].")
    if eps_opt < 0 or eps_opt > 1:
        raise Exception("eps_opt is a probability and should be taken in the interval [0, 1].")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")
        
def Q2_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save):
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if eps_min < 0 or eps_min > eps_max:
        raise Exception("eps_min is a probability and should be taken in the interval [0, eps_max].")
    if eps_max < eps_min or eps_max > 1:
        raise Exception("eps_max is a probability and should be taken in the interval [eps_min, 1].")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")
        
def Q3_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save):
    if np.min(N_star) < 1:
        raise Exception("n* is supposed to be at least 1. Verify also you have entered integers.")
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if eps_min < 0 or eps_min > eps_max:
        raise Exception("eps_min is a probability and should be taken in the interval [0, eps_max].")
    if eps_max < eps_min or eps_max > 1:
        raise Exception("eps_max is a probability and should be taken in the interval [eps_min, 1].")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")
        
def Q4_warning(Eps_opt, n_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save):
    if np.min(Eps_opt) < 0 or np.max(Eps_opt) > 1:
        raise Exception("The elements of Eps_opt must be taken in [0, 1].")
    if n_star < 1:
        raise Exception("n* must be at least 1.")
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if eps_min < 0 or eps_min > eps_max:
        raise Exception("eps_min is a probability and should be taken in the interval [0, eps_max].")
    if eps_max < eps_min or eps_max > 1:
        raise Exception("eps_max is a probability and should be taken in the interval [eps_min, 1].")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")
        
def Q7_warning(Eps, nb_games, alpha, gamma, step, question, nb_samples, save):
    if np.min(Eps) < 0 or np.max(Eps) > 1:
        raise Exception("The elements of Eps must be taken in [0, 1].")
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")
        
def Q8_warning(N_star, nb_games, eps_min, eps_max, alpha, gamma, step, question, nb_samples, save):
    if np.min(N_star) < 1:
        raise Exception("The elements of N* must be at least 1.")
    if not isinstance(nb_games, int):
        raise Exception("The number of games should be an integer.")
    if not nb_games > 0:
        raise Exception("The number of games should be at least 1.")
    if eps_min < 0 or eps_min > eps_max:
        raise Exception("eps_min is a probability and should be taken in the interval [0, eps_max].")
    if eps_max < eps_min or eps_max > 1:
        raise Exception("eps_max is a probability and should be taken in the interval [eps_min, 1].")
    if alpha < 0:
        warnings.warn("Error for alpha: you entered a negative learning rate. Be sure that this is what you want.")
    if gamma < 0 or gamma > 1:
        raise Exception("Error for gamma: a discount factor should be taken in [0, 1]")
    if step < 0:
        raise Exception("The number of steps should be at least 1")
    if not isinstance(step, int):
        raise Exception("The number of steps should be an integer.")
    if not isinstance(question, str) and save:
        raise Exception("If you want to save you should specify a string for the argument question")
    if not isinstance(nb_samples, int):
        raise Exception("The number of samples has to be integer")
    if not nb_samples > 0:
        raise Exception("The number of samples should be at least 1")