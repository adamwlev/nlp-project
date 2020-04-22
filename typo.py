# adapted from this https://stackoverflow.com/a/29590495/5552894

from math import sqrt
import numpy as np

keyboard_cartesian = {'q': {'x':0, 'y':0}, 'w': {'x':1, 'y':0}, 'e': {'x':2, 'y':0}, 'r': {'x':3, 'y':0}, 't': {'x':4, 'y':0}, 'y': {'x':5, 'y':0}, 'u': {'x':6, 'y':0}, 'i': {'x':7, 'y':0}, 'o': {'x':8, 'y':0}, 'p': {'x':9, 'y':0}, 'a': {'x':0, 'y':1},'z': {'x':0, 'y':2},'s': {'x':1, 'y':1},'x': {'x':1, 'y':2},'d': {'x':2, 'y':1},'c': {'x':2, 'y':2}, 'f': {'x':3, 'y':1}, 'b': {'x':4, 'y':2}, 'm': {'x':6, 'y':2}, 'j': {'x':6, 'y':1}, 'g': {'x':4, 'y':1}, 'h': {'x':5, 'y':1}, 'j': {'x':6, 'y':1}, 'k': {'x':7, 'y':1}, 'l': {'x':8, 'y':1}, 'v': {'x':3, 'y':2}, 'n': {'x':5, 'y':2}, }

def euclidean_distance_sq(a,b):
    X = (keyboard_cartesian[a]['x'] - keyboard_cartesian[b]['x'])**2
    Y = (keyboard_cartesian[a]['y'] - keyboard_cartesian[b]['y'])**2
    return X+Y

lookup = {}
for i in keyboard_cartesian.keys():
    lookup[i] = [[],[]]
    for j in keyboard_cartesian.keys():
        if i==j:
            continue
        dist = euclidean_distance_sq(i, j)
        if dist<=5:
            lookup[i][0].append(j)
            lookup[i][1].append(dist)
    lookup[i][1] = np.array(lookup[i][1])
    lookup[i][1] = 1/lookup[i][1]
    lookup[i][1] = lookup[i][1]/lookup[i][1].sum()

def sample(letter,preserve_case):
    if letter.lower() not in lookup:
        return letter
    if preserve_case and letter.isupper():
        upper = True
    else:
        upper = False
    letter = letter.lower()
    probs = lookup[letter][1]
    ind = np.random.choice(len(probs),p=probs)
    new_letter = lookup[letter][0][ind]
    if upper:
        return new_letter.upper()
    else:
        return new_letter

def get_typo(word,preserve_case=True):
    if len(word)<=2:
        return word
    p = list(range(1,int(min(len(word)+1,4)))) + [5 for _ in range(len(word)-3)]
    p = np.array(p,dtype=np.float)
    p /= p.sum()
    typo_ind = np.random.choice(len(word),p=p)
    new_word = word[:typo_ind] + sample(word[typo_ind],preserve_case) + word[typo_ind+1:]
    return new_word
