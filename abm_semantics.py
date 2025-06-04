## imports ##

import pandas as pd 
import numpy as np 
import math 
import umap
import random
import mesa
import seaborn as sns
import numpy as np
# random base model 
# Import Cell Agent, OrthogonalMooreGrid and VoronoiGrid
from mesa.discrete_space import FixedAgent, CellAgent, OrthogonalMooreGrid, VoronoiGrid 

## function definitions ##
# "semantic scent" from Bader et al. (2024) cf. Zhang and Jones (2022)
def semantic_scent(cos_sims, current_item, neighbors, local_sensitivity):
    sim_sum = np.sum(cos_sims.loc[current_item,neighbors])
    p_switch = local_sensitivity / (local_sensitivity + sim_sum)
    return p_switch

# convert cosine similarity to angular distance 
def angular_dist(x):
    ad = np.arccos(x) / np.pi
    return ad

# check if it works
test = angular_dist(1)
test

# set random seed
random.seed(123)

## abm for semantic/ word search? : see flowchart.pdf and ABM-Pseudocode.pdf by Bader et al. (2024) ##

# Import FixedAgent, Cell Agent, OrthogonalMooreGrid classes
from mesa.discrete_space import FixedAgent, CellAgent, OrthogonalMooreGrid, VoronoiGrid 

# Build the model landscape: sample a number of words weighted by production frequency
animals = pd.read_csv('labels.csv', header=None)
frequencies = pd.read_csv('frequencies.csv', header=None)
cos_sims = pd.read_csv('similaritymatrix.csv', header=None)

#formatting lists
animal_itemlist = animals.values
animal = []
for i in animal_itemlist:
    for p in i:
        animal.append(p)
    
frequencies_list = frequencies.values
freq = []
for i in frequencies_list:
    for p in i:
        freq.append(p)


# Min/Max normalization to create sample probability weights
normfreaqs = (freq - np.min(freq)) / (np.max(freq) - np.min(freq))
print(normfreaqs)

# sample 30 words to create model landscape 
sampled_words = random.choices(animal, weights= normfreaqs, k = 30)
print(sampled_words)

# sample cosine similarities of sampled words
len(animal)
cos_sims.index = animal
cos_sims.columns = animal
cos_sims.head

sampled_cosims = cos_sims.loc[sampled_words,:]
sampled_cosims.head
sampled_cosims.shape


# sample angular distances for the sampled words
sampled_distances = sampled_cosims.applymap(angular_dist)
sampled_distances.head

# create 2-D map of sampled data with umap
reducer = umap.UMAP(random_state=123) 
dist_mapper = reducer.fit_transform(sampled_distances) 
dist_mapper.shape
dist_mapper[1:7,]

# convert to dataframe to label rows for sampling "word zero" of the model
dist_mapper_df = pd.DataFrame(dist_mapper, index = sampled_words)
dist_mapper_df.head

# maybe take the abolute values of each coordinate to stay in one quadrand? Does that make sense? 
dist_mapper_abs = np.abs(dist_mapper)

# convert to dataframe to label for sampling word zero
dist_mapper_df = pd.DataFrame(dist_mapper_abs, index = sampled_words)
dist_mapper_df.head


# format distance matrices for experimentation
# 1 as lists
dist_mapper_list_1 = []
for i in dist_mapper_abs:
    l = list(i)
    dist_mapper_list_1.append(l)

# 2 as tuple 
dist_mapper_list_2 = []
for i in dist_mapper_abs:
    l = [abs(int(p)) for p in i]
    l = tuple(l)
    dist_mapper_list_2.append(l)


# select initial agent position based on ln(freq)
frequencies.index = animal
sample_frequencies = frequencies.loc[sampled_words,:]
sample_f = []
for i in range(0,30):
    sample_f.append(int(sample_frequencies.iloc[i,0]))


# create a class that creates words as "fixed agents" on the grid (coordinates = 2-D angular distances) -> they don't change their positions
class WordAgentGrid(FixedAgent):
    """One word that does not change its position"""

    def __init__(self, model, cell, name, word_frequency):
        super().__init__(model)
        self.cell = cell  # Instantiate agent with location (x,y)
        self.name = name
        self.freq = word_frequency 

    def say_hi(self):
        # The agent's step will go here.
        print(f"Hi i am {self.name!s} and you find me {self.freq!s} times.")


# class that creates a moving agent with a step function that uses the semantic search algorithm
class WordSeeker(CellAgent):
    """One "wordseeker" that changes its position"""

    def __init__(self, model, cell, name, word_frequency, cos_sims):
        super().__init__(model)
        self.cell = cell  # Instantiate agent with location (x,y)
        self.name = name
        self.freq = word_frequency 
        self.cos_sims = cos_sims
        self.local_sensitivity = 0.6
        self.global_sensitivity = 0.8
        self.reported_words = []
    def report_start_word(self):
        # start word at each tick 
        print(self.name)

    def step(self) -> None:

        # get neighborhood of initial word 
        neighbors = list(self.cell.get_neighborhood(radius=1).agents)  # "radius as parameter: determine semantic neighborhood size dynamically?"
        neighbor_labels = []
        neighbor_freq = []
        neighbor_cells = []
        for i in neighbors:
            neighbor_labels.append(i.name)
            neighbor_freq.append(i.freq)
            neighbor_cells.append(i.cell)

        # any words nearby?
        if len(neighbors) > 0:   

            # local search mode 
            p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
            p_continue = 1 - p_switch
            last_scent = p_switch

            # stay in local search mode ?
            if p_continue > p_switch: 
                # scent probabilistic selection 
                next_cell = random.choices(neighbors, weights = neighbor_freq, k = 1)
                #print(next_cell[0].name)
                # move to selected word 
                self.cell = next_cell[0].cell
                self.name = next_cell[0].name
                self.freq = next_cell[0].freq
                # report selected word 
                if self.name in self.reported_words:
                    pass
                else:
                    print(self.name)
                self.reported_words.append(next_cell[0].name)
                # compute switching probability 
                neighbors = list(self.cell.get_neighborhood(radius=1).agents)  
                neighbor_labels = []
                neighbor_freq = []
                neighbor_cells = []
                for i in neighbors:
                    neighbor_labels.append(i.name)
                    neighbor_freq.append(i.freq)
                    neighbor_cells.append(i.cell)
                p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                p_continue = 1 - p_switch

            # continue in global search mode ? 
            elif p_continue < p_switch:
                # global search, is this correct ?
                self.cell = self.model.grid.select_random_empty_cell()
                neighbors = list(self.cell.get_neighborhood(radius=1).agents)
            
                neighbor_labels = []
                neighbor_freq = []
                neighbor_cells = []
                for i in neighbors:
                    neighbor_labels.append(i.name)
                    neighbor_freq.append(i.freq)
                    neighbor_cells.append(i.cell)

                # any words nearby in global search mode? 
                if len(neighbors) > 0:

                    # another round of searching
                    p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                    p_continue = 1 - p_switch
                    last_scent = p_switch
                    # go back to local search mode ? 
                    if p_switch > 0.5:

                        # scent probabilistic selection 
                        next_cell = random.choices(neighbors, weights = neighbor_freq, k = 1)
                        # move to selected word 
                        self.cell = next_cell[0].cell
                        self.name = next_cell[0].name
                        self.freq = next_cell[0].freq
                        self.reported_words.append(next_cell[0].name)
                        # report word
                        if self.name in self.reported_words:
                            pass
                        else: 
                            print(self.name)  
                        #compute switching probability of new local neighborhood 
                        neighbors = list(self.cell.get_neighborhood(radius=1).agents)
                    
                        neighbor_labels = []
                        neighbor_freq = []
                        neighbor_cells = []
                        for i in neighbors:
                            neighbor_labels.append(i.name)
                            neighbor_freq.append(i.freq)
                            neighbor_cells.append(i.cell)
                        
                        p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                        p_continue = 1 - p_switch

                    # continue in global search ?
                    if p_switch < p_continue:
                        # is current global better than last local? 
                        if p_switch > last_scent:
                            # move forward (? how ? next neighbor ?)
                            next_cell = neighbors[1]
                            # move to selected word 
                            self.cell = next_cell[0].cell
                            self.name = next_cell[0].name
                            self.freq = next_cell[0].freq
                            self.reported_words.append(next_cell[0].name)
                            if self.name in self.reported_words:
                                pass
                            else:
                                print(self.name)
                            last_scent = p_switch
                        else: 
                            # random walk
                            self.cell = self.model.grid.select_random_empty_cell()
                            #global search 
                            neighbors = list(self.cell.get_neighborhood(radius=1).agents)
            
                            neighbor_labels = []
                            neighbor_freq = []
                            neighbor_cells = []
                            for i in neighbors:
                                neighbor_labels.append(i.name)
                                neighbor_freq.append(i.freq)
                                neighbor_cells.append(i.cell)
                            #scent computation
                            p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                            p_continue = 1 - p_switch

        elif len(neighbors) == 0: # (if no neighbors are nearby, initially)  
            # global search immediately 
            self.cell = self.model.grid.select_random_empty_cell()
            neighbors = list(self.cell.get_neighborhood(radius=1).agents)
            neighbor_labels = []
            neighbor_freq = []
            neighbor_cells = []
            for i in neighbors:
                neighbor_labels.append(i.name)
                neighbor_freq.append(i.freq)
                neighbor_cells.append(i.cell)
            # any words nearby in global search mode? 
            if len(neighbors) > 0:
                # another round of searching
                p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                p_continue = 1 - p_switch
                last_scent = p_switch
                # go back to local search mode ? 
                if p_switch > p_continue:
                    # scent probabilistic selection 
                    next_cell = random.choices(neighbors, weights = neighbor_freq, k = 1)
                    # move to selected word 
                    self.cell = next_cell[0].cell
                    self.name = next_cell[0].name
                    self.freq = next_cell[0].freq
                    self.reported_words.append(next_cell[0].name)
                    # report word 
                    if self.name in self.reported_words:
                        pass
                    else:
                        print(self.name)
                    #compute switching probability of new local neighborhood 
                    neighbors = list(self.cell.get_neighborhood(radius=1).agents)
                
                    neighbor_labels = []
                    neighbor_freq = []
                    neighbor_cells = []
                    for i in neighbors:
                        neighbor_labels.append(i.name)
                        neighbor_freq.append(i.freq)
                        neighbor_cells.append(i.cell)
                    
                    p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                    p_continue = 1 - p_switch
                # continue in global search ?
                if p_switch < p_continue:
                    # is current global better than last local? 
                    if p_switch > last_scent:
                        # move forward (? how ? next neighbor ?)
                        next_cell = neighbors[1]
                        # move to selected word 
                        self.cell = next_cell[0].cell
                        self.name = next_cell[0].name
                        self.freq = next_cell[0].freq
                        last_scent = p_switch
                    else: 
                        # random walk
                        self.cell = self.model.grid.select_random_empty_cell()
                        #global search 
                        neighbors = list(self.cell.get_neighborhood(radius=1).agents)
        
                        neighbor_labels = []
                        neighbor_freq = []
                        neighbor_cells = []
                        for i in neighbors:
                            neighbor_labels.append(i.name)
                            neighbor_freq.append(i.freq)
                            neighbor_cells.append(i.cell)
                        #scent computation
                        p_switch = semantic_scent(self.cos_sims, self.name, neighbor_labels, self.local_sensitivity)
                        p_continue = 1 - p_switch

# Create the Model that places word cells on the grid and moves the wordseeker cell around 
class WordModelGrid(mesa.Model):
    """A model with some number of agents."""

    def __init__(self, n, word_ids, word_frequency, word_positions, cos_sims, sample_frequencies, seed=123):
        super().__init__(seed=seed)

        self.num_agents = n
        self.word_frequency = word_frequency
        self.cos_sims = cos_sims
        self.sample_frequencies = sample_frequencies
        self.word_positions = word_positions
        self.word_ids = word_ids
        # Landscape - What type of landscape makes sense? 
        # Initialize grid
        self.grid = OrthogonalMooreGrid((15, 15), random=random, capacity=5)
        # Create a fixed agent for each word and place them on the grid
        for cell in self.grid.all_cells:
            for idx, pos in enumerate(self.word_positions):
                if cell.coordinate == pos:
                    agent_name = self.word_ids[idx]
                    freq = self.word_frequency[idx]
                    fixed_agents = WordAgentGrid.create_agents(self, self.num_agents, cell, agent_name, freq)
        
        # Create a moving agent that starts at one word and moves according to some search function 
        random_idx = np.random.randint(1, 30) # replace with variable with number of sampled agents
        random_position = word_positions[random_idx]
        r_id = word_ids[random_idx]
        r_freq = word_frequency[random_idx]

        for cell in self.grid.all_cells: #there must be a better method to select a random word
                if cell.coordinate == random_position:
                    self.moving_agent = WordSeeker.create_agents(self, self.num_agents, cell, r_id, r_freq, cos_sims)

        #self.word_seeking_agent = random.choices(self.agents, k=1)
        #print(self.word_seeking_agent.name)

    def step(self):
        #self.agents.do("say_hi")
        self.moving_agent.do("step")

# simulate 1 participant
model = WordModelGrid(1, sampled_words, sample_f, dist_mapper_list_2, cos_sims, sample_frequencies)
for _ in range(60):
    model.step()

# simulate another participant
model = WordModelGrid(1, sampled_words, sample_f, dist_mapper_list_2, cos_sims, sample_frequencies)
for _ in range(60):
    model.step()

# check how many cells are there: 30 (N = fixed word samples) + 1 (one word seeker) 
print(f"You have {len(model.agents)} agents.")
